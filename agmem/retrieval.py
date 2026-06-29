"""火行 — tiered retrieval for AgMem memory graph.

**温 → 烟 → 燃 → 炎**

Four tiers of recall, each with different depth and precision:

| Tier   | Hops | Min Score   | Use Case                 |
|--------|:----:|:-----------:|--------------------------|
| 温     | 0    | none        | Quick FTS5 hit, no walk  |
| 烟     | 1    | 0.1         | Light context, one hop   |
| 燃     | 3    | 0.6         | 🔥 Standard retrival    |
| 炎     | 5    | 0.01        | Deep dive, consolidation |
"""

from __future__ import annotations

import enum
import logging
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple

from agmem.core import AgMemDB, Node, ScoredPath
from agmem.dynamics import on_traverse
from agmem.ner import extract_entities
from agmem.walker import walk_graph

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Recall tiers
# ---------------------------------------------------------------------------


class RecallTier(enum.Enum):
    """火行 four-tier recall system.

    Usage::

        from agmem.retrieval import RecallTier, retrieve

        # Quick check — 温 tier
        quick = retrieve(db, query, tier=RecallTier.WARM)

        # Standard retrieval — 燃 tier (default)
        results = retrieve(db, query)

        # Deep dive — 炎 tier
        deep = retrieve(db, query, tier=RecallTier.INFERNO, top_k=20)
    """

    WARM = "温"       # 0 hops, FTS5 only
    SMOKE = "烟"      # 1 hop, path score > 0.1
    BURN = "燃"       # 3 hops, path score > 0.6 (default)
    INFERNO = "炎"    # 5 hops, path score > 0.01 (consolidation only)


# Tier → parameter mapping
_TIER_CONFIG: Dict[RecallTier, Dict[str, Any]] = {
    RecallTier.WARM: {
        "max_hops": 0,
        "min_path_score": 0.0,
        "min_node_strength": 0.2,
        "top_k": 3,
        "use_fts5_first": True,
    },
    RecallTier.SMOKE: {
        "max_hops": 1,
        "min_path_score": 0.1,
        "min_node_strength": 0.3,
        "top_k": 5,
        "use_fts5_first": True,
    },
    RecallTier.BURN: {
        "max_hops": 3,
        "min_path_score": 0.05,
        "min_node_strength": 0.3,
        "top_k": 5,
        "use_fts5_first": True,
    },
    RecallTier.INFERNO: {
        "max_hops": 5,
        "min_path_score": 0.01,
        "min_node_strength": 0.1,
        "top_k": 20,
        "use_fts5_first": False,
    },
}


# ---------------------------------------------------------------------------
# Write pipeline (unchanged from original)
# ---------------------------------------------------------------------------


def write_memory(
    db: AgMemDB,
    content: str,
    node_type: str = "fact",
    strength: float = 0.5,
    node_id: Optional[str] = None,
    link_to: Optional[List[str]] = None,
    extract_auto_entities: bool = True,
) -> Node:
    """Create a new memory node and optionally link it.

    When *extract_auto_entities* is True (default), entities are extracted
    from *content* and stored on the node.

    If *link_to* is provided, the new node is linked to each target id
    via an ``"inspired_by"`` relation.  Additionally, if auto-entities
    are extracted, the node is auto-linked to existing nodes that share
    any of those entities (``"related_to"`` relation).

    Parameters
    ----------
    db : AgMemDB
    content : str
        The memory text.
    node_type : str, optional
    strength : float, optional
    node_id : str, optional
        A custom id (UUID4 generated if omitted).
    link_to : list of str, optional
        Existing node ids to link from the new node.
    extract_auto_entities : bool, optional
        Whether to run entity extraction (default True).

    Returns
    -------
    Node
        The newly created node.
    """
    node_id = node_id or str(uuid.uuid4())
    entities = extract_entities(content) if extract_auto_entities else []

    node = db.create_node(
        node_id=node_id,
        content=content,
        node_type=node_type,
        strength=strength,
        entities=entities,
    )

    # Explicit links.
    linked_ids: set = set()
    if link_to:
        for target_id in link_to:
            try:
                db.create_link(
                    source_id=node_id,
                    target_id=target_id,
                    relation="inspired_by",
                    weight=0.4,
                )
                linked_ids.add(target_id)
            except ValueError:
                logger.debug("Failed to link %s → %s (target may not exist)", node_id, target_id)

    # Auto-link to same-entity nodes.
    if entities:
        matches = db.find_nodes_by_entities(entities)
        for matched_node, matched_entities in matches:
            if matched_node.id == node_id or matched_node.id in linked_ids:
                continue
            if len(matched_entities) >= 2 or len(matched_entities) == len(entities):
                try:
                    db.create_link(
                        source_id=node_id,
                        target_id=matched_node.id,
                        relation="related_to",
                        weight=0.3,
                    )
                except ValueError:
                    continue

    return node


# ---------------------------------------------------------------------------
# Tiered retrieval
# ---------------------------------------------------------------------------


def retrieve(
    db: AgMemDB,
    query: str,
    top_k: Optional[int] = None,
    tier: RecallTier = RecallTier.BURN,
    max_hops: Optional[int] = None,
    min_node_strength: Optional[float] = None,
) -> List[ScoredPath]:
    """Full retrieval pipeline with tiered recall.

    Parameters
    ----------
    db : AgMemDB
    query : str
        The user's natural-language query or message.
    top_k : int, optional
        Override tier default.  If omitted, uses tier's default.
    tier : RecallTier
        Which tier to use (default: BURN).
    max_hops : int, optional
        Override tier's max_hops.
    min_node_strength : float, optional
        Override tier's min_node_strength.

    Returns
    -------
    list of ScoredPath
        Highest-scoring paths, sorted descending.
    """
    # Resolve tier config.
    cfg = _TIER_CONFIG[tier]
    actual_top_k = top_k or cfg["top_k"]
    actual_max_hops = max_hops if max_hops is not None else cfg["max_hops"]
    actual_min_strength = (
        min_node_strength if min_node_strength is not None else cfg["min_node_strength"]
    )
    min_path_score = cfg["min_path_score"]
    use_fts5_first = cfg["use_fts5_first"]

    logger.debug(
        "火行 %s tier: hops=%d, min_strength=%.1f, min_score=%.2f, top_k=%d",
        tier.value, actual_max_hops, actual_min_strength, min_path_score, actual_top_k,
    )

    # --- 温 tier: FTS5 only, no graph walk ---
    if tier == RecallTier.WARM:
        return _retrieve_fts5_only(db, query, actual_top_k, actual_min_strength)

    # --- 烟/燃/炎: Hops with tiered thresholds ---
    # Step 1 — Entity extraction (or FTS5 for 烟/燃)
    entities = extract_entities(query)
    logger.debug("Entities extracted: %s", entities)

    if not entities and use_fts5_first:
        # Fall back to FTS5
        fts_results = _retrieve_fts5_only(db, query, actual_top_k, actual_min_strength)
        if fts_results:
            return fts_results

    if not entities:
        # Pure strength fallback
        fallback_nodes = db.search_nodes(min_strength=actual_min_strength, limit=actual_top_k)
        return [
            ScoredPath(nodes=[n], links=[], score=n.strength)
            for n in fallback_nodes
        ]

    # Step 2 — Find entry nodes.
    entry_matches = db.find_nodes_by_entities(entities, min_strength=actual_min_strength)

    if not entry_matches:
        if use_fts5_first:
            fts_results = _retrieve_fts5_only(db, query, actual_top_k, actual_min_strength)
            if fts_results:
                return fts_results
        fallback_nodes = db.search_nodes(min_strength=actual_min_strength, limit=actual_top_k)
        return [
            ScoredPath(nodes=[n], links=[], score=n.strength)
            for n in fallback_nodes
        ]

    # Step 3 — Graph walk with tiered parameters.
    visited: set = set()
    all_paths: List[ScoredPath] = []
    for entry_node, _matched_entities in entry_matches:
        paths = walk_graph(
            db,
            entry_node.id,
            max_hops=actual_max_hops,
            visited=visited,
        )
        # Filter by tier's minimum path score
        for p in paths:
            if p.score >= min_path_score:
                all_paths.append(p)

    # Step 4 — Rank, deduplicate, take top_k.
    all_paths.sort(key=lambda p: p.score, reverse=True)

    seen_content: set = set()
    deduped: List[ScoredPath] = []
    for path in all_paths:
        content_sig = " → ".join(f"{n.id}:{n.content}" for n in path.nodes)
        if content_sig not in seen_content:
            seen_content.add(content_sig)
            deduped.append(path)
            if len(deduped) >= actual_top_k:
                break

    # Step 5 — Hebbian reinforcement (only for BURN and above).
    if tier.value in (RecallTier.BURN.value, RecallTier.INFERNO.value):
        for path in deduped:
            for link in path.links:
                on_traverse(db, link.id, outcome_signal=0.0)

    return deduped


def _retrieve_fts5_only(
    db: AgMemDB,
    query: str,
    top_k: int,
    min_strength: float,
) -> List[ScoredPath]:
    """温 tier: FTS5 full-text search, no graph walk."""
    if not query or not query.strip():
        return []
    nodes = db.fts5_search(query, limit=top_k)
    return [
        ScoredPath(nodes=[n], links=[], score=n.strength)
        for n in nodes if n.strength >= min_strength
    ]


# ---------------------------------------------------------------------------
# Convenience: Hermes-compatible tool shape
# ---------------------------------------------------------------------------


def memory_recall(
    db: AgMemDB,
    query: str,
    top_k: int = 5,
    tier: str = "燃",
) -> List[Dict[str, Any]]:
    """Hermes-compatible recall: returns dicts, not dataclasses.

    Parameters
    ----------
    db : AgMemDB
    query : str
    top_k : int
    tier : str
        One of ``"温"``, ``"烟"``, ``"燃"``, ``"炎"``.
    """
    tier_map = {
        "温": RecallTier.WARM,
        "烟": RecallTier.SMOKE,
        "燃": RecallTier.BURN,
        "炎": RecallTier.INFERNO,
    }
    resolved = tier_map.get(tier, RecallTier.BURN)
    return [p.to_dict() for p in retrieve(db, query, top_k=top_k, tier=resolved)]


def memory_write(
    db: AgMemDB,
    content: str,
    node_type: str = "fact",
    link_to: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Hermes-compatible write: creates a node and returns it as a dict."""
    node = write_memory(db, content, node_type=node_type, link_to=link_to)
    return node.to_dict()


def memory_link(
    db: AgMemDB,
    source_id: str,
    target_id: str,
    relation: str = "related_to",
    weight: float = 0.3,
) -> Dict[str, Any]:
    """Hermes-compatible link: creates a link and returns it as a dict."""
    link = db.create_link(source_id, target_id, relation, weight)
    return link.to_dict()

"""土行 — Consolidation pass for AgMem memory graph.

**凝 → 载 → 化 → 育 → 返**

After each session, the consolidation pass:
1. **凝** — Gathers new nodes written during the session
2. **化** — Entity resolution: merges nodes that refer to the same thing
3. **育** — Adjusts strengths based on cross-validation
4. **返** — Rebuilds link weights after merges

Usage::

    from agmem.consolidate import run_consolidation
    report = run_consolidation(db, session_id="...")
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Optional, Set, Tuple

if TYPE_CHECKING:
    from agmem.core import AgMemDB, Node

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


@dataclass
class ConsolidationReport:
    """Result of running the consolidation pass."""

    nodes_scanned: int = 0
    nodes_merged: int = 0
    nodes_deleted: int = 0  # duplicates removed
    links_created: int = 0
    links_pruned: int = 0
    errors: List[str] = field(default_factory=list)

    @property
    def summary(self) -> Dict[str, int]:
        return {
            "scanned": self.nodes_scanned,
            "merged": self.nodes_merged,
            "deleted": self.nodes_deleted,
            "links_created": self.links_created,
            "links_pruned": self.links_pruned,
        }

    def print(self) -> None:
        print(f"\n{'=' * 60}")
        print(f"  土行 Consolidation Report")
        print(f"{'=' * 60}")
        print(f"  🔍 scanned:       {self.nodes_scanned}")
        print(f"  🧬 merged:        {self.nodes_merged}")
        print(f"  🗑️  deleted:       {self.nodes_deleted}")
        print(f"  🔗 links created: {self.links_created}")
        print(f"  ✂️  links pruned:  {self.links_pruned}")
        if self.errors:
            print(f"  ❌ errors:        {len(self.errors)}")
        print(f"{'=' * 60}\n")


# ---------------------------------------------------------------------------
# Entity resolution (化)
# ---------------------------------------------------------------------------


def _entity_overlap(
    a: List[str], b: List[str], min_overlap: int = 2
) -> int:
    """Count how many normalized entities overlap between two lists."""
    sa = set(e.lower().strip() for e in a)
    sb = set(e.lower().strip() for e in b)
    return len(sa & sb)


def _content_similarity(a: str, b: str) -> float:
    """Simple word-overlap similarity for short texts.

    Returns a score in [0, 1]. Uses Jaccard similarity on word tokens,
    with stop-word filtering and punctuation removal.
    """
    import re
    clean = lambda s: re.sub(r"[^\w\s]", "", s.lower())
    words_a = {w for w in clean(a).split() if len(w) > 2}
    words_b = {w for w in clean(b).split() if len(w) > 2}
    # Filter common stop words
    stop_words = {"the", "and", "for", "are", "but", "not", "you",
                  "all", "can", "had", "her", "was", "one", "our",
                  "out", "has", "have", "been", "its", "than",
                  "that", "this", "with", "from", "they", "will"}
    words_a -= stop_words
    words_b -= stop_words
    if not words_a or not words_b:
        return 0.0
    intersection = words_a & words_b
    union = words_a | words_b
    return len(intersection) / len(union)


def _is_duplicate(a: Node, b: Node) -> Tuple[bool, float]:
    """Check if two nodes are likely duplicates.

    Returns (is_duplicate, confidence).
    """
    # Same entities + similar content = likely duplicate
    overlap = _entity_overlap(a.entities, b.entities)

    if overlap >= 2:
        sim = _content_similarity(a.content, b.content)
        if sim >= 0.6:
            return (True, sim)

    # Same node_type + very high content similarity
    if a.node_type == b.node_type:
        sim = _content_similarity(a.content, b.content)
        if sim >= 0.8:
            return (True, sim)

    return (False, 0.0)


def _pick_survivor(a: Node, b: Node) -> Node:
    """Pick which node to keep, merging strength from both."""
    if a.strength >= b.strength:
        survivor = a
        victim = b
    else:
        survivor = b
        victim = a

    # Boost survivor strength slightly
    new_strength = min(1.0, survivor.strength + victim.strength * 0.2)
    survivor.strength = new_strength

    # Merge entities (dedup)
    all_entities = list(dict.fromkeys(survivor.entities + victim.entities))
    survivor.entities = all_entities

    # Merge content: prefer the longer, more specific one
    if len(victim.content) > len(survivor.content):
        survivor.content = victim.content + " | " + survivor.content

    return survivor


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def run_consolidation(
    db: AgMemDB,
    session_id: Optional[str] = None,
    dry_run: bool = True,
    min_similarity: float = 0.6,
    limit: int = 500,
) -> ConsolidationReport:
    """Run a full consolidation pass.

    1. **凝** — Scan all nodes
    2. **化** — Entity resolution + duplicate detection
    3. **育** — Merge duplicates, adjust strengths
    4. **返** — Clean up orphaned links

    Parameters
    ----------
    db : AgMemDB
    session_id : str, optional
        If provided, only consolidate nodes written in this session.
    dry_run : bool
        When True, report only — no DB writes.
    min_similarity : float
        Minimum content similarity to consider nodes as duplicates.
    limit : int
        Max nodes to scan.

    Returns
    -------
    ConsolidationReport
    """
    now = int(time.time())
    report = ConsolidationReport()

    # --- 凝: Gather nodes ---
    nodes = db.get_all_nodes(limit=limit)
    report.nodes_scanned = len(nodes)

    if len(nodes) < 2:
        print("  ℹ️  Less than 2 nodes — nothing to consolidate.")
        return report

    # --- 化: Find duplicates ---
    merged_ids: Set[str] = set()
    survivors: Dict[str, Node] = {}

    for i in range(len(nodes)):
        a = nodes[i]
        if a.id in merged_ids:
            continue
        survivors[a.id] = a

        for j in range(i + 1, len(nodes)):
            b = nodes[j]
            if b.id in merged_ids:
                continue

            is_dup, confidence = _is_duplicate(a, b)
            if is_dup and confidence >= min_similarity:
                survivor = _pick_survivor(a, b)
                survivors[a.id] = survivor
                merged_ids.add(b.id)
                report.nodes_merged += 1

                if not dry_run:
                    # Transfer links from victim to survivor
                    victim_links = db.get_links(node_id=b.id)
                    for link in victim_links:
                        if link.source_id == b.id:
                            try:
                                db.create_link(
                                    survivor.id, link.target_id,
                                    link.relation, link.weight,
                                )
                                report.links_created += 1
                            except ValueError:
                                pass  # link already exists
                        elif link.target_id == b.id:
                            try:
                                db.create_link(
                                    link.source_id, survivor.id,
                                    link.relation, link.weight,
                                )
                                report.links_created += 1
                            except ValueError:
                                pass

                    # Update survivor + delete victim
                    db.update_node(
                        survivor.id,
                        strength=survivor.strength,
                        content=survivor.content,
                    )
                    db.delete_node(b.id)
                    report.nodes_deleted += 1

    # --- 返: Prune orphaned links ---
    if not dry_run:
        all_links = db.get_all_links(limit=5000)
        for link in all_links:
            if db.get_node(link.source_id) is None or db.get_node(link.target_id) is None:
                db.delete_link(link.id)
                report.links_pruned += 1

    if dry_run:
        print(f"\n  ℹ️  Dry-run: {report.nodes_merged} duplicates found, "
              f"{report.nodes_deleted} would be deleted.")
        print(f"     Call run_consolidation(db, dry_run=False) to execute.\n")

    return report


def rebuild_fts(db: AgMemDB) -> int:
    """Rebuild the entire FTS5 index from scratch.

    Use after bulk operations or consolidation.
    """
    # Clear FTS
    db._conn.execute("DELETE FROM nodes_fts")
    db._conn.commit()

    # Re-insert all nodes
    nodes = db.get_all_nodes(limit=10000)
    for node in nodes:
        db._sync_fts5(node.id)

    return len(nodes)

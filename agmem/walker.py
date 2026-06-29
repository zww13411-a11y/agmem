"""Graph-walk retrieval for the AgMem memory graph.

Performs breadth-limited traversal from entry nodes, collects candidate
paths, and scores them by link-weight product, target-node strength,
and hop-distance decay.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Set, Tuple

from agmem.core import AgMemDB, Link, Node, ScoredPath

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Scoring constants
# ---------------------------------------------------------------------------

HOP_DECAY_FACTOR: float = 0.85
"""Discount factor applied per hop: ``score *= HOP_DECAY_FACTOR ** hop``."""

MIN_LINK_WEIGHT: float = 0.05
"""Skip links whose weight is below this threshold during traversal."""

MIN_PATH_SCORE: float = 0.01
"""Paths with a composite score below this are dropped."""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def walk_graph(
    db: AgMemDB,
    entry_node_id: str,
    max_hops: int = 3,
    *,
    min_link_weight: float = MIN_LINK_WEIGHT,
    visited: Optional[Set[str]] = None,
) -> List[ScoredPath]:
    """Breadth-limited walk from *entry_node_id*, returning all reachable paths.

    Each returned ``ScoredPath`` includes the full node and link chain so
    callers can inspect *how* a memory was reached.

    Parameters
    ----------
    db : AgMemDB
    entry_node_id : str
        Starting node for the walk.
    max_hops : int
        Maximum number of link traversals from the entry node.
    min_link_weight : float
        Skip links below this weight.
    visited : set of str, optional
        External visited set used to avoid re-visiting nodes across
        multiple walks.  A new one is created if not provided.

    Returns
    -------
    list of ScoredPath
        Paths sorted descending by score.
    """
    if visited is None:
        visited = {entry_node_id}

    entry = db.get_node(entry_node_id)
    if entry is None:
        return []

    paths: List[ScoredPath] = [
        ScoredPath(nodes=[entry], links=[], score=entry.strength)
    ]

    # BFS-like expansion.
    for _ in range(max_hops):
        new_paths: List[ScoredPath] = []
        for path in paths:
            tip = path.nodes[-1]
            outgoing = db.get_links(
                node_id=tip.id,
                min_weight=min_link_weight,
                direction="outgoing",
            )
            for link in outgoing:
                neighbor = db.get_node(link.target_id)
                if neighbor is None or neighbor.id in visited:
                    continue
                visited.add(neighbor.id)
                score = _compute_path_score(path, link, neighbor)
                if score < MIN_PATH_SCORE:
                    continue
                new_paths.append(
                    ScoredPath(
                        nodes=[*path.nodes, neighbor],
                        links=[*path.links, link],
                        score=score,
                    )
                )

            # Also traverse incoming links backwards (undirected fallback).
            incoming = db.get_links(
                node_id=tip.id,
                min_weight=min_link_weight,
                direction="incoming",
            )
            for link in incoming:
                neighbor_id = link.source_id
                if neighbor_id in visited:
                    continue
                neighbor = db.get_node(neighbor_id)
                if neighbor is None:
                    continue
                visited.add(neighbor_id)
                score = _compute_path_score(path, link, neighbor)
                if score < MIN_PATH_SCORE:
                    continue
                new_paths.append(
                    ScoredPath(
                        nodes=[*path.nodes, neighbor],
                        links=[*path.links, link],
                        score=score,
                    )
                )

        if not new_paths:
            break
        paths.extend(new_paths)

    paths.sort(key=lambda p: p.score, reverse=True)
    return paths


def score_path(
    nodes: List[Node],
    links: List[Link],
    *,
    hop_decay: float = HOP_DECAY_FACTOR,
) -> float:
    """Compute a composite score for a node-and-link chain.

    Formula::

        score = ∏(node_i.strength) × ∏(link.weight) × hop_decay ** hops

    where ``hops`` is the number of link traversals, consistent with the
    incremental scoring used during graph walk.
    """
    if not nodes:
        return 0.0

    score = nodes[0].strength
    for i, link in enumerate(links):
        target_node = nodes[i + 1]
        score *= link.weight * target_node.strength * hop_decay

    return score


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _compute_path_score(
    prefix: ScoredPath,
    tail_link: Optional[Link],
    tail_node: Optional[Node],
    *,
    hop_decay: float = HOP_DECAY_FACTOR,
) -> float:
    """Score of ``prefix + (tail_link, tail_node)``."""
    score = prefix.score

    if tail_link is not None:
        score *= tail_link.weight
    if tail_node is not None:
        score *= tail_node.strength

    # Apply hop decay once per new link (not hop_decay^total_hops,
    # which would double-count since prefix already includes its own decay).
    if tail_link is not None:
        score *= hop_decay

    return score

"""Weight dynamics for the AgMem memory graph.

Implements Hebbian-style weight updates on traversal, time-based decay,
and pruning of links that fall below a threshold.
"""

from __future__ import annotations

import math
import time
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from agmem.core import AgMemDB, Link, Node


def sigmoid(x: float) -> float:
    """Squash any real value into the range ``(0, 1)``.

    The midpoint is at ``x=0`` where ``sigmoid(0) = 0.5``.
    """
    if x >= 0:
        # Numerically stable for large positive x.
        return 1.0 / (1.0 + math.exp(-x))
    else:
        # Avoid overflow for large negative x: exp(x) / (1 + exp(x))
        return math.exp(x) / (1.0 + math.exp(x))


_LT_THRESHOLD: float = 0.2
"""Links with weight below this threshold are candidates for pruning."""


def on_traverse(
    db: AgMemDB,
    link_id: int,
    outcome_signal: float = 0.0,
) -> None:
    """Hebbian-style weight update fired when a path crosses *link*.

    Implements the spec's simple update::

        weight += delta
        weight = sigmoid(weight)

    The outcome signal encodes the valence of the access:

    +--------------+---------+--------------------------------------+
    | Signal       | Delta   | Use case                             |
    +--------------+---------+--------------------------------------+
    | > 0 (pos)    | +0.10   | User confirmed the memory is useful   |
    | < 0 (neg)    | -0.05   | User indicated irrelevance            |
    | = 0 (neutral)| +0.02   | Passive retrieval (no feedback)       |
    +--------------+---------+--------------------------------------+

    The raw update is squashed through ``sigmoid``, so weight always stays
    in ``(0, 1)``.
    """
    link = db.get_link(link_id)
    if link is None:
        return

    if outcome_signal > 0:
        delta = 0.10
    elif outcome_signal < 0:
        delta = -0.05
    else:
        delta = 0.02

    raw = link.weight + delta
    new_weight = sigmoid(raw)

    now = int(time.time())
    db.update_link(
        link_id,
        weight=round(new_weight, 6),
        traverse_count=link.traverse_count + 1,
        last_traversed=now,
    )


def daily_decay(db: AgMemDB, node_id: Optional[str] = None) -> int:
    """Apply time-based decay to all links, pruning those below threshold.

    The decay formula:

        weight *= (1 - node.decay_rate) ** days_since_last_traversal

    Links whose weight falls below ``_LT_THRESHOLD (0.2)`` are deleted.

    Parameters
    ----------
    db : AgMemDB
    node_id : str, optional
        If provided, only decay links incident to this node.

    Returns
    -------
    int
        Number of links pruned.
    """
    now = int(time.time())
    pruned = 0

    links = db.get_links(node_id=node_id, min_weight=0.0) if node_id else db.get_all_links()

    for link in links:
        source_node = db.get_node(link.source_id)
        if source_node is None:
            continue

        last_ts = link.last_traversed if link.last_traversed else link.created_at
        days_since = max(0.0, (now - last_ts) / 86400.0)
        decayed = link.weight * ((1.0 - source_node.decay_rate) ** days_since)

        if decayed < _LT_THRESHOLD:
            db.delete_link(link.id)
            pruned += 1
        else:
            db.update_link(link.id, weight=round(decayed, 6))

    return pruned

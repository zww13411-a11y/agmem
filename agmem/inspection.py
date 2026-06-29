"""金行 — Inspection pass for AgMem memory graph.

**涩 → 剥 → 断 → 割 → 决**

Each node is examined before pruning. The inspection pass assigns a verdict:

- **keep** — memory is healthy, no action needed
- **archive** — low value but worth keeping for history (strength frozen, no decay)
- **prune** — candidate for deletion (low access, negative outcome, decaying fast)
- **review** — borderline case, flag for human review

Usage::

    from agmem.inspection import run_inspection
    report = run_inspection(db)
    # report.summary == {"keep": 5, "archive": 2, "prune": 3, "review": 1}
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Optional

if TYPE_CHECKING:
    from agmem.core import AgMemDB, Node

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants — the 涩 (resistance) thresholds
# ---------------------------------------------------------------------------

# Nodes below this strength are candidates for pruning.
STRENGTH_FLOOR: float = 0.25

# Nodes with access_count below this after > 30 days are low-value.
MIN_ACCESS_LIFETIME: int = 5

# If a node's most recent outcome is FAILURE, boost prune score.
OUTCOME_PENALTY: float = 0.3

# Days since creation before a node is eligible for inspection.
MIN_AGE_DAYS: int = 1

# Days since last access before a node is considered "cold".
COLD_THRESHOLD_DAYS: int = 14

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


@dataclass
class InspectionReport:
    """Result of running the inspection pass."""

    inspected_count: int = 0
    keep: List[str] = field(default_factory=list)
    archive: List[str] = field(default_factory=list)
    prune: List[str] = field(default_factory=list)
    review: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    @property
    def summary(self) -> Dict[str, int]:
        return {
            "keep": len(self.keep),
            "archive": len(self.archive),
            "prune": len(self.prune),
            "review": len(self.review),
        }

    def print(self) -> None:
        """Pretty-print the report."""
        total = self.inspected_count
        print(f"\n{'=' * 60}")
        print(f"  金行 Inspection Report — {total} nodes examined")
        print(f"{'=' * 60}")
        print(f"  ✅ keep:    {len(self.keep):3d}  — healthy, no action")
        print(f"  📦 archive: {len(self.archive):3d}  — low value, freeze")
        print(f"  ✂️  prune:   {len(self.prune):3d}  — candidate for deletion")
        print(f"  🔍 review:  {len(self.review):3d}  — needs human judgment")
        if self.errors:
            print(f"  ❌ errors:  {len(self.errors):d}")
            for e in self.errors:
                print(f"       {e}")
        print(f"{'=' * 60}\n")


# ---------------------------------------------------------------------------
# Core inspection logic
# ---------------------------------------------------------------------------


def _age_days(node: Node, now: int) -> float:
    return max(0.0, (now - node.created_at) / 86400.0)


def _days_since_last_access(node: Node, now: int) -> float:
    last = node.last_accessed if node.last_accessed is not None else node.created_at
    return max(0.0, (now - last) / 86400.0)


def _compute_prune_score(node: Node, now: int) -> float:
    """Compute a prune-likelihood score in [0, 1].

    Factors:
    - Low strength → higher prune score
    - Low access count → higher prune score
    - Outcome == FAILURE → higher prune score
    - Cold (not accessed recently) → higher prune score
    """
    score = 0.0

    # Strength factor (inverted: low strength = high prune score)
    if node.strength < STRENGTH_FLOOR:
        score += 0.4 * (1.0 - node.strength / STRENGTH_FLOOR)

    # Access factor
    age = _age_days(node, now)
    if age > 7 and node.access_count < MIN_ACCESS_LIFETIME:
        score += 0.3

    # Outcome factor
    if node.outcome == "FAILURE":
        score += OUTCOME_PENALTY

    # Cold factor
    cold_days = _days_since_last_access(node, now)
    if cold_days > COLD_THRESHOLD_DAYS:
        score += 0.2 * min(1.0, cold_days / 90.0)

    return min(1.0, score)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def run_inspection(
    db: AgMemDB,
    dry_run: bool = True,
    min_strength: float = 0.0,
    limit: int = 1000,
) -> InspectionReport:
    """Run the 金行 inspection pass on all eligible nodes.

    Parameters
    ----------
    db : AgMemDB
    dry_run : bool
        When True (default), verdicts are computed but NOT written to the DB.
        Set to False to commit verdicts.
    min_strength : float
        Only inspect nodes with strength >= this value. 0 = all.
    limit : int
        Max nodes to inspect.

    Returns
    -------
    InspectionReport
    """
    now = int(time.time())
    report = InspectionReport()

    nodes = db.get_all_nodes(limit=limit)

    for node in nodes:
        if node.strength < min_strength:
            continue

        age = _age_days(node, now)
        if age < MIN_AGE_DAYS:
            # Too young to judge — keep
            report.keep.append(node.id)
            continue

        report.inspected_count += 1
        prune_score = _compute_prune_score(node, now)

        # --- Verdict assignment (涩 → 剥 → 断) ---

        if prune_score >= 0.7:
            # 断 — strong candidate for deletion
            verdict = "prune"
            report.prune.append(node.id)
        elif prune_score >= 0.5:
            # 剥 — on the edge, flag for review
            verdict = "review"
            report.review.append(node.id)
        elif prune_score >= 0.3:
            # 涩 — low value, archive but don't delete
            verdict = "archive"
            report.archive.append(node.id)
        else:
            verdict = "keep"
            report.keep.append(node.id)

        # Commit verdict to DB (only if not dry_run)
        if not dry_run and verdict != node.inspection_verdict:
            try:
                db.update_node(
                    node.id,
                    inspection_verdict=verdict,
                    last_inspected=now,
                )
            except Exception as exc:
                report.errors.append(f"Failed to update {node.id[:12]}: {exc}")

    return report


def apply_prune(
    db: AgMemDB,
    report: InspectionReport,
    confirm: bool = True,
) -> int:
    """Delete nodes with 'prune' verdict.

    Parameters
    ----------
    db : AgMemDB
    report : InspectionReport
        An inspection report with prune candidates.
    confirm : bool
        When True (default), prints the list and returns without deleting.
        Set to False to actually delete.

    Returns
    -------
    int
        Number of nodes that would be / were deleted.
    """
    candidates = report.prune
    if not candidates:
        return 0

    print(f"\n  ✂️  Prune candidates ({len(candidates)}):")
    for i, nid in enumerate(candidates):
        node = db.get_node(nid)
        if node:
            preview = node.content[:60].replace("\n", " ")
            print(f"    {i+1:3d}. [{node.node_type}] {preview}…")

    if confirm:
        print(f"\n  ⚠️  Dry-run: {len(candidates)} would be pruned.")
        print(f"     Call apply_prune(db, report, confirm=False) to execute.\n")
        return 0

    deleted = 0
    for nid in candidates:
        if db.delete_node(nid):
            deleted += 1
            logger.info("Pruned node %s", nid)
    return deleted

"""水行 — User profile generator for system prompt injection.

**凝 → 润 → 流 → 渊**

Pipeline:
1. **凝** — Gather high-confidence memories (strength > 0.6, keep verdict)
2. **润** — Group by node_type, extract preference patterns
3. **流** — Generate a concise, readable user profile summary
4. Format as a system prompt block for session-start injection

Usage::

    from agmem.profile import generate_profile

    profile_text = generate_profile(db)
    # → "## 👤 User Profile\\n\\n**Preferences**:..." 

    # Inject this into system prompt at session start.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Dict, List, Tuple

if TYPE_CHECKING:
    from agmem.core import AgMemDB, Node

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------

# Only include nodes with strength >= this value.
MIN_PROFILE_STRENGTH: float = 0.6

# Max total characters for the profile block (fits in system prompt).
MAX_PROFILE_CHARS: int = 2000

# Max items per section.
MAX_ITEMS_PER_SECTION: int = 8

# ---------------------------------------------------------------------------
# Profile generation
# ---------------------------------------------------------------------------


def _collect_high_confidence_nodes(
    db: AgMemDB,
    min_strength: float = MIN_PROFILE_STRENGTH,
) -> List[Node]:
    """Gather high-confidence, non-pruned memories."""
    all_nodes = db.get_all_nodes(limit=500)
    profile_nodes: List[Node] = []

    for node in all_nodes:
        # Skip pruned/review/archive nodes.
        if node.inspection_verdict in ("prune", "review"):
            continue
        if node.strength < min_strength:
            continue
        # Skip temporary/low-value types.
        if node.node_type == "temp":
            continue
        profile_nodes.append(node)

    # Sort by strength descending, then access count.
    profile_nodes.sort(
        key=lambda n: (n.strength, n.access_count or 0),
        reverse=True,
    )

    return profile_nodes


def _group_by_type(
    nodes: List[Node],
) -> Dict[str, List[Node]]:
    """Group nodes by their node_type."""
    groups: Dict[str, List[Node]] = {}
    for node in nodes:
        t = node.node_type
        if t not in groups:
            groups[t] = []
        groups[t].append(node)
    return groups


def _format_preferences(nodes: List[Node]) -> str:
    """Format preference-type memories."""
    if not nodes:
        return ""

    lines: List[str] = []
    for node in nodes[:MAX_ITEMS_PER_SECTION]:
        content = _truncate(node.content.strip(), 150)
        if content:
            lines.append(f"- {content}")

    if lines:
        return "**Preferences**:\n" + "\n".join(lines) + "\n"
    return ""


def _format_facts(nodes: List[Node]) -> str:
    """Format factual memories."""
    if not nodes:
        return ""

    lines: List[str] = []
    for node in nodes[:MAX_ITEMS_PER_SECTION]:
        content = _truncate(node.content.strip(), 150)
        if content:
            lines.append(f"- {content}")

    if lines:
        return "**Known facts**:\n" + "\n".join(lines) + "\n"
    return ""


def _format_decisions(nodes: List[Node]) -> str:
    """Format decision memories."""
    if not nodes:
        return ""

    lines: List[str] = []
    for node in nodes[:MAX_ITEMS_PER_SECTION]:
        content = _truncate(node.content.strip(), 120)
        lines.append(f"- {content}")

    if lines:
        return "**Past decisions**:\n" + "\n".join(lines) + "\n"
    return ""


def _truncate(text: str, max_len: int) -> str:
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def generate_profile(
    db: AgMemDB,
    min_strength: float = MIN_PROFILE_STRENGTH,
    max_chars: int = MAX_PROFILE_CHARS,
) -> str:
    """Generate a user profile summary from high-confidence memories.

    The output is formatted as a system-prompt block, ready for injection
    at session start.

    Parameters
    ----------
    db : AgMemDB
    min_strength : float
        Minimum node strength to include (default 0.6).
    max_chars : int
        Maximum total characters (default 2000).

    Returns
    -------
    str
        Profile text, or empty string if no qualifying memories.
    """
    nodes = _collect_high_confidence_nodes(db, min_strength)

    if not nodes:
        return ""

    groups = _group_by_type(nodes)

    sections: List[str] = []
    current_len = 0
    header = "## 👤 User Profile\n\n"

    # Preferences first (highest value).
    pref_text = _format_preferences(groups.get("preference", []))
    if pref_text and current_len + len(pref_text) <= max_chars:
        sections.append(pref_text)
        current_len += len(pref_text)

    # Facts next.
    fact_text = _format_facts(groups.get("fact", []))
    if fact_text and current_len + len(fact_text) <= max_chars:
        sections.append(fact_text)
        current_len += len(fact_text)

    # Decisions last.
    dec_text = _format_decisions(groups.get("decision", []))
    if dec_text and current_len + len(dec_text) <= max_chars:
        sections.append(dec_text)
        current_len += len(dec_text)

    # Insights.
    insight_nodes = groups.get("insight", [])
    if insight_nodes and current_len < max_chars:
        insight_lines: List[str] = []
        for node in insight_nodes[:4]:
            content = _truncate(node.content.strip(), 100)
            insight_lines.append(f"- {content}")
        if insight_lines:
            insight_block = "**Insights**:\n" + "\n".join(insight_lines) + "\n"
            if current_len + len(insight_block) <= max_chars:
                sections.append(insight_block)
                current_len += len(insight_block)

    if not sections:
        return ""

    return header + "\n".join(sections)


def generate_summary_line(db: AgMemDB) -> str:
    """Generate a one-line summary of the user.

    For situations where a full profile is too long (e.g. tool description).
    """
    nodes = _collect_high_confidence_nodes(db, min_strength=0.7)

    if not nodes:
        return ""

    # Count by type.
    type_count: Dict[str, int] = {}
    for n in nodes:
        type_count[n.node_type] = type_count.get(n.node_type, 0) + 1

    pref_count = type_count.get("preference", 0)
    fact_count = type_count.get("fact", 0)

    parts = []
    if pref_count:
        parts.append(f"{pref_count} preferences")
    if fact_count:
        parts.append(f"{fact_count} facts")

    top = nodes[0].content[:60].replace("\n", " ")

    stats = ", ".join(parts) if parts else f"{len(nodes)} memories"
    return f"[Profile: {stats} | Top: {top}]"

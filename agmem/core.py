"""SQLite-backed persistent storage for the AgMem memory graph.

Schema
------
- nodes:     each memory is a typed node with strength, decay, and entity tags
- links:     weighted directed edges between nodes (extends, contradicts, …)
- sessions:  conversation-episode tracking for future consolidation

All timestamps are Unix epoch seconds (int).
"""

from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class Node:
    """A single memory node in the graph."""

    id: str
    content: str
    node_type: str  # preference | fact | decision | insight | process
    strength: float = 0.5
    created_at: int = 0
    last_accessed: Optional[int] = None
    access_count: int = 0
    decay_rate: float = 0.05
    entities: List[str] = field(default_factory=list)
    outcome: Optional[str] = None    # SUCCESS | FAILURE | pending | null
    last_inspected: Optional[int] = None  # unix epoch of last inspection pass
    inspection_verdict: Optional[str] = None  # keep | archive | prune | review

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "content": self.content,
            "node_type": self.node_type,
            "strength": self.strength,
            "created_at": self.created_at,
            "last_accessed": self.last_accessed,
            "access_count": self.access_count,
            "decay_rate": self.decay_rate,
            "entities": self.entities,
            "outcome": self.outcome,
            "last_inspected": self.last_inspected,
            "inspection_verdict": self.inspection_verdict,
        }


@dataclass
class Link:
    """A weighted directed edge between two memory nodes."""

    id: int = 0
    source_id: str = ""
    target_id: str = ""
    relation: str = ""  # extends | contradicts | example_of | inspired_by | related_to
    weight: float = 0.3
    created_at: int = 0
    last_traversed: Optional[int] = None
    traverse_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "source_id": self.source_id,
            "target_id": self.target_id,
            "relation": self.relation,
            "weight": self.weight,
            "created_at": self.created_at,
            "last_traversed": self.last_traversed,
            "traverse_count": self.traverse_count,
        }


@dataclass
class ScoredPath:
    """Result of a graph walk — a path of linked nodes with a composite score."""

    nodes: List[Node]
    links: List[Link]
    score: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "content": " → ".join(n.content for n in self.nodes),
            "path": " → ".join(n.id for n in self.nodes),
            "score": round(self.score, 4),
            "node_ids": [n.id for n in self.nodes],
            "nodes": [n.to_dict() for n in self.nodes],
        }


# ---------------------------------------------------------------------------
# Default decay rates per node type
# ---------------------------------------------------------------------------

DECAY_RATES: Dict[str, float] = {
    "preference": 0.01,
    "fact": 0.05,
    "decision": 0.03,
    "insight": 0.04,
    "process": 0.07,
    "temp": 0.10,
}

VALID_NODE_TYPES: set = set(DECAY_RATES.keys())
VALID_RELATIONS: set = {
    "extends",
    "contradicts",
    "example_of",
    "inspired_by",
    "related_to",
}


# ---------------------------------------------------------------------------
# Database manager
# ---------------------------------------------------------------------------


class AgMemDB:
    """Manages the SQLite database for the AgMem memory graph.

    Usage
    -----
    db = AgMemDB(":memory:")          # in-memory (testing)
    db = AgMemDB("agmem.db")          # on-disk persistence
    """

    def __init__(self, db_path: str = "agmem.db") -> None:
        self._db_path = db_path
        self._conn: sqlite3.Connection = sqlite3.connect(db_path)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._init_schema()

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    def _init_schema(self) -> None:
        """Create tables if they do not exist."""
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS nodes (
                id              TEXT PRIMARY KEY,
                content         TEXT NOT NULL,
                node_type       TEXT NOT NULL,
                strength        REAL DEFAULT 0.5,
                created_at      INTEGER NOT NULL,
                last_accessed   INTEGER,
                access_count    INTEGER DEFAULT 0,
                decay_rate      REAL DEFAULT 0.05,
                entities        TEXT,
                outcome         TEXT,
                last_inspected  INTEGER,
                inspection_verdict TEXT
            );

            CREATE TABLE IF NOT EXISTS links (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                source_id       TEXT NOT NULL REFERENCES nodes(id),
                target_id       TEXT NOT NULL REFERENCES nodes(id),
                relation        TEXT NOT NULL,
                weight          REAL DEFAULT 0.3,
                created_at      INTEGER NOT NULL,
                last_traversed  INTEGER,
                traverse_count  INTEGER DEFAULT 0,
                UNIQUE(source_id, target_id, relation)
            );

            CREATE TABLE IF NOT EXISTS sessions (
                id              TEXT PRIMARY KEY,
                started_at      INTEGER NOT NULL,
                ended_at        INTEGER,
                turn_count      INTEGER DEFAULT 0
            );

            CREATE INDEX IF NOT EXISTS idx_nodes_entities ON nodes(entities);
            CREATE INDEX IF NOT EXISTS idx_nodes_type     ON nodes(node_type);
            CREATE INDEX IF NOT EXISTS idx_links_source   ON links(source_id);
            CREATE INDEX IF NOT EXISTS idx_links_target   ON links(target_id);

            -- FTS5 full-text search index
            CREATE VIRTUAL TABLE IF NOT EXISTS nodes_fts USING fts5(
                content,
                node_type UNINDEXED,
                entities UNINDEXED,
                id UNINDEXED,
                tokenize='unicode61'
            );
        """)
        self._conn.commit()

    # ------------------------------------------------------------------
    # Node CRUD
    # ------------------------------------------------------------------

    def create_node(
        self,
        node_id: str,
        content: str,
        node_type: str = "fact",
        strength: float = 0.5,
        entities: Optional[List[str]] = None,
        decay_rate: Optional[float] = None,
    ) -> Node:
        """Insert a new node. Raises ValueError on invalid type or duplicate id."""
        if node_type not in VALID_NODE_TYPES:
            raise ValueError(
                f"Invalid node_type {node_type!r}. "
                f"Must be one of {sorted(VALID_NODE_TYPES)}"
            )
        now = int(time.time())
        dr = decay_rate if decay_rate is not None else DECAY_RATES[node_type]
        entities_json = json.dumps(entities or [])

        try:
            self._conn.execute(
                """INSERT INTO nodes
                   (id, content, node_type, strength, created_at, decay_rate, entities)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (node_id, content, node_type, strength, now, dr, entities_json),
            )
            self._conn.commit()
        except sqlite3.IntegrityError:
            raise ValueError(f"Node {node_id!r} already exists") from None

        # Direct FTS5 insert (no delete needed — fresh node)
        entities_text = " ".join(entities or [])
        self._conn.execute(
            "INSERT OR IGNORE INTO nodes_fts (id, content, node_type, entities) VALUES (?, ?, ?, ?)",
            (node_id, content, node_type, entities_text),
        )
        self._conn.commit()
        return Node(
            id=node_id,
            content=content,
            node_type=node_type,
            strength=strength,
            created_at=now,
            decay_rate=dr,
            entities=entities or [],
            outcome=None,
        )

    def get_node(self, node_id: str) -> Optional[Node]:
        """Fetch a node by id, or None."""
        row = self._conn.execute(
            "SELECT * FROM nodes WHERE id = ?", (node_id,)
        ).fetchone()
        if row is None:
            return None
        return self._row_to_node(row)

    def update_node(self, node_id: str, **kwargs: Any) -> Optional[Node]:
        """Update scalar fields on a node in-place.

        Accepted keys: strength, decay_rate, last_accessed, access_count, content.
        Returns the updated node or None if not found.
        """
        allowed = {"strength", "decay_rate", "last_accessed", "access_count", "content", "outcome", "last_inspected", "inspection_verdict"}
        updates = {k: v for k, v in kwargs.items() if k in allowed}
        if not updates:
            return self.get_node(node_id)

        sets = ", ".join(f"{k} = ?" for k in updates)
        values = list(updates.values()) + [node_id]
        self._conn.execute(f"UPDATE nodes SET {sets} WHERE id = ?", values)
        self._conn.commit()
        self._sync_fts5(node_id)
        return self.get_node(node_id)

    def delete_node(self, node_id: str) -> bool:
        """Delete a node and all incident links. Returns True if deleted."""
        self._conn.execute(
            "DELETE FROM links WHERE source_id = ? OR target_id = ?",
            (node_id, node_id),
        )
        cur = self._conn.execute("DELETE FROM nodes WHERE id = ?", (node_id,))
        self._conn.commit()
        self._sync_fts5(node_id)
        return cur.rowcount > 0

    def search_nodes(
        self,
        *,
        node_type: Optional[str] = None,
        min_strength: float = 0.0,
        limit: int = 50,
    ) -> List[Node]:
        """Search nodes by optional type and minimum strength."""
        clauses: List[str] = []
        params: List[Any] = []

        if node_type is not None:
            clauses.append("node_type = ?")
            params.append(node_type)
        if min_strength > 0.0:
            clauses.append("strength >= ?")
            params.append(min_strength)

        where = " AND ".join(clauses) if clauses else "1"
        rows = self._conn.execute(
            f"SELECT * FROM nodes WHERE {where} ORDER BY strength DESC LIMIT ?",
            params + [limit],
        ).fetchall()
        return [self._row_to_node(r) for r in rows]

    def find_nodes_by_entities(
        self,
        entities: List[str],
        *,
        min_strength: float = 0.0,
        limit: int = 50,
    ) -> List[Tuple[Node, List[str]]]:
        """Find nodes whose stored entity list overlaps with *entities*.

        Returns ``(node, matched_entities)`` pairs sorted by number of matches.
        """
        if not entities:
            return []

        rows = self._conn.execute(
            "SELECT * FROM nodes WHERE entities IS NOT NULL "
            "AND strength >= ? ORDER BY strength DESC LIMIT ?",
            (min_strength, limit),
        ).fetchall()

        entity_set = set(e.lower() for e in entities)
        results: List[Tuple[Node, List[str]]] = []
        for row in rows:
            stored: List[str] = json.loads(row["entities"]) if row["entities"] else []
            matched = [e for e in stored if e.lower() in entity_set]
            if matched:
                results.append((self._row_to_node(row), matched))

        # Sort by match count descending, then strength descending.
        results.sort(key=lambda t: (len(t[1]), t[0].strength), reverse=True)
        return results[:limit]

    # ------------------------------------------------------------------
    # Link CRUD
    # ------------------------------------------------------------------

    def create_link(
        self,
        source_id: str,
        target_id: str,
        relation: str = "related_to",
        weight: float = 0.3,
    ) -> Link:
        """Create a directed link between two existing nodes.

        Raises ValueError if either node is missing or relation is invalid.
        """
        if relation not in VALID_RELATIONS:
            raise ValueError(
                f"Invalid relation {relation!r}. "
                f"Must be one of {sorted(VALID_RELATIONS)}"
            )
        if self.get_node(source_id) is None:
            raise ValueError(f"Source node {source_id!r} does not exist")
        if self.get_node(target_id) is None:
            raise ValueError(f"Target node {target_id!r} does not exist")

        now = int(time.time())
        try:
            self._conn.execute(
                """INSERT INTO links
                   (source_id, target_id, relation, weight, created_at)
                   VALUES (?, ?, ?, ?, ?)""",
                (source_id, target_id, relation, weight, now),
            )
            self._conn.commit()
        except sqlite3.IntegrityError:
            raise ValueError(
                f"Link {source_id}→{target_id} ({relation}) already exists"
            ) from None

        return Link(
            id=self._conn.execute("SELECT last_insert_rowid()").fetchone()[0],
            source_id=source_id,
            target_id=target_id,
            relation=relation,
            weight=weight,
            created_at=now,
            traverse_count=0,
        )

    def get_link(self, link_id: int) -> Optional[Link]:
        """Fetch a link by id."""
        row = self._conn.execute(
            "SELECT * FROM links WHERE id = ?", (link_id,)
        ).fetchone()
        return self._row_to_link(row) if row is not None else None

    def get_links(
        self,
        *,
        node_id: Optional[str] = None,
        relation: Optional[str] = None,
        min_weight: float = 0.0,
        direction: str = "both",
        limit: int = 100,
    ) -> List[Link]:
        """Query links by node, relation, or minimum weight.

        Parameters
        ----------
        node_id : str, optional
            Filter links involving this node.
        relation : str, optional
            Filter by relation type.
        min_weight : float
            Minimum link weight (default 0.0).
        direction : "outgoing" | "incoming" | "both"
            Link direction relative to *node_id*. Ignored when *node_id* is None.
        """
        clauses: List[str] = ["weight >= ?"]
        params: List[Any] = [min_weight]

        if relation is not None:
            clauses.append("relation = ?")
            params.append(relation)

        if node_id is not None:
            if direction == "outgoing":
                clauses.append("source_id = ?")
                params.append(node_id)
            elif direction == "incoming":
                clauses.append("target_id = ?")
                params.append(node_id)
            else:
                clauses.append("(source_id = ? OR target_id = ?)")
                params.extend([node_id, node_id])

        where = " AND ".join(clauses)
        rows = self._conn.execute(
            f"SELECT * FROM links WHERE {where} ORDER BY weight DESC LIMIT ?",
            params + [limit],
        ).fetchall()
        return [self._row_to_link(r) for r in rows]

    def update_link(self, link_id: int, **kwargs: Any) -> Optional[Link]:
        """Update link fields in-place.

        Accepted keys: weight, last_traversed, traverse_count.
        """
        allowed = {"weight", "last_traversed", "traverse_count"}
        updates = {k: v for k, v in kwargs.items() if k in allowed}
        if not updates:
            return self.get_link(link_id)

        sets = ", ".join(f"{k} = ?" for k in updates)
        values = list(updates.values()) + [link_id]
        self._conn.execute(f"UPDATE links SET {sets} WHERE id = ?", values)
        self._conn.commit()
        return self.get_link(link_id)

    def delete_link(self, link_id: int) -> bool:
        """Delete a link by id. Returns True if deleted."""
        cur = self._conn.execute("DELETE FROM links WHERE id = ?", (link_id,))
        self._conn.commit()
        return cur.rowcount > 0

    # ------------------------------------------------------------------
    # Session tracking
    # ------------------------------------------------------------------

    def create_session(self, session_id: str) -> None:
        now = int(time.time())
        self._conn.execute(
            "INSERT OR IGNORE INTO sessions (id, started_at) VALUES (?, ?)",
            (session_id, now),
        )
        self._conn.commit()

    def end_session(self, session_id: str) -> None:
        now = int(time.time())
        self._conn.execute(
            "UPDATE sessions SET ended_at = ? WHERE id = ?", (now, session_id)
        )
        self._conn.commit()

    def increment_turns(self, session_id: str) -> None:
        self._conn.execute(
            "UPDATE sessions SET turn_count = turn_count + 1 WHERE id = ?",
            (session_id,),
        )
        self._conn.commit()

    # ------------------------------------------------------------------
    # FTS5 Full-Text Search
    # ------------------------------------------------------------------

    def _sync_fts5(self, node_id: str) -> None:
        """Insert or update the FTS5 index for a node."""
        node = self.get_node(node_id)
        if node is None:
            self._conn.execute(
                "DELETE FROM nodes_fts WHERE id = ?", (node_id,)
            )
            self._conn.commit()
            return
        # Upsert: delete then insert
        self._conn.execute("DELETE FROM nodes_fts WHERE id = ?", (node_id,))
        # Store entities as space-separated plain text for clean FTS5 search
        entities_text = " ".join(node.entities) if node.entities else ""
        self._conn.execute(
            "INSERT INTO nodes_fts (id, content, node_type, entities) VALUES (?, ?, ?, ?)",
            (node.id, node.content, node.node_type, entities_text),
        )
        self._conn.commit()

    def fts5_search(
        self,
        query: str,
        limit: int = 20,
    ) -> List[Node]:
        """Full-text search across all nodes using FTS5.

        Supports FTS5 query syntax:
        - Simple terms: ``python`` → matches nodes with "python"
        - Prefix: ``pyt*`` → matches "python", "pytorch", etc.
        - Phrase: ``"data science"`` → exact phrase match
        - Column: ``content: redis`` → search in content only

        Returns nodes sorted by FTS5 rank (best match first).
        """
        if not query or not query.strip():
            return []

        try:
            rows = self._conn.execute(
                """SELECT n.* FROM nodes n
                   JOIN nodes_fts fts ON n.id = fts.id
                   WHERE nodes_fts MATCH ?
                   ORDER BY rank
                   LIMIT ?""",
                (query.strip(), limit),
            ).fetchall()
        except sqlite3.OperationalError:
            # Invalid FTS5 query syntax — fall back to LIKE
            like = f"%{query.strip()}%"
            rows = self._conn.execute(
                "SELECT * FROM nodes WHERE content LIKE ? OR entities LIKE ? LIMIT ?",
                (like, like, limit),
            ).fetchall()

        return [self._row_to_node(r) for r in rows]
    # ------------------------------------------------------------------

    def get_all_nodes(self, limit: int = 1000) -> List[Node]:
        rows = self._conn.execute(
            "SELECT * FROM nodes ORDER BY strength DESC LIMIT ?", (limit,)
        ).fetchall()
        return [self._row_to_node(r) for r in rows]

    def get_all_links(self, limit: int = 5000) -> List[Link]:
        rows = self._conn.execute(
            "SELECT * FROM links ORDER BY weight DESC LIMIT ?", (limit,)
        ).fetchall()
        return [self._row_to_link(r) for r in rows]

    def node_count(self) -> int:
        return self._conn.execute("SELECT COUNT(*) FROM nodes").fetchone()[0]

    def link_count(self) -> int:
        return self._conn.execute("SELECT COUNT(*) FROM links").fetchone()[0]

    # ------------------------------------------------------------------
    # Housekeeping
    # ------------------------------------------------------------------

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> AgMemDB:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _row_to_node(row: sqlite3.Row) -> Node:
        entities: List[str] = json.loads(row["entities"]) if row["entities"] else []
        return Node(
            id=row["id"],
            content=row["content"],
            node_type=row["node_type"],
            strength=row["strength"],
            created_at=row["created_at"],
            last_accessed=row["last_accessed"],
            access_count=row["access_count"],
            decay_rate=row["decay_rate"],
            entities=entities,
            outcome=row["outcome"],
            last_inspected=row["last_inspected"],
            inspection_verdict=row["inspection_verdict"],
        )

    @staticmethod
    def _row_to_link(row: sqlite3.Row) -> Link:
        return Link(
            id=row["id"],
            source_id=row["source_id"],
            target_id=row["target_id"],
            relation=row["relation"],
            weight=row["weight"],
            created_at=row["created_at"],
            last_traversed=row["last_traversed"],
            traverse_count=row["traverse_count"],
        )

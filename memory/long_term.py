"""
Long-term memory — SQLite-backed persistent storage of conversations and facts.

Schema
------
conversations  — full turn-by-turn history with session grouping
facts          — key/value pairs extracted or manually stored

Retrieval uses keyword LIKE search (no vector embeddings required).
Embeddings can be layered on top later by switching the `search` method.
"""

from __future__ import annotations

import sqlite3
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, Generator, List, Optional

from config import settings
from utils.logger import get_logger

log = get_logger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS conversations (
    id          TEXT PRIMARY KEY,
    session_id  TEXT NOT NULL,
    timestamp   REAL NOT NULL,
    role        TEXT NOT NULL,
    content     TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_conv_session   ON conversations (session_id);
CREATE INDEX IF NOT EXISTS idx_conv_timestamp ON conversations (timestamp DESC);

CREATE TABLE IF NOT EXISTS facts (
    id        TEXT PRIMARY KEY,
    timestamp REAL NOT NULL,
    key       TEXT NOT NULL UNIQUE,
    value     TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_facts_key ON facts (key);
"""


class LongTermMemory:
    """
    Persistent conversation and fact store backed by SQLite.

    Thread-safe: uses a per-call connection from the pool (check_same_thread=False).
    """

    def __init__(self, db_path: Optional[Path] = None) -> None:
        self._db_path = db_path or settings.long_term_db_path
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()
        log.info("LongTermMemory initialised", extra={"db": str(self._db_path)})

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @contextmanager
    def _conn(self) -> Generator[sqlite3.Connection, None, None]:
        conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_db(self) -> None:
        with self._conn() as conn:
            conn.executescript(_SCHEMA)

    # ------------------------------------------------------------------
    # Conversations
    # ------------------------------------------------------------------

    def save(self, role: str, content: str, session_id: Optional[str] = None) -> str:
        """Persist a single message. Returns the generated row id."""
        row_id = str(uuid.uuid4())
        session_id = session_id or "default"
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO conversations (id, session_id, timestamp, role, content) "
                "VALUES (?, ?, ?, ?, ?)",
                (row_id, session_id, time.time(), role, content),
            )
        return row_id

    def get_recent(self, limit: int = 20, session_id: Optional[str] = None) -> List[Dict]:
        """Return the most recent messages, newest last."""
        with self._conn() as conn:
            if session_id:
                rows = conn.execute(
                    "SELECT role, content, timestamp FROM conversations "
                    "WHERE session_id = ? ORDER BY timestamp DESC LIMIT ?",
                    (session_id, limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT role, content, timestamp FROM conversations "
                    "ORDER BY timestamp DESC LIMIT ?",
                    (limit,),
                ).fetchall()
        return [dict(r) for r in reversed(rows)]

    def search(self, query: str, limit: int = 5) -> List[Dict]:
        """
        Keyword search across conversation content.
        Each word in `query` must appear somewhere (AND logic across words,
        OR across rows via LIKE matching).
        """
        if not query.strip():
            return []

        words = query.strip().split()
        # Build: content LIKE '%word1%' AND content LIKE '%word2%' ...
        clauses = " AND ".join(["content LIKE ?"] * len(words))
        params = [f"%{w}%" for w in words] + [limit]

        with self._conn() as conn:
            rows = conn.execute(
                f"SELECT role, content, timestamp FROM conversations "
                f"WHERE {clauses} ORDER BY timestamp DESC LIMIT ?",
                params,
            ).fetchall()
        return [dict(r) for r in rows]

    def delete_session(self, session_id: str) -> int:
        """Delete all rows for a session. Returns rows deleted."""
        with self._conn() as conn:
            cur = conn.execute(
                "DELETE FROM conversations WHERE session_id = ?", (session_id,)
            )
            return cur.rowcount

    # ------------------------------------------------------------------
    # Facts (key/value store)
    # ------------------------------------------------------------------

    def set_fact(self, key: str, value: str) -> None:
        """Upsert a named fact."""
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO facts (id, timestamp, key, value) VALUES (?, ?, ?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value, timestamp=excluded.timestamp",
                (str(uuid.uuid4()), time.time(), key, value),
            )

    def get_fact(self, key: str) -> Optional[str]:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT value FROM facts WHERE key = ?", (key,)
            ).fetchone()
        return row["value"] if row else None

    def list_facts(self) -> List[Dict]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT key, value, timestamp FROM facts ORDER BY key"
            ).fetchall()
        return [dict(r) for r in rows]

    def delete_fact(self, key: str) -> bool:
        with self._conn() as conn:
            cur = conn.execute("DELETE FROM facts WHERE key = ?", (key,))
            return cur.rowcount > 0

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def stats(self) -> Dict:
        with self._conn() as conn:
            conv_count = conn.execute("SELECT COUNT(*) FROM conversations").fetchone()[0]
            fact_count = conn.execute("SELECT COUNT(*) FROM facts").fetchone()[0]
        return {"conversations": conv_count, "facts": fact_count, "db": str(self._db_path)}

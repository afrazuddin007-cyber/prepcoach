"""SQLite-backed attempt tracking and interview sessions, scoped per user.

Every record carries a user_id so several people can share one server without sharing
progress. Callers that don't pass a user_id all share the "default" user.
"""
import json
import os
import sqlite3
import time

DB_PATH = os.environ.get("PREPCOACH_DB", os.path.join(os.path.dirname(__file__), "..", "prepcoach.db"))
DEFAULT_USER = "default"


def norm_user(user_id: str | None) -> str:
    uid = (user_id or "").strip()[:64]
    return uid or DEFAULT_USER


def _conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """CREATE TABLE IF NOT EXISTS attempts (
               id INTEGER PRIMARY KEY AUTOINCREMENT,
               question_id TEXT NOT NULL,
               topic TEXT NOT NULL,
               correct INTEGER NOT NULL,
               ts REAL NOT NULL,
               user_id TEXT NOT NULL DEFAULT 'default'
           )"""
    )
    conn.execute(
        """CREATE TABLE IF NOT EXISTS interviews (
               id TEXT PRIMARY KEY,
               state TEXT NOT NULL,
               updated REAL NOT NULL,
               user_id TEXT NOT NULL DEFAULT 'default'
           )"""
    )
    # Databases created before per-user support get the new column, existing rows become 'default'.
    for table in ("attempts", "interviews"):
        cols = {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}
        if "user_id" not in cols:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN user_id TEXT NOT NULL DEFAULT 'default'")
    return conn


def record_attempt(question_id: str, topic: str, correct: bool, user_id: str | None = None) -> None:
    with _conn() as conn:
        conn.execute(
            "INSERT INTO attempts (question_id, topic, correct, ts, user_id) VALUES (?, ?, ?, ?, ?)",
            (question_id, topic, int(correct), time.time(), norm_user(user_id)),
        )


def topic_stats(user_id: str | None = None) -> dict[str, dict]:
    """Return {topic: {attempts, correct, accuracy}} for this user's topics with attempts."""
    with _conn() as conn:
        rows = conn.execute(
            "SELECT topic, COUNT(*), SUM(correct) FROM attempts WHERE user_id = ? GROUP BY topic",
            (norm_user(user_id),),
        ).fetchall()
    return {
        topic: {"attempts": n, "correct": int(c or 0), "accuracy": round((c or 0) / n, 2)}
        for topic, n, c in rows
    }


def seen_question_ids(user_id: str | None = None) -> set[str]:
    with _conn() as conn:
        return {
            r[0]
            for r in conn.execute(
                "SELECT DISTINCT question_id FROM attempts WHERE user_id = ?", (norm_user(user_id),)
            )
        }


def save_interview(session_id: str, state: dict) -> None:
    with _conn() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO interviews (id, state, updated, user_id) VALUES (?, ?, ?, ?)",
            (session_id, json.dumps(state), time.time(), norm_user(state.get("user_id"))),
        )


def load_interview(session_id: str) -> dict | None:
    with _conn() as conn:
        row = conn.execute("SELECT state FROM interviews WHERE id = ?", (session_id,)).fetchone()
    return json.loads(row[0]) if row else None


def reset(user_id: str | None = None) -> None:
    """Clear one user's history. With no user_id, clear everyone's (used by tests)."""
    with _conn() as conn:
        if user_id is None:
            conn.execute("DELETE FROM attempts")
            conn.execute("DELETE FROM interviews")
        else:
            uid = norm_user(user_id)
            conn.execute("DELETE FROM attempts WHERE user_id = ?", (uid,))
            conn.execute("DELETE FROM interviews WHERE user_id = ?", (uid,))

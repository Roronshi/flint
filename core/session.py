# core/session.py — Session management and conversation log

from __future__ import annotations

import json
import logging
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import config

log = logging.getLogger(__name__)

SCHEMA_VERSION = 4


class ConversationDB:
    """SQLite store for conversation, reflection and background job data."""

    def __init__(self, db_path: str = None):
        self.db_path = db_path or config.CONVERSATIONS_DB
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_db()
        self.ensure_default_records()

    def _init_db(self):
        with self._conn() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS schema_version (
                    version INTEGER NOT NULL
                );

                CREATE TABLE IF NOT EXISTS companions (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    description TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    default_model_id TEXT,
                    active_initiative_profile_id TEXT,
                    status TEXT NOT NULL DEFAULT 'active',
                    last_session_id TEXT,
                    last_active_at TEXT
                );

                CREATE TABLE IF NOT EXISTS models (
                    id TEXT PRIMARY KEY,
                    engine_type TEXT NOT NULL,
                    family TEXT NOT NULL,
                    name TEXT NOT NULL,
                    version TEXT,
                    manifest_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS model_installations (
                    id TEXT PRIMARY KEY,
                    model_id TEXT NOT NULL,
                    install_path TEXT NOT NULL,
                    backend TEXT NOT NULL,
                    is_available INTEGER NOT NULL DEFAULT 1,
                    is_default INTEGER NOT NULL DEFAULT 0,
                    verification_status TEXT NOT NULL DEFAULT 'unknown',
                    last_verified_at TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY(model_id) REFERENCES models(id)
                );

                CREATE TABLE IF NOT EXISTS initiative_profiles (
                    id TEXT PRIMARY KEY,
                    companion_id TEXT NOT NULL,
                    profile_name TEXT NOT NULL,
                    reflection_frequency_minutes INTEGER NOT NULL,
                    outreach_max_per_day INTEGER NOT NULL,
                    minimum_priority_threshold REAL NOT NULL,
                    minimum_groundedness_threshold REAL NOT NULL,
                    minimum_novelty_threshold REAL NOT NULL,
                    allow_notifications INTEGER NOT NULL DEFAULT 0,
                    work_hours_silent INTEGER NOT NULL DEFAULT 0,
                    meeting_mode_silent INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    active INTEGER NOT NULL DEFAULT 0,
                    FOREIGN KEY(companion_id) REFERENCES companions(id)
                );

                CREATE TABLE IF NOT EXISTS sessions (
                    id TEXT PRIMARY KEY,
                    companion_id TEXT,
                    model_id TEXT,
                    started_at TEXT NOT NULL,
                    ended_at TEXT,
                    turn_count INTEGER DEFAULT 0,
                    message_count INTEGER DEFAULT 0,
                    lora_version TEXT,
                    source TEXT NOT NULL DEFAULT 'native',
                    title TEXT,
                    session_kind TEXT NOT NULL DEFAULT 'chat',
                    import_batch_id TEXT,
                    FOREIGN KEY(companion_id) REFERENCES companions(id),
                    FOREIGN KEY(model_id) REFERENCES models(id)
                );

                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    companion_id TEXT,
                    model_id TEXT,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    turn_index INTEGER,
                    token_count INTEGER,
                    imported INTEGER NOT NULL DEFAULT 0,
                    metadata_json TEXT,
                    FOREIGN KEY (session_id) REFERENCES sessions(id)
                );
                CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id);
                CREATE INDEX IF NOT EXISTS idx_messages_timestamp ON messages(timestamp DESC);
                CREATE INDEX IF NOT EXISTS idx_messages_companion ON messages(companion_id, timestamp DESC);

                CREATE TABLE IF NOT EXISTS conversation_blocks (
                    id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    companion_id TEXT NOT NULL,
                    model_id TEXT,
                    start_message_id INTEGER NOT NULL,
                    end_message_id INTEGER NOT NULL,
                    block_type TEXT NOT NULL DEFAULT 'recent',
                    block_text TEXT NOT NULL,
                    message_count INTEGER NOT NULL DEFAULT 0,
                    turn_count INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL,
                    summarized_at TEXT,
                    FOREIGN KEY(session_id) REFERENCES sessions(id)
                );
                CREATE INDEX IF NOT EXISTS idx_blocks_summarized ON conversation_blocks(companion_id, summarized_at);

                CREATE TABLE IF NOT EXISTS summaries (
                    id TEXT PRIMARY KEY,
                    companion_id TEXT NOT NULL,
                    model_id TEXT,
                    source_block_id TEXT,
                    source_summary_ids_json TEXT,
                    summary_level TEXT NOT NULL,
                    summary_type TEXT NOT NULL,
                    summary_text TEXT NOT NULL,
                    key_topics_json TEXT,
                    open_loops_json TEXT,
                    signals_json TEXT,
                    coverage_start_at TEXT,
                    coverage_end_at TEXT,
                    created_at TEXT NOT NULL,
                    quality_score REAL,
                    FOREIGN KEY(source_block_id) REFERENCES conversation_blocks(id)
                );
                CREATE INDEX IF NOT EXISTS idx_summaries_companion ON summaries(companion_id, summary_level, created_at DESC);

                CREATE TABLE IF NOT EXISTS semantic_memory (
                    id TEXT PRIMARY KEY,
                    companion_id TEXT NOT NULL,
                    model_id TEXT,
                    memory_type TEXT NOT NULL,
                    title TEXT NOT NULL,
                    content TEXT NOT NULL,
                    evidence_summary_ids_json TEXT,
                    importance_score REAL NOT NULL DEFAULT 0.5,
                    recency_score REAL NOT NULL DEFAULT 0.5,
                    stability_score REAL NOT NULL DEFAULT 0.5,
                    last_reinforced_at TEXT,
                    created_at TEXT NOT NULL,
                    archived_at TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_semantic_memory ON semantic_memory(companion_id, memory_type, archived_at);

                CREATE TABLE IF NOT EXISTS reflections (
                    id TEXT PRIMARY KEY,
                    companion_id TEXT NOT NULL,
                    model_id TEXT,
                    reflection_type TEXT NOT NULL,
                    input_bundle_json TEXT NOT NULL,
                    reflection_text TEXT NOT NULL,
                    question_text TEXT,
                    supporting_summary_ids_json TEXT,
                    supporting_memory_ids_json TEXT,
                    novelty_score REAL NOT NULL DEFAULT 0.0,
                    relevance_score REAL NOT NULL DEFAULT 0.0,
                    groundedness_score REAL NOT NULL DEFAULT 0.0,
                    sensitivity_score REAL NOT NULL DEFAULT 0.0,
                    priority_score REAL NOT NULL DEFAULT 0.0,
                    created_at TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'new',
                    gating_reason TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_reflections_companion ON reflections(companion_id, status, created_at DESC);

                CREATE TABLE IF NOT EXISTS outreach_candidates (
                    id TEXT PRIMARY KEY,
                    reflection_id TEXT NOT NULL,
                    companion_id TEXT NOT NULL,
                    model_id TEXT,
                    candidate_type TEXT NOT NULL,
                    draft_text TEXT NOT NULL,
                    priority_score REAL NOT NULL DEFAULT 0.0,
                    channel TEXT NOT NULL DEFAULT 'in_app',
                    approved_at TEXT NOT NULL,
                    delivered_at TEXT,
                    dismissed_at TEXT,
                    expires_at TEXT,
                    visible INTEGER NOT NULL DEFAULT 0,
                    FOREIGN KEY(reflection_id) REFERENCES reflections(id)
                );
                CREATE INDEX IF NOT EXISTS idx_outreach_companion ON outreach_candidates(companion_id, delivered_at, visible);

                CREATE TABLE IF NOT EXISTS outreach_events (
                    id TEXT PRIMARY KEY,
                    outreach_candidate_id TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    metadata_json TEXT,
                    FOREIGN KEY(outreach_candidate_id) REFERENCES outreach_candidates(id)
                );

                CREATE TABLE IF NOT EXISTS lora_runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ran_at TEXT NOT NULL,
                    sessions_used TEXT,
                    adapter_path TEXT,
                    success INTEGER DEFAULT 0,
                    notes TEXT
                );

                CREATE TABLE IF NOT EXISTS job_runs (
                    id TEXT PRIMARY KEY,
                    job_type TEXT NOT NULL,
                    companion_id TEXT,
                    status TEXT NOT NULL,
                    scheduled_for TEXT,
                    started_at TEXT,
                    finished_at TEXT,
                    last_success_at TEXT,
                    error_message TEXT,
                    metadata_json TEXT
                );

                --
                -- Persisted runtime state snapshots for companions.  Each snapshot
                -- records where the serialized state is stored on disk.  The
                -- snapshot_path should be an absolute or project-relative path
                -- to a file under data/states/.  These snapshots allow Flint to
                -- restore a companion's conversation and model state across
                -- restarts and between devices.  Notes can store metadata
                -- about the circumstances under which the snapshot was taken.
                --
                CREATE TABLE IF NOT EXISTS runtime_state_snapshots (
                    id TEXT PRIMARY KEY,
                    companion_id TEXT NOT NULL,
                    model_id TEXT,
                    snapshot_path TEXT NOT NULL,
                    notes TEXT,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(companion_id) REFERENCES companions(id),
                    FOREIGN KEY(model_id) REFERENCES models(id)
                );
                CREATE INDEX IF NOT EXISTS idx_runtime_snapshots_companion
                    ON runtime_state_snapshots(companion_id, created_at DESC);

                --
                -- Versioned LoRA adapter metadata.  Each entry corresponds to
                -- a trained adapter file on disk and carries a monotonically
                -- increasing version number per companion/model combination.
                -- parent_version_id can link to a previous adapter to form a
                -- lineage chain.  Notes may include training context or
                -- evaluation metrics.  Version numbers restart at 1 when no
                -- adapters exist.
                --
                CREATE TABLE IF NOT EXISTS adapter_versions (
                    id TEXT PRIMARY KEY,
                    companion_id TEXT NOT NULL,
                    model_id TEXT,
                    adapter_path TEXT NOT NULL,
                    version INTEGER NOT NULL,
                    parent_version_id TEXT,
                    notes TEXT,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(companion_id) REFERENCES companions(id),
                    FOREIGN KEY(model_id) REFERENCES models(id)
                );
                CREATE INDEX IF NOT EXISTS idx_adapter_versions_companion
                    ON adapter_versions(companion_id, model_id, version DESC);

                CREATE TABLE IF NOT EXISTS settings_overrides (
                    id TEXT PRIMARY KEY,
                    scope_type TEXT NOT NULL,
                    scope_id TEXT,
                    setting_key TEXT NOT NULL,
                    setting_value_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    UNIQUE(scope_type, scope_id, setting_key)
                );

                CREATE TABLE IF NOT EXISTS training_runs (
                    id TEXT PRIMARY KEY,
                    companion_id TEXT,
                    started_at TEXT NOT NULL,
                    completed_at TEXT,
                    steps INTEGER DEFAULT 0,
                    avg_loss REAL,
                    min_loss REAL,
                    loss_curve TEXT,
                    success INTEGER DEFAULT 0
                );
                CREATE INDEX IF NOT EXISTS idx_training_runs_companion
                    ON training_runs(companion_id, started_at DESC);
                """
            )

            tables = {
                r[0] for r in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ).fetchall()
            }
            if "messages_fts" not in tables:
                conn.execute(
                    """
                    CREATE VIRTUAL TABLE messages_fts
                    USING fts5(message_id UNINDEXED, session_id UNINDEXED,
                               role UNINDEXED, content)
                    """
                )

            if not conn.execute("SELECT version FROM schema_version").fetchone():
                conn.execute("INSERT INTO schema_version VALUES (?)", (SCHEMA_VERSION,))
            else:
                conn.execute("UPDATE schema_version SET version = ?", (SCHEMA_VERSION,))

            self._ensure_column(conn, "sessions", "companion_id", "TEXT")
            self._ensure_column(conn, "sessions", "model_id", "TEXT")
            self._ensure_column(conn, "sessions", "message_count", "INTEGER DEFAULT 0")
            self._ensure_column(conn, "sessions", "source", "TEXT DEFAULT 'native'")
            self._ensure_column(conn, "messages", "companion_id", "TEXT")
            self._ensure_column(conn, "messages", "model_id", "TEXT")
            self._ensure_column(conn, "messages", "turn_index", "INTEGER")
            self._ensure_column(conn, "messages", "token_count", "INTEGER")
            self._ensure_column(conn, "messages", "imported", "INTEGER DEFAULT 0")
            self._ensure_column(conn, "messages", "metadata_json", "TEXT")
            self._ensure_column(conn, "reflections", "shown", "INTEGER DEFAULT 0")

    # Exhaustive set of tables that may receive dynamic column additions at init time.
    _ALLOWED_MIGRATION_TABLES = frozenset({"sessions", "messages", "reflections"})

    def _ensure_column(self, conn: sqlite3.Connection, table: str, column: str, definition: str):
        if table not in self._ALLOWED_MIGRATION_TABLES:
            raise ValueError(f"_ensure_column called on unexpected table: {table!r}")
        cols = {row[1] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}
        if column not in cols:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")

    @contextmanager
    def _conn(self):
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        try:
            yield conn
            conn.commit()
        except BaseException:
            conn.rollback()
            raise
        finally:
            conn.close()

    def ensure_default_records(self):
        companion_id = self.get_or_create_default_companion()
        for profile_name, freq, max_per_day, prio, grounded, novelty, active in [
            ("silent", 720, 0, 1.0, 1.0, 1.0, 0),
            ("gentle", 360, 1, 0.68, 0.65, 0.52, 0),
            ("normal", 180, 2, 0.55, 0.55, 0.45, 1),
            ("active", 90, 4, 0.45, 0.50, 0.35, 0),
        ]:
            self.upsert_initiative_profile(
                companion_id,
                profile_name,
                freq,
                max_per_day,
                prio,
                grounded,
                novelty,
                active=active,
            )

    # ── Runtime state snapshots ──────────────────────────────────────────────

    def add_runtime_state_snapshot(
        self,
        companion_id: str,
        model_id: Optional[str],
        snapshot_path: str,
        notes: str = "",
    ) -> str:
        """
        Record a new runtime state snapshot.  This should be called after
        saving the model/session state to disk under the given path.

        Returns the generated snapshot id.
        """
        snapshot_id = str(uuid.uuid4())
        now = datetime.now().isoformat()
        with self._conn() as conn:
            conn.execute(
                """
                INSERT INTO runtime_state_snapshots
                    (id, companion_id, model_id, snapshot_path, notes, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (snapshot_id, companion_id, model_id, snapshot_path, notes, now),
            )
        return snapshot_id

    def get_latest_runtime_state_snapshot(self, companion_id: str, model_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Fetch the most recent runtime state snapshot for the given companion and
        model.  If model_id is None, the most recent snapshot regardless of
        model is returned.  Returns None if no snapshots exist.
        """
        query = "SELECT * FROM runtime_state_snapshots WHERE companion_id = ?"
        params: List[Any] = [companion_id]
        if model_id:
            query += " AND model_id = ?"
            params.append(model_id)
        query += " ORDER BY created_at DESC LIMIT 1"
        with self._conn() as conn:
            row = conn.execute(query, params).fetchone()
        return dict(row) if row else None

    def get_runtime_state_snapshots(self, companion_id: str, model_id: Optional[str] = None, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Return a list of recent runtime state snapshots for the given companion
        and model.  If ``model_id`` is None, snapshots for any model are
        returned.  ``limit`` bounds the number of rows returned.  Results are
        ordered by creation time descending.
        """
        query = "SELECT * FROM runtime_state_snapshots WHERE companion_id = ?"
        params: List[Any] = [companion_id]
        if model_id:
            query += " AND model_id = ?"
            params.append(model_id)
        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)
        with self._conn() as conn:
            rows = conn.execute(query, params).fetchall()
        return [dict(row) for row in rows]

    def delete_runtime_state_snapshots(self, snapshot_ids: List[str]) -> None:
        """Remove snapshot DB rows by id.  Does not touch files on disk."""
        if not snapshot_ids:
            return
        placeholders = ",".join("?" for _ in snapshot_ids)
        with self._conn() as conn:
            conn.execute(
                f"DELETE FROM runtime_state_snapshots WHERE id IN ({placeholders})",
                snapshot_ids,
            )

    # ── Adapter versions ─────────────────────────────────────────────────────

    def add_adapter_version(
        self,
        companion_id: str,
        model_id: Optional[str],
        adapter_path: str,
        notes: str = "",
    ) -> str:
        """
        Record a new adapter version.  Automatically computes the next version
        number for the given companion and model.  Returns the adapter id.
        """
        now = datetime.now().isoformat()
        with self._conn() as conn:
            # Determine next version number and parent
            row = conn.execute(
                "SELECT id, version FROM adapter_versions WHERE companion_id = ? AND model_id IS ? ORDER BY version DESC LIMIT 1",
                (companion_id, model_id),
            ).fetchone()
            if row:
                parent_version_id = row["id"]
                next_version = row["version"] + 1
            else:
                parent_version_id = None
                next_version = 1
            adapter_id = str(uuid.uuid4())
            conn.execute(
                """
                INSERT INTO adapter_versions
                    (id, companion_id, model_id, adapter_path, version, parent_version_id, notes, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (adapter_id, companion_id, model_id, adapter_path, next_version, parent_version_id, notes, now),
            )
        return adapter_id

    def get_latest_adapter_version(self, companion_id: str, model_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Fetch the most recent adapter version for a companion and model.  If
        model_id is None, the latest version regardless of model is returned.
        """
        query = "SELECT * FROM adapter_versions WHERE companion_id = ?"
        params: List[Any] = [companion_id]
        if model_id:
            query += " AND model_id = ?"
            params.append(model_id)
        query += " ORDER BY version DESC LIMIT 1"
        with self._conn() as conn:
            row = conn.execute(query, params).fetchone()
        return dict(row) if row else None

    # ── Companion / model metadata ───────────────────────────────────────────

    def get_or_create_default_companion(self) -> str:
        with self._conn() as conn:
            row = conn.execute("SELECT id FROM companions ORDER BY created_at ASC LIMIT 1").fetchone()
            if row:
                return row["id"]
            companion_id = str(uuid.uuid4())
            now = datetime.now().isoformat()
            conn.execute(
                "INSERT INTO companions (id, name, created_at, updated_at, status) VALUES (?, ?, ?, ?, 'active')",
                (companion_id, config.BOT_NAME, now, now),
            )
            return companion_id

    def upsert_model(self, model_id: str, engine_type: str, family: str, name: str, version: Optional[str], manifest_json: str):
        now = datetime.now().isoformat()
        with self._conn() as conn:
            conn.execute(
                """
                INSERT INTO models (id, engine_type, family, name, version, manifest_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    engine_type=excluded.engine_type,
                    family=excluded.family,
                    name=excluded.name,
                    version=excluded.version,
                    manifest_json=excluded.manifest_json
                """,
                (model_id, engine_type, family, name, version, manifest_json, now),
            )

    def upsert_model_installation(self, model_id: str, install_path: str, backend: str, is_default: bool, verification_status: str):
        now = datetime.now().isoformat()
        installation_id = f"{model_id}::{install_path}"
        with self._conn() as conn:
            conn.execute(
                """
                INSERT INTO model_installations
                    (id, model_id, install_path, backend, is_available, is_default, verification_status, last_verified_at, created_at, updated_at)
                VALUES (?, ?, ?, ?, 1, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    backend=excluded.backend,
                    is_available=1,
                    is_default=excluded.is_default,
                    verification_status=excluded.verification_status,
                    last_verified_at=excluded.last_verified_at,
                    updated_at=excluded.updated_at
                """,
                (installation_id, model_id, install_path, backend, 1 if is_default else 0, verification_status, now, now, now),
            )

    def get_model(self, model_id: Optional[str]) -> Optional[Dict[str, Any]]:
        if not model_id:
            return None
        with self._conn() as conn:
            row = conn.execute("SELECT * FROM models WHERE id = ?", (model_id,)).fetchone()
        return dict(row) if row else None

    def upsert_initiative_profile(
        self,
        companion_id: str,
        profile_name: str,
        reflection_frequency_minutes: int,
        outreach_max_per_day: int,
        minimum_priority_threshold: float,
        minimum_groundedness_threshold: float,
        minimum_novelty_threshold: float,
        active: int = 0,
    ):
        now = datetime.now().isoformat()
        profile_id = f"{companion_id}::{profile_name}"
        with self._conn() as conn:
            conn.execute(
                """
                INSERT INTO initiative_profiles
                    (id, companion_id, profile_name, reflection_frequency_minutes, outreach_max_per_day,
                     minimum_priority_threshold, minimum_groundedness_threshold, minimum_novelty_threshold,
                     created_at, updated_at, active)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    reflection_frequency_minutes=excluded.reflection_frequency_minutes,
                    outreach_max_per_day=excluded.outreach_max_per_day,
                    minimum_priority_threshold=excluded.minimum_priority_threshold,
                    minimum_groundedness_threshold=excluded.minimum_groundedness_threshold,
                    minimum_novelty_threshold=excluded.minimum_novelty_threshold,
                    updated_at=excluded.updated_at,
                    active=excluded.active
                """,
                (
                    profile_id,
                    companion_id,
                    profile_name,
                    reflection_frequency_minutes,
                    outreach_max_per_day,
                    minimum_priority_threshold,
                    minimum_groundedness_threshold,
                    minimum_novelty_threshold,
                    now,
                    now,
                    active,
                ),
            )
            if active:
                conn.execute(
                    "UPDATE initiative_profiles SET active = 0 WHERE companion_id = ? AND id != ?",
                    (companion_id, profile_id),
                )
                conn.execute(
                    "UPDATE companions SET active_initiative_profile_id = ?, updated_at = ? WHERE id = ?",
                    (profile_id, now, companion_id),
                )

    def get_active_initiative_profile(self, companion_id: str) -> Dict[str, Any]:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM initiative_profiles WHERE companion_id = ? AND active = 1 LIMIT 1",
                (companion_id,),
            ).fetchone()
            if not row:
                row = conn.execute(
                    "SELECT * FROM initiative_profiles WHERE companion_id = ? ORDER BY profile_name = 'normal' DESC, created_at ASC LIMIT 1",
                    (companion_id,),
                ).fetchone()
        return dict(row) if row else {
            "profile_name": "normal",
            "reflection_frequency_minutes": 180,
            "outreach_max_per_day": 2,
            "minimum_priority_threshold": 0.55,
            "minimum_groundedness_threshold": 0.55,
            "minimum_novelty_threshold": 0.45,
        }

    def get_initiative_profiles(self, companion_id: str) -> List[Dict[str, Any]]:
        """Return all initiative profiles for the given companion.

        If no profiles exist for the companion, an empty list is returned.
        """
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM initiative_profiles WHERE companion_id = ? ORDER BY created_at ASC",
                (companion_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    # ── Sessions ───────────────────────────────────────────────────────────────

    def new_session(self, companion_id: Optional[str] = None, model_id: Optional[str] = None) -> str:
        session_id = str(uuid.uuid4())[:8]
        now = datetime.now().isoformat()
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO sessions (id, companion_id, model_id, started_at) VALUES (?, ?, ?, ?)",
                (session_id, companion_id, model_id, now),
            )
            if companion_id:
                conn.execute(
                    "UPDATE companions SET last_session_id = ?, last_active_at = ?, updated_at = ? WHERE id = ?",
                    (session_id, now, now, companion_id),
                )
        log.debug("New session: %s", session_id)
        return session_id

    def end_session(self, session_id: str, lora_version: str = None):
        with self._conn() as conn:
            conn.execute(
                "UPDATE sessions SET ended_at = ?, lora_version = ? WHERE id = ?",
                (datetime.now().isoformat(), lora_version, session_id),
            )

    def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        companion_id: Optional[str] = None,
        model_id: Optional[str] = None,
        turn_index: Optional[int] = None,
    ) -> int:
        ts = datetime.now().isoformat()
        with self._conn() as conn:
            cursor = conn.execute(
                """
                INSERT INTO messages (session_id, companion_id, model_id, role, content, timestamp, turn_index)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (session_id, companion_id, model_id, role, content, ts, turn_index),
            )
            message_id = cursor.lastrowid
            conn.execute(
                "UPDATE sessions SET message_count = message_count + 1 WHERE id = ?",
                (session_id,),
            )
            if role == "assistant":
                conn.execute(
                    "UPDATE sessions SET turn_count = turn_count + 1 WHERE id = ?",
                    (session_id,),
                )
            conn.execute(
                "INSERT INTO messages_fts (message_id, session_id, role, content) VALUES (?, ?, ?, ?)",
                (message_id, session_id, role, content),
            )
        return int(message_id)

    def search(self, query: str, limit: int = 10) -> list:
        if not query.strip():
            return []
        try:
            with self._conn() as conn:
                rows = conn.execute(
                    """
                    SELECT timestamp, role, content, session_id
                    FROM messages
                    WHERE id IN (
                        SELECT CAST(message_id AS INTEGER)
                        FROM messages_fts
                        WHERE messages_fts MATCH ?
                    )
                    ORDER BY timestamp DESC
                    LIMIT ?
                    """,
                    (query, limit),
                ).fetchall()
            return rows
        except sqlite3.OperationalError:
            with self._conn() as conn:
                rows = conn.execute(
                    """
                    SELECT timestamp, role, content, session_id
                    FROM messages
                    WHERE content LIKE ?
                    ORDER BY timestamp DESC
                    LIMIT ?
                    """,
                    (f"%{query}%", limit),
                ).fetchall()
            return rows

    def get_recent_messages(self, limit: int = 50, session_id: str | None = None) -> list:
        with self._conn() as conn:
            if session_id is not None:
                rows = conn.execute(
                    "SELECT timestamp, role, content, session_id FROM messages "
                    "WHERE session_id = ? ORDER BY timestamp DESC LIMIT ?",
                    (session_id, limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT timestamp, role, content, session_id FROM messages ORDER BY timestamp DESC LIMIT ?",
                    (limit,),
                ).fetchall()
        return list(reversed(rows))

    def get_messages_after_last_block(self, companion_id: str) -> List[Dict[str, Any]]:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT MAX(end_message_id) AS last_id FROM conversation_blocks WHERE companion_id = ?",
                (companion_id,),
            ).fetchone()
            last_id = row["last_id"] or 0
            rows = conn.execute(
                """
                SELECT id, session_id, role, content, timestamp, companion_id
                FROM messages
                WHERE companion_id = ? AND id > ?
                ORDER BY id ASC
                """,
                (companion_id, last_id),
            ).fetchall()
        return [dict(r) for r in rows]

    def create_conversation_block(
        self,
        companion_id: str,
        model_id: str,
        session_id: str,
        start_message_id: int,
        end_message_id: int,
        block_text: str,
        message_count: int,
        turn_count: int,
        block_type: str = "recent",
    ) -> str:
        block_id = str(uuid.uuid4())
        with self._conn() as conn:
            conn.execute(
                """
                INSERT INTO conversation_blocks
                    (id, session_id, companion_id, model_id, start_message_id, end_message_id, block_type, block_text, message_count, turn_count, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    block_id,
                    session_id,
                    companion_id,
                    model_id,
                    start_message_id,
                    end_message_id,
                    block_type,
                    block_text,
                    message_count,
                    turn_count,
                    datetime.now().isoformat(),
                ),
            )
        return block_id

    def get_unsummarized_blocks(self, companion_id: str, limit: int = 5) -> List[Dict[str, Any]]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM conversation_blocks WHERE companion_id = ? AND summarized_at IS NULL ORDER BY created_at ASC LIMIT ?",
                (companion_id, limit),
            ).fetchall()
        return [dict(r) for r in rows]

    def mark_block_summarized(self, block_id: str):
        with self._conn() as conn:
            conn.execute(
                "UPDATE conversation_blocks SET summarized_at = ? WHERE id = ?",
                (datetime.now().isoformat(), block_id),
            )

    def add_summary(
        self,
        companion_id: str,
        model_id: str,
        source_block_id: Optional[str],
        source_summary_ids: Optional[Iterable[str]],
        summary_level: str,
        summary_type: str,
        summary_text: str,
        key_topics: Optional[Iterable[str]],
        open_loops: Optional[Iterable[str]],
        signals: Optional[Dict[str, Any]],
        coverage_start_at: Optional[str],
        coverage_end_at: Optional[str],
    ) -> str:
        summary_id = str(uuid.uuid4())
        with self._conn() as conn:
            conn.execute(
                """
                INSERT INTO summaries
                    (id, companion_id, model_id, source_block_id, source_summary_ids_json, summary_level, summary_type,
                     summary_text, key_topics_json, open_loops_json, signals_json, coverage_start_at, coverage_end_at, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    summary_id,
                    companion_id,
                    model_id,
                    source_block_id,
                    json.dumps(list(source_summary_ids or [])),
                    summary_level,
                    summary_type,
                    summary_text,
                    json.dumps(list(key_topics or [])),
                    json.dumps(list(open_loops or [])),
                    json.dumps(signals or {}),
                    coverage_start_at,
                    coverage_end_at,
                    datetime.now().isoformat(),
                ),
            )
        return summary_id

    def get_recent_summaries(self, companion_id: str, level: Optional[str], limit: int = 5) -> List[Dict[str, Any]]:
        with self._conn() as conn:
            if level:
                rows = conn.execute(
                    "SELECT * FROM summaries WHERE companion_id = ? AND summary_level = ? ORDER BY created_at DESC LIMIT ?",
                    (companion_id, level, limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM summaries WHERE companion_id = ? ORDER BY created_at DESC LIMIT ?",
                    (companion_id, limit),
                ).fetchall()
        return [self._decode_summary_row(dict(r)) for r in rows]

    def sample_historical_summaries(self, companion_id: str, exclude_ids: List[str], limit: int) -> List[Dict[str, Any]]:
        placeholders = ",".join("?" for _ in exclude_ids) if exclude_ids else ""
        with self._conn() as conn:
            if exclude_ids:
                rows = conn.execute(
                    f"SELECT * FROM summaries WHERE companion_id = ? AND id NOT IN ({placeholders}) ORDER BY created_at DESC LIMIT 50",
                    [companion_id, *exclude_ids],
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM summaries WHERE companion_id = ? ORDER BY created_at DESC LIMIT 50",
                    (companion_id,),
                ).fetchall()
        decoded = [self._decode_summary_row(dict(r)) for r in rows]
        if len(decoded) <= limit:
            return decoded
        # Weighted recent-biased sampling without repeats.
        sample_pool = decoded[:]
        selected: List[Dict[str, Any]] = []
        while sample_pool and len(selected) < limit:
            weights = [1 / (idx + 1) for idx, _ in enumerate(sample_pool)]
            pick = sample_pool.pop(self._weighted_choice(weights))
            selected.append(pick)
        return selected

    def _weighted_choice(self, weights: List[float]) -> int:
        total = sum(weights)
        r = total * __import__("random").random()
        upto = 0.0
        for idx, weight in enumerate(weights):
            upto += weight
            if upto >= r:
                return idx
        return len(weights) - 1

    def add_reflection(
        self,
        companion_id: str,
        model_id: str,
        reflection_type: str,
        input_bundle: Dict[str, Any],
        reflection_text: str,
        question_text: Optional[str],
        supporting_summary_ids: List[str],
        supporting_memory_ids: List[str],
        novelty_score: float,
        relevance_score: float,
        groundedness_score: float,
        sensitivity_score: float,
        priority_score: float,
    ) -> str:
        reflection_id = str(uuid.uuid4())
        with self._conn() as conn:
            conn.execute(
                """
                INSERT INTO reflections
                    (id, companion_id, model_id, reflection_type, input_bundle_json, reflection_text, question_text,
                     supporting_summary_ids_json, supporting_memory_ids_json, novelty_score, relevance_score,
                     groundedness_score, sensitivity_score, priority_score, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    reflection_id,
                    companion_id,
                    model_id,
                    reflection_type,
                    json.dumps(input_bundle),
                    reflection_text,
                    question_text,
                    json.dumps(supporting_summary_ids),
                    json.dumps(supporting_memory_ids),
                    novelty_score,
                    relevance_score,
                    groundedness_score,
                    sensitivity_score,
                    priority_score,
                    datetime.now().isoformat(),
                ),
            )
        return reflection_id

    def get_recent_thought(self, companion_id: str, reflection_type: str) -> Optional[Dict[str, Any]]:
        """Return the most recently created reflection of a given type."""
        with self._conn() as conn:
            row = conn.execute(
                """SELECT reflection_text, question_text, created_at FROM reflections
                   WHERE companion_id = ? AND reflection_type = ?
                   ORDER BY created_at DESC LIMIT 1""",
                (companion_id, reflection_type),
            ).fetchone()
        return dict(row) if row else None

    def get_dream_texts(self, companion_id: str, limit: int = 50) -> List[str]:
        """Return recent dream reflection texts for LoRA training."""
        with self._conn() as conn:
            rows = conn.execute(
                """SELECT reflection_text FROM reflections
                   WHERE companion_id = ? AND reflection_type = 'dream'
                   ORDER BY created_at DESC LIMIT ?""",
                (companion_id, limit),
            ).fetchall()
        return [r["reflection_text"] for r in rows if r["reflection_text"]]

    def get_top_dream_thought(self, companion_id: str) -> Optional[Dict[str, str]]:
        """Return the highest-ranked unseen dream thought for the welcome-back message."""
        with self._conn() as conn:
            row = conn.execute(
                """SELECT id, reflection_text as text FROM reflections
                   WHERE companion_id = ? AND reflection_type = 'dream' AND shown = 0
                   ORDER BY priority_score * novelty_score DESC LIMIT 1""",
                (companion_id,),
            ).fetchone()
        return dict(row) if row else None

    def mark_dream_shown(self, reflection_id: str):
        """Mark a dream reflection as shown so it is not surfaced again."""
        with self._conn() as conn:
            conn.execute("UPDATE reflections SET shown = 1 WHERE id = ?", (reflection_id,))

    def get_new_reflections(self, companion_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM reflections WHERE companion_id = ? AND status = 'new' ORDER BY priority_score DESC, created_at DESC LIMIT ?",
                (companion_id, limit),
            ).fetchall()
        return [dict(r) for r in rows]

    def update_reflection_status(self, reflection_id: str, status: str, reason: str):
        with self._conn() as conn:
            conn.execute(
                "UPDATE reflections SET status = ?, gating_reason = ? WHERE id = ?",
                (status, reason, reflection_id),
            )

    def create_outreach_candidate(
        self,
        reflection_id: str,
        companion_id: str,
        model_id: str,
        candidate_type: str,
        draft_text: str,
        priority_score: float,
        channel: str,
    ) -> str:
        candidate_id = str(uuid.uuid4())
        with self._conn() as conn:
            conn.execute(
                """
                INSERT INTO outreach_candidates
                    (id, reflection_id, companion_id, model_id, candidate_type, draft_text, priority_score, channel, approved_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    candidate_id,
                    reflection_id,
                    companion_id,
                    model_id,
                    candidate_type,
                    draft_text,
                    priority_score,
                    channel,
                    datetime.now().isoformat(),
                ),
            )
        return candidate_id

    def mark_ready_outreach_visible(self, companion_id: str) -> int:
        with self._conn() as conn:
            cur = conn.execute(
                "UPDATE outreach_candidates SET visible = 1 WHERE companion_id = ? AND visible = 0 AND delivered_at IS NULL",
                (companion_id,),
            )
            return cur.rowcount

    def get_visible_outreach(self, companion_id: str, limit: int = 5) -> List[Dict[str, Any]]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM outreach_candidates WHERE companion_id = ? AND visible = 1 AND dismissed_at IS NULL ORDER BY approved_at DESC LIMIT ?",
                (companion_id, limit),
            ).fetchall()
        return [dict(r) for r in rows]

    def mark_outreach_delivered(self, candidate_id: str, event_type: str = "shown"):
        now = datetime.now().isoformat()
        with self._conn() as conn:
            conn.execute(
                "UPDATE outreach_candidates SET delivered_at = COALESCE(delivered_at, ?) WHERE id = ?",
                (now, candidate_id),
            )
            conn.execute(
                "INSERT INTO outreach_events (id, outreach_candidate_id, event_type, created_at) VALUES (?, ?, ?, ?)",
                (str(uuid.uuid4()), candidate_id, event_type, now),
            )

    def dismiss_outreach(self, candidate_id: str):
        now = datetime.now().isoformat()
        with self._conn() as conn:
            conn.execute("UPDATE outreach_candidates SET dismissed_at = ? WHERE id = ?", (now, candidate_id))
            conn.execute(
                "INSERT INTO outreach_events (id, outreach_candidate_id, event_type, created_at) VALUES (?, ?, 'dismissed', ?)",
                (str(uuid.uuid4()), candidate_id, now),
            )

    def count_outreach_today(self, companion_id: str) -> int:
        start = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0).isoformat()
        with self._conn() as conn:
            row = conn.execute(
                "SELECT COUNT(*) AS c FROM outreach_candidates WHERE companion_id = ? AND approved_at >= ?",
                (companion_id, start),
            ).fetchone()
        return int(row["c"])

    def recent_outreach_exists(self, companion_id: str, text: str, hours: int = 24) -> bool:
        cutoff = (datetime.now() - timedelta(hours=hours)).isoformat()
        needle = text[:80]
        with self._conn() as conn:
            row = conn.execute(
                "SELECT 1 FROM outreach_candidates WHERE companion_id = ? AND approved_at >= ? AND draft_text LIKE ? LIMIT 1",
                (companion_id, cutoff, f"%{needle[:20]}%"),
            ).fetchone()
        return bool(row)

    def upsert_semantic_memory(
        self,
        companion_id: str,
        model_id: str,
        memory_type: str,
        title: str,
        content: str,
        importance_score: float,
    ):
        now = datetime.now().isoformat()
        with self._conn() as conn:
            row = conn.execute(
                "SELECT id FROM semantic_memory WHERE companion_id = ? AND memory_type = ? AND title = ? AND archived_at IS NULL LIMIT 1",
                (companion_id, memory_type, title),
            ).fetchone()
            if row:
                conn.execute(
                    """
                    UPDATE semantic_memory
                    SET content = ?, importance_score = ?, recency_score = 1.0, stability_score = MIN(1.0, stability_score + 0.1), last_reinforced_at = ?
                    WHERE id = ?
                    """,
                    (content, importance_score, now, row["id"]),
                )
            else:
                conn.execute(
                    """
                    INSERT INTO semantic_memory
                        (id, companion_id, model_id, memory_type, title, content, importance_score, recency_score, stability_score, last_reinforced_at, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, 1.0, 0.5, ?, ?)
                    """,
                    (str(uuid.uuid4()), companion_id, model_id, memory_type, title, content, importance_score, now, now),
                )

    def get_semantic_memory(self, companion_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM semantic_memory WHERE companion_id = ? AND archived_at IS NULL ORDER BY importance_score DESC, last_reinforced_at DESC LIMIT ?",
                (companion_id, limit),
            ).fetchall()
        return [dict(r) for r in rows]

    # ── Job runs ──────────────────────────────────────────────────────────────

    def log_job_run(self, job_type: str, companion_id: Optional[str], status: str, scheduled_for: Optional[str]) -> str:
        run_id = str(uuid.uuid4())
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO job_runs (id, job_type, companion_id, status, scheduled_for, started_at) VALUES (?, ?, ?, ?, ?, ?)",
                (run_id, job_type, companion_id, status, scheduled_for, datetime.now().isoformat()),
            )
        return run_id

    def finish_job_run(self, run_id: str, status: str, metadata: Optional[Dict[str, Any]] = None, error_message: Optional[str] = None):
        with self._conn() as conn:
            conn.execute(
                "UPDATE job_runs SET status = ?, finished_at = ?, last_success_at = CASE WHEN ? = 'success' THEN ? ELSE last_success_at END, error_message = ?, metadata_json = ? WHERE id = ?",
                (status, datetime.now().isoformat(), status, datetime.now().isoformat(), error_message, json.dumps(metadata or {}), run_id),
            )

    # ── LoRA data ──────────────────────────────────────────────────────────────

    def get_unprocessed_sessions(self) -> list:
        try:
            with self._conn() as conn:
                rows = conn.execute(
                    """
                    SELECT s.id FROM sessions s
                    WHERE s.ended_at IS NOT NULL
                    AND s.id NOT IN (
                        SELECT json_each.value
                        FROM lora_runs, json_each(lora_runs.sessions_used)
                        WHERE lora_runs.success = 1
                    )
                    ORDER BY s.started_at ASC
                    """
                ).fetchall()
            return [r["id"] for r in rows]
        except sqlite3.OperationalError:
            log.warning("json_each unavailable — returning all completed sessions for LoRA")
            with self._conn() as conn:
                rows = conn.execute(
                    "SELECT id FROM sessions WHERE ended_at IS NOT NULL ORDER BY started_at ASC"
                ).fetchall()
            return [r["id"] for r in rows]

    def get_session_as_training_text(self, session_id: str) -> Optional[str]:
        with self._conn() as conn:
            messages = conn.execute(
                "SELECT role, content FROM messages WHERE session_id = ? ORDER BY timestamp ASC",
                (session_id,),
            ).fetchall()
        if not messages:
            return None
        lines = []
        for role, content in messages:
            prefix = "User" if role == "user" else "Assistant"
            lines.append(f"{prefix}: {content}")
        return "\n\n".join(lines)

    def get_random_old_sessions(self, n: int) -> list:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT id FROM sessions WHERE ended_at IS NOT NULL ORDER BY RANDOM() LIMIT ?",
                (n,),
            ).fetchall()
        return [r["id"] for r in rows]

    # ── Training run history ──────────────────────────────────────────────────

    def begin_training_run(self, companion_id: str | None) -> str:
        run_id = str(uuid.uuid4())
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO training_runs (id, companion_id, started_at) VALUES (?, ?, ?)",
                (run_id, companion_id, datetime.now().isoformat()),
            )
        return run_id

    def update_training_run(
        self,
        run_id: str,
        steps: int,
        avg_loss: float,
        min_loss: float,
        loss_curve: list,
        success: bool,
    ):
        with self._conn() as conn:
            conn.execute(
                """UPDATE training_runs
                   SET completed_at = ?, steps = ?, avg_loss = ?, min_loss = ?,
                       loss_curve = ?, success = ?
                   WHERE id = ?""",
                (
                    datetime.now().isoformat(),
                    steps,
                    avg_loss,
                    min_loss,
                    json.dumps(loss_curve),
                    1 if success else 0,
                    run_id,
                ),
            )

    def get_training_history(self, companion_id: str | None = None, limit: int = 10) -> list:
        with self._conn() as conn:
            if companion_id:
                rows = conn.execute(
                    """SELECT * FROM training_runs WHERE companion_id = ?
                       ORDER BY started_at DESC LIMIT ?""",
                    (companion_id, limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM training_runs ORDER BY started_at DESC LIMIT ?",
                    (limit,),
                ).fetchall()
        return [dict(r) for r in rows]

    def log_lora_run(self, sessions_used: list, adapter_path: str, success: bool, notes: str = ""):
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO lora_runs (ran_at, sessions_used, adapter_path, success, notes) VALUES (?, ?, ?, ?, ?)",
                (datetime.now().isoformat(), json.dumps(sessions_used), adapter_path, 1 if success else 0, notes),
            )

    def stats(self) -> dict:
        with self._conn() as conn:
            total_sessions = conn.execute(
                "SELECT COUNT(*) AS c FROM sessions WHERE ended_at IS NOT NULL"
            ).fetchone()["c"]
            total_messages = conn.execute(
                "SELECT COUNT(*) AS c FROM messages"
            ).fetchone()["c"]
            lora_runs = conn.execute(
                "SELECT COUNT(*) AS c FROM lora_runs WHERE success = 1"
            ).fetchone()["c"]
            reflection_count = conn.execute(
                "SELECT COUNT(*) AS c FROM reflections"
            ).fetchone()["c"]
            pending_outreach = conn.execute(
                "SELECT COUNT(*) AS c FROM outreach_candidates WHERE visible = 1 AND dismissed_at IS NULL"
            ).fetchone()["c"]
            try:
                unprocessed = conn.execute(
                    """
                    SELECT COUNT(*) AS c FROM sessions
                    WHERE ended_at IS NOT NULL
                    AND id NOT IN (
                        SELECT json_each.value
                        FROM lora_runs, json_each(lora_runs.sessions_used)
                        WHERE lora_runs.success = 1
                    )
                    """
                ).fetchone()["c"]
            except Exception:
                unprocessed = 0
        return {
            "total_sessions": total_sessions,
            "total_messages": total_messages,
            "lora_runs": lora_runs,
            "unprocessed_sessions": unprocessed,
            "reflections": reflection_count,
            "pending_outreach": pending_outreach,
        }

    def _decode_summary_row(self, row: Dict[str, Any]) -> Dict[str, Any]:
        row["key_topics"] = json.loads(row.get("key_topics_json") or "[]")
        row["open_loops"] = json.loads(row.get("open_loops_json") or "[]")
        row["signals"] = json.loads(row.get("signals_json") or "{}")
        row["source_summary_ids"] = json.loads(row.get("source_summary_ids_json") or "[]")
        return row


class Session:
    """Manages a single active conversation session."""

    def __init__(self, db: ConversationDB, companion_id: Optional[str] = None, model_id: Optional[str] = None):
        self.db = db
        self.companion_id = companion_id
        self.model_id = model_id
        self.session_id = db.new_session(companion_id=companion_id, model_id=model_id)
        self.turn_count = 0
        self.started_at = datetime.now()

    def add_turn(self, user_input: str, bot_response: str):
        turn_index = self.turn_count + 1
        self.db.add_message(
            self.session_id,
            "user",
            user_input,
            companion_id=self.companion_id,
            model_id=self.model_id,
            turn_index=turn_index,
        )
        self.db.add_message(
            self.session_id,
            "assistant",
            bot_response,
            companion_id=self.companion_id,
            model_id=self.model_id,
            turn_index=turn_index,
        )
        self.turn_count += 1

    def end(self, lora_version: str = None):
        self.db.end_session(self.session_id, lora_version)
        duration = datetime.now() - self.started_at
        log.info("Session %s ended: %s turns, %s", self.session_id, self.turn_count, duration)

"""SQLite-WAL observation envelopes and immutable snapshot watermarks."""

from __future__ import annotations

import json
import sqlite3
import threading
import time
import uuid
from collections.abc import Mapping
from pathlib import Path
from typing import Any

STORE_SCHEMA_VERSION = 1


class ObservationStore:
    """Sole resource-agentd-owned writer for normalized observations."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(self.path, check_same_thread=False)
        self._connection.row_factory = sqlite3.Row
        self._lock = threading.RLock()
        with self._connection:
            mode = self._connection.execute("PRAGMA journal_mode=WAL").fetchone()[0]
            if str(mode).lower() != "wal":
                raise RuntimeError(f"SQLite refused WAL mode: {mode!r}")
            self._connection.execute("PRAGMA foreign_keys=ON")
            self._connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS observations (
                    ingestion_sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                    observation_id TEXT NOT NULL UNIQUE,
                    run_id TEXT NOT NULL,
                    trace_id TEXT NOT NULL,
                    call_id TEXT NOT NULL,
                    workspace_scope TEXT NOT NULL,
                    canonicalizer_version TEXT NOT NULL,
                    command_digest TEXT NOT NULL,
                    observation_start REAL,
                    observation_end REAL,
                    telemetry_eligible INTEGER NOT NULL,
                    ingest_eligible INTEGER NOT NULL,
                    rejection_reasons_json TEXT NOT NULL,
                    envelope_json TEXT NOT NULL,
                    visible INTEGER NOT NULL DEFAULT 0,
                    ingested_at REAL NOT NULL
                );
                CREATE INDEX IF NOT EXISTS observations_snapshot
                    ON observations(visible, ingestion_sequence, workspace_scope);
                CREATE TABLE IF NOT EXISTS snapshots (
                    snapshot_id TEXT PRIMARY KEY,
                    max_ingestion_sequence INTEGER NOT NULL,
                    created_at REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS snapshot_observations (
                    snapshot_id TEXT NOT NULL REFERENCES snapshots(snapshot_id),
                    observation_id TEXT NOT NULL REFERENCES observations(observation_id),
                    PRIMARY KEY (snapshot_id, observation_id)
                );
                """
            )

    def close(self) -> None:
        with self._lock:
            self._connection.close()

    def insert_observation(self, envelope: Mapping[str, Any]) -> tuple[bool, int]:
        required = {
            "observation_id",
            "run_id",
            "trace_id",
            "call_id",
            "workspace_scope",
            "canonicalizer_version",
            "command_digest",
            "observation_interval",
            "telemetry_eligible",
            "ingest_eligible",
            "rejection_reasons",
        }
        if not required <= set(envelope):
            raise ValueError("normalized observation envelope is incomplete")
        interval = envelope["observation_interval"]
        if not isinstance(interval, Mapping):
            raise ValueError("observation_interval must be an object")
        envelope_json = json.dumps(dict(envelope), sort_keys=True)
        with self._lock, self._connection:
            cursor = self._connection.execute(
                """
                INSERT OR IGNORE INTO observations (
                    observation_id, run_id, trace_id, call_id, workspace_scope,
                    canonicalizer_version, command_digest, observation_start,
                    observation_end, telemetry_eligible, ingest_eligible,
                    rejection_reasons_json, envelope_json, visible, ingested_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, ?)
                """,
                (
                    str(envelope["observation_id"]),
                    str(envelope["run_id"]),
                    str(envelope["trace_id"]),
                    str(envelope["call_id"]),
                    str(envelope["workspace_scope"]),
                    str(envelope["canonicalizer_version"]),
                    str(envelope["command_digest"]),
                    interval.get("start"),
                    interval.get("end"),
                    int(envelope["telemetry_eligible"] is True),
                    int(envelope["ingest_eligible"] is True),
                    json.dumps(envelope["rejection_reasons"], sort_keys=True),
                    envelope_json,
                    time.time(),
                ),
            )
            row = self._connection.execute(
                "SELECT ingestion_sequence, envelope_json FROM observations "
                "WHERE observation_id=?",
                (str(envelope["observation_id"]),),
            ).fetchone()
        assert row is not None
        if row["envelope_json"] != envelope_json:
            raise ValueError(
                "observation_id was reused with a different normalized envelope"
            )
        return cursor.rowcount == 1, int(row["ingestion_sequence"])

    def promote_observations(self, observation_ids: set[str]) -> int:
        if not observation_ids:
            return 0
        placeholders = ",".join("?" for _ in observation_ids)
        with self._lock, self._connection:
            cursor = self._connection.execute(
                "UPDATE observations SET visible=1 "
                "WHERE visible=0 AND ingest_eligible=1 "
                f"AND observation_id IN ({placeholders})",
                sorted(observation_ids),
            )
        return cursor.rowcount

    def create_snapshot(self) -> str:
        with self._lock, self._connection:
            row = self._connection.execute(
                "SELECT COALESCE(MAX(ingestion_sequence), 0) AS watermark "
                "FROM observations WHERE visible=1"
            ).fetchone()
            snapshot_id = uuid.uuid4().hex
            self._connection.execute(
                "INSERT INTO snapshots VALUES (?, ?, ?)",
                (snapshot_id, int(row["watermark"]), time.time()),
            )
            self._connection.execute(
                """
                INSERT INTO snapshot_observations
                    (snapshot_id, observation_id)
                SELECT ?, observation_id FROM observations
                WHERE visible=1 AND ingest_eligible=1
                """,
                (snapshot_id,),
            )
        return snapshot_id

    def require_snapshot(self, snapshot_id: str) -> int:
        with self._lock:
            row = self._connection.execute(
                "SELECT max_ingestion_sequence FROM snapshots WHERE snapshot_id=?",
                (snapshot_id,),
            ).fetchone()
        if row is None:
            raise ValueError(f"unknown snapshot {snapshot_id!r}")
        return int(row["max_ingestion_sequence"])

    def observations_for_snapshot(
        self,
        snapshot_id: str,
        workspace_scope: str | None = None,
    ) -> list[dict[str, Any]]:
        """Return snapshot envelopes, optionally restricted to one workspace."""

        self.require_snapshot(snapshot_id)
        scope_clause = "" if workspace_scope is None else "AND o.workspace_scope=?"
        parameters = (
            (snapshot_id,)
            if workspace_scope is None
            else (snapshot_id, workspace_scope)
        )
        with self._lock:
            rows = self._connection.execute(
                f"""
                SELECT o.envelope_json
                FROM observations AS o
                JOIN snapshot_observations AS s
                  ON s.observation_id=o.observation_id
                WHERE s.snapshot_id=? {scope_clause}
                ORDER BY o.ingestion_sequence
                """,
                parameters,
            ).fetchall()
        return [json.loads(row["envelope_json"]) for row in rows]

    def observation_count(self) -> int:
        with self._lock:
            return int(
                self._connection.execute(
                    "SELECT COUNT(*) FROM observations"
                ).fetchone()[0]
            )


__all__ = ["ObservationStore", "STORE_SCHEMA_VERSION"]

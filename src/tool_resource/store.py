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

STORE_SCHEMA_VERSION = 2


class ObservationStore:
    """Sole resource-agentd-owned writer for normalized observations.

    A snapshot is a promotion watermark, not a materialized membership list.
    Observations become visible in promotion batches with a monotonic
    ``promotion_sequence``; a snapshot records the highest batch that existed
    when it was taken, so a later promotion can never enter an earlier
    snapshot. Storing membership per snapshot instead cost one row per visible
    observation per snapshot, which grew as runs x observations without end.
    """

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
            self._require_current_schema()
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
                    promotion_sequence INTEGER,
                    ingested_at REAL NOT NULL
                );
                CREATE INDEX IF NOT EXISTS observations_snapshot
                    ON observations(promotion_sequence, workspace_scope);
                CREATE TABLE IF NOT EXISTS snapshots (
                    snapshot_id TEXT PRIMARY KEY,
                    max_promotion_sequence INTEGER NOT NULL,
                    created_at REAL NOT NULL
                );
                """
            )

    def _require_current_schema(self) -> None:
        """Fail loudly on a store written by an older, incompatible schema."""

        existing = self._connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='observations'"
        ).fetchone()
        if existing is None:
            return
        columns = {
            row["name"]
            for row in self._connection.execute("PRAGMA table_info(observations)")
        }
        if "promotion_sequence" not in columns:
            raise RuntimeError(
                f"{self.path} predates store schema {STORE_SCHEMA_VERSION} "
                "(no promotion_sequence column); it cannot be read by this "
                "build. Point --database at a new file."
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
                    rejection_reasons_json, envelope_json, promotion_sequence,
                    ingested_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, ?)
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
        """Make eligible observations visible as one atomic promotion batch."""

        if not observation_ids:
            return 0
        placeholders = ",".join("?" for _ in observation_ids)
        with self._lock, self._connection:
            # Load-bearing invariant: rows are never deleted from `observations`,
            # so MAX(promotion_sequence) only ever grows and every batch is
            # strictly above every existing watermark. Pruning this table would
            # let a new batch reuse a number at or below a live watermark, which
            # silently admits a late promotion into an earlier snapshot -- the
            # one property snapshots exist to provide. Prune by copying to a new
            # database, never in place.
            batch = int(
                self._connection.execute(
                    "SELECT COALESCE(MAX(promotion_sequence), 0) + 1 FROM observations"
                ).fetchone()[0]
            )
            cursor = self._connection.execute(
                "UPDATE observations SET promotion_sequence=? "
                "WHERE promotion_sequence IS NULL AND ingest_eligible=1 "
                f"AND observation_id IN ({placeholders})",
                (batch, *sorted(observation_ids)),
            )
        return cursor.rowcount

    def create_snapshot(self) -> str:
        with self._lock, self._connection:
            row = self._connection.execute(
                "SELECT COALESCE(MAX(promotion_sequence), 0) AS watermark "
                "FROM observations"
            ).fetchone()
            snapshot_id = uuid.uuid4().hex
            self._connection.execute(
                "INSERT INTO snapshots VALUES (?, ?, ?)",
                (snapshot_id, int(row["watermark"]), time.time()),
            )
        return snapshot_id

    def require_snapshot(self, snapshot_id: str) -> int:
        with self._lock:
            row = self._connection.execute(
                "SELECT max_promotion_sequence FROM snapshots WHERE snapshot_id=?",
                (snapshot_id,),
            ).fetchone()
        if row is None:
            raise ValueError(f"unknown snapshot {snapshot_id!r}")
        return int(row["max_promotion_sequence"])

    def observations_for_snapshot(
        self,
        snapshot_id: str,
        workspace_scope: str | None = None,
    ) -> list[dict[str, Any]]:
        """Return snapshot envelopes, optionally restricted to one workspace.

        Membership is "promoted at or before this snapshot's watermark", so an
        observation promoted afterwards is excluded even though its ingestion
        sequence may be lower.
        """

        watermark = self.require_snapshot(snapshot_id)
        scope_clause = "" if workspace_scope is None else "AND workspace_scope=?"
        parameters: tuple[Any, ...] = (
            (watermark,) if workspace_scope is None else (watermark, workspace_scope)
        )
        with self._lock:
            rows = self._connection.execute(
                f"""
                SELECT envelope_json
                FROM observations
                WHERE promotion_sequence IS NOT NULL
                  AND promotion_sequence <= ?
                  AND ingest_eligible=1 {scope_clause}
                ORDER BY ingestion_sequence
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

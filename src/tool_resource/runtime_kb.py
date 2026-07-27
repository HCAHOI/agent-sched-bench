"""Causal clause latency and resource-class knowledge base.

Public and repo layers intentionally use different key granularity because
they encode different environment assumptions:

- **Public layer** (frozen after fitting): heterogeneous repositories, so it
  holds only coarse per-binary and global cold-start knowledge.
- **Repo layer** (accumulated causally online): same workspace, recurring
  command templates, so it may key by exact clause and ordered argument
  prefixes before backing off to the local binary.

Backoff order for a query in repo R (hard repo-first, deepest non-empty node
wins):

1. repo exact clause
2. repo ordered argument prefix, deepest to shallowest
3. repo binary
4. public binary
5. public global

Only observations completed strictly before a query become visible. Compound
command bucket IDs remain uncomposed because sequential and pipeline clauses
have different physical timing semantics.
"""

from __future__ import annotations

import heapq
import math
import re
from bisect import bisect_left
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass
from decimal import Decimal, InvalidOperation
from typing import Any

from tool_resource.clause_parser import parse_command_clauses

NodeKey = tuple[str, str]


def _nodes_to_json(
    nodes: Mapping[NodeKey, Sequence[float]],
) -> list[list[Any]]:
    return [[kind, key, list(values)] for (kind, key), values in nodes.items()]


def _nodes_from_json(
    rows: Iterable[Sequence[Any]],
) -> Iterator[tuple[NodeKey, list[float]]]:
    for kind, key, values in rows:
        yield (str(kind), str(key)), [float(value) for value in values]


# ==========================================================================
# Clause latency bucket predictor
# ==========================================================================

_CLAUSE_SCHEMA = "runtime_clause_resource_kb_v6"
_CLAUSE_MAX_DEPTH = 4  # frozen ordered argv-prefix depth budget
_DELIM = "\x00"  # argv tokens may contain spaces; NUL cannot collide
GENERIC_ARGV_CANONICALIZER_VERSION = "generic-argv-v2-shape"
_ENV_ASSIGNMENT = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)=(.*)", re.DOTALL)
_NUMBER = re.compile(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?")
_UUID = re.compile(
    r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}",
    re.IGNORECASE,
)
_HEX_ID = re.compile(r"(?:0x)?[0-9a-f]{8,}", re.IGNORECASE)
_URL = re.compile(r"[A-Za-z][A-Za-z0-9+.-]*://.+")
_LONG_OPTION = re.compile(r"--[A-Za-z][A-Za-z0-9_-]*")
_SHORT_OPTION = re.compile(r"-[A-Za-z]")
_SHORT_ATTACHED_VALUE = re.compile(r"(-[A-Za-z])(.+)", re.DOTALL)

# Aggregated clause-observation value sources. Each is a per-clause
# MEASURED metric (see ``tool_resource.clause_bridge``), not an eBPF exit field:
#   latency_ms          -- clause wall interval;
#   peak_cpu_cores       -- windowed peak CPU cores (never cpu_ns/wall_ns);
#   sampled_peak_rss_mb  -- max aligned distinct-mm RSS (never lifetime hiwater).
#   disk_read_write_bytes_total -- task-I/O-accounting read + write bytes.
_LATENCY_MS = "latency_ms"
_PEAK_CPU_CORES = "peak_cpu_cores"
_SAMPLED_PEAK_RSS_MB = "sampled_peak_rss_mb"
_DISK_READ_WRITE_BYTES_TOTAL = "disk_read_write_bytes_total"
_CLAUSE_SOURCES = (
    _LATENCY_MS,
    _PEAK_CPU_CORES,
    _SAMPLED_PEAK_RSS_MB,
    _DISK_READ_WRITE_BYTES_TOTAL,
)

SHORT_NULL_LIGHT_MAX_LATENCY_MS = 500.0
CANONICAL_RESOURCE_HEAVY_THRESHOLDS = {
    _PEAK_CPU_CORES: 2.0,
    _SAMPLED_PEAK_RSS_MB: 500.0,
    _DISK_READ_WRITE_BYTES_TOTAL: float(100 * 1024 * 1024),
}


@dataclass(frozen=True)
class ClauseObservation:
    """One completed *static mvdan clause*, aggregated from clause telemetry.

    Identity is the mvdan clause (``bin``, ordered ``argv``) — NOT a runtime
    exec-image occurrence. A single static clause may own an exec chain
    (``env -> nice -> workload``) and descendants; the bridge
    (``tool_resource.clause_bridge``) aggregates all owned exec images into one
    observation. The four fields are per-clause MEASURED metrics, each
    ``None`` when its target-specific coverage was insufficient:

    - ``latency_ms``      -- clause wall interval;
    - ``peak_cpu_cores``  -- windowed peak CPU cores over the owned lineage;
    - ``sampled_peak_rss_mb`` -- max aligned distinct-mm RSS over the lineage.
    - ``disk_read_write_bytes_total`` -- task-I/O read + write byte deltas.

    ``cpu_ns_cumulative`` is preserved as a separate raw field.
    ``ts_start``/``ts_end`` are wall-clock seconds for the causal
    contract.
    """

    repo: str
    bin: str
    argv: tuple[str, ...]
    ts_start: float
    ts_end: float
    latency_ms: float | None = None
    peak_cpu_cores: float | None = None
    sampled_peak_rss_mb: float | None = None
    disk_read_write_bytes_total: float | None = None
    impute_short_null_resources_as_light: bool = False
    cpu_ns_cumulative: int | None = None  # raw, separate; never a flag source
    in_loop: bool = False
    in_pipe: bool = False
    in_subst: bool = False
    pipeline_position: int = -1

    def __post_init__(self) -> None:
        if not self.argv:
            raise ValueError("clause argv must be non-empty")
        if not (math.isfinite(self.ts_start) and math.isfinite(self.ts_end)):
            raise ValueError("ts_start and ts_end must be finite")
        if self.ts_end < self.ts_start:
            raise ValueError(f"ts_end {self.ts_end} precedes ts_start {self.ts_start}")


@dataclass(frozen=True)
class LatencyBuckets:
    """Positive boundaries for ``T > boundary`` latency decisions."""

    edges_ms: tuple[float, ...]

    def __post_init__(self) -> None:
        if not self.edges_ms:
            raise ValueError("at least one latency bucket edge is required")
        previous = 0.0
        for edge in self.edges_ms:
            if (
                not isinstance(edge, (int, float))
                or isinstance(edge, bool)
                or not math.isfinite(edge)
                or edge <= previous
            ):
                raise ValueError(
                    "latency bucket edges must be finite, positive, and "
                    "strictly increasing"
                )
            previous = edge

    @property
    def bucket_count(self) -> int:
        return len(self.edges_ms) + 1

    def bucket_id(self, latency_ms: float) -> int:
        """Return i for [0, b_0], then (b_{i-1}, b_i], and the final tail."""

        if not math.isfinite(latency_ms) or latency_ms < 0.0:
            raise ValueError("latency_ms must be finite and non-negative")
        return bisect_left(self.edges_ms, latency_ms)


CANONICAL_LATENCY_BUCKET_EDGES_MS = (
    500.0,
    1000.0,
    2000.0,
    4000.0,
    8000.0,
    16000.0,
    32000.0,
    64000.0,
)
CANONICAL_LATENCY_BUCKETS = LatencyBuckets(CANONICAL_LATENCY_BUCKET_EDGES_MS)


@dataclass(frozen=True)
class ClauseLatencyBucketPrediction:
    """Empirical latency-bucket prediction for one clause."""

    probability_by_bucket: tuple[float, ...]
    scope: str
    key_kind: str
    evidence_count: int
    fallback_path: tuple[str, ...]


@dataclass(frozen=True)
class ClauseHeavyLightPrediction:
    """Empirical Heavy/Light prediction for one clause resource."""

    resource: str
    threshold: float
    probability_heavy: float
    label: str
    scope: str
    key_kind: str
    evidence_count: int
    fallback_path: tuple[str, ...]


@dataclass(frozen=True)
class CommandLatencyBucketPrediction:
    """Command result; compound commands remain explicitly uncomposed."""

    repo: str
    command: str
    parse_failed: bool
    clause_bins: tuple[str, ...]
    prediction: ClauseLatencyBucketPrediction | None
    unavailable_reason: str | None = None


def _clause_value(obs: ClauseObservation, source: str) -> float | None:
    if source == _LATENCY_MS:
        return obs.latency_ms
    if source == _PEAK_CPU_CORES:
        value = obs.peak_cpu_cores
    elif source == _SAMPLED_PEAK_RSS_MB:
        value = obs.sampled_peak_rss_mb
    elif source == _DISK_READ_WRITE_BYTES_TOTAL:
        value = obs.disk_read_write_bytes_total
    else:
        raise ValueError(f"unknown clause value source {source!r}")
    if (
        value is None
        and obs.impute_short_null_resources_as_light
        and obs.latency_ms is not None
        and obs.latency_ms < SHORT_NULL_LIGHT_MAX_LATENCY_MS
    ):
        return 0.0
    return value


def _clause_tokens(bin_: str, argv: Sequence[str]) -> tuple[str, ...]:
    # Identity token stream: bin head then the argv tail (argv[0] may be a full
    # path; bin is its basename, already normalized by mvdan).
    return (bin_, *argv[1:])


def _canonical_dynamic_value(value: str) -> str:
    if _URL.fullmatch(value):
        return "<URL>"
    if _UUID.fullmatch(value):
        return "<ID>"
    if _NUMBER.fullmatch(value):
        try:
            number = Decimal(value)
        except InvalidOperation:
            return "<NUM:EXTREME>"
        if not number:
            return "<NUM:0>"
        exponent = max(-9, min(9, number.copy_abs().adjusted()))
        sign = "-" if number.is_signed() else "+"
        return f"<NUM:{sign}E{exponent}>"
    if _HEX_ID.fullmatch(value) and (
        value.lower().startswith("0x")
        or any(character in "abcdefABCDEF" for character in value)
    ):
        return "<ID>"
    if "/" in value or "\\" in value or value.startswith(("~", ".")):
        return "<PATH>"
    return "<ARG>"


def _generic_argv_tokens(bin_: str, argv: Sequence[str]) -> tuple[str, ...]:
    tokens = [bin_]
    operands_only = False
    for token in argv[1:]:
        if token == "--":
            tokens.append(token)
            operands_only = True
            continue
        assignment = _ENV_ASSIGNMENT.fullmatch(token)
        if assignment:
            tokens.append(
                f"{assignment.group(1)}={_canonical_dynamic_value(assignment.group(2))}"
            )
            continue
        if operands_only:
            tokens.append(_canonical_dynamic_value(token))
            continue
        if token.startswith("--") and "=" in token:
            flag, value = token.split("=", 1)
            name = flag if _LONG_OPTION.fullmatch(flag) else "<OPT>"
            tokens.append(f"{name}={_canonical_dynamic_value(value)}")
            continue
        if _NUMBER.fullmatch(token):
            tokens.append(_canonical_dynamic_value(token))
            continue
        if _LONG_OPTION.fullmatch(token) or _SHORT_OPTION.fullmatch(token):
            tokens.append(token)
            continue
        attached = _SHORT_ATTACHED_VALUE.fullmatch(token)
        if attached:
            tokens.append(
                f"{attached.group(1)}={_canonical_dynamic_value(attached.group(2))}"
            )
            continue
        tokens.append(
            "<OPT>" if token.startswith("-") else _canonical_dynamic_value(token)
        )
    return tuple(tokens)


def generic_argv_keys(bin_: str, argv: Sequence[str]) -> list[NodeKey]:
    """Development-only generic canonical exact/prefix clause keys."""

    tokens = _generic_argv_tokens(bin_, argv)
    keys: list[NodeKey] = [("exact_clause", _DELIM.join(tokens))]
    depth = min(len(tokens), _CLAUSE_MAX_DEPTH)
    for length in range(depth, 1, -1):
        keys.append((f"argv_prefix_depth_{length}", _DELIM.join(tokens[:length])))
    keys.append(("bin", bin_))
    return keys


def _clause_repo_keys(bin_: str, argv: Sequence[str]) -> list[NodeKey]:
    """Repo backoff keys, most-specific first, for clause identity (bin, argv).

    Order: exact clause -> shorter bin-qualified argv prefixes -> bin. Every
    prefix key is nested under ``bin`` (its first token is ``bin``), so ``bin``
    is the LAST, most-general node queried — a more-specific prefix always wins
    before the bare bin node.
    """

    tokens = _clause_tokens(bin_, argv)
    keys: list[NodeKey] = [("exact_clause", _DELIM.join(tokens))]
    depth = min(len(tokens), _CLAUSE_MAX_DEPTH)
    # depth-1 prefix equals the bin node's content, so stop prefixes at 2.
    for length in range(depth, 1, -1):
        keys.append((f"argv_prefix_depth_{length}", _DELIM.join(tokens[:length])))
    keys.append(("bin", bin_))
    return keys


def _clause_public_keys(bin_: str) -> list[NodeKey]:
    """Public clause keys: coarse bin prior then global."""

    return [("bin", bin_), ("global", "")]


class ClauseResourceKB:
    """Causal clause history with latency and resource-class APIs.

    Public bin priors are frozen after construction; repo clause/prefix nodes
    accumulate causally under a monotonic-query guard.
    """

    def __init__(self) -> None:
        self._public: dict[str, dict[NodeKey, tuple[float, ...]]] = {
            source: {} for source in _CLAUSE_SOURCES
        }
        self._repo: dict[str, dict[str, dict[NodeKey, list[float]]]] = {}
        self._pending: list[tuple[float, int, ClauseObservation]] = []
        self._pending_seq = 0
        self._last_query_ts: float | None = None

    @classmethod
    def fit_public(
        cls,
        observations: Iterable[ClauseObservation],
    ) -> ClauseResourceKB:
        """Fit frozen public bin/global priors from historical clauses."""

        acc: dict[str, dict[NodeKey, list[float]]] = {
            source: {} for source in _CLAUSE_SOURCES
        }
        for obs in observations:
            keys = _clause_public_keys(obs.bin)
            for source in _CLAUSE_SOURCES:
                value = _clause_value(obs, source)
                if value is None:
                    continue
                for key in keys:
                    acc[source].setdefault(key, []).append(value)
        if not acc[_LATENCY_MS].get(("global", "")):
            raise ValueError("fit corpus has no clause latency evidence")
        kb = cls()
        kb._public = {
            source: {key: tuple(values) for key, values in nodes.items()}
            for source, nodes in acc.items()
        }
        return kb

    def observe_completed_clause(self, obs: ClauseObservation) -> None:
        """Buffer a completed clause; visible only once strictly causally prior."""

        heapq.heappush(self._pending, (obs.ts_end, self._pending_seq, obs))
        self._pending_seq += 1

    def _absorb_completed(self, ts_start: float) -> None:
        while self._pending and self._pending[0][0] < ts_start:
            _, _, obs = heapq.heappop(self._pending)
            repo_sources = self._repo.setdefault(
                obs.repo, {source: {} for source in _CLAUSE_SOURCES}
            )
            keys = _clause_repo_keys(obs.bin, obs.argv)
            for source in _CLAUSE_SOURCES:
                value = _clause_value(obs, source)
                if value is None:
                    continue
                for key in keys:
                    repo_sources[source].setdefault(key, []).append(value)

    def _select(
        self, repo: str, source: str, bin_: str, argv: Sequence[str]
    ) -> tuple[Sequence[float], str, str, tuple[str, ...]] | None:
        repo_nodes = self._repo.get(repo, {}).get(source, {})
        public_nodes = self._public[source]
        path: list[str] = []
        for key in _clause_repo_keys(bin_, argv):
            path.append(f"repo:{key[0]}")
            values = repo_nodes.get(key)
            if values:
                return values, "repo", key[0], tuple(path)
        for key in _clause_public_keys(bin_):
            path.append(f"public:{key[0]}")
            values = public_nodes.get(key)
            if values:
                return values, "public", key[0], tuple(path)
        return None

    def predict_clause_latency_bucket(
        self,
        repo: str,
        bin_: str,
        argv: Sequence[str],
        buckets: LatencyBuckets,
        *,
        ts_start: float | None = None,
    ) -> ClauseLatencyBucketPrediction:
        """Predict the empirical latency-bucket PMF for one clause."""

        if ts_start is not None:
            self._advance(ts_start)
        selected = self._select(repo, _LATENCY_MS, bin_, argv)
        if selected is None:
            raise ValueError("no public global clause latency node")
        values, scope, kind, path = selected
        counts = [0] * buckets.bucket_count
        for value in values:
            counts[buckets.bucket_id(value)] += 1
        return ClauseLatencyBucketPrediction(
            probability_by_bucket=tuple(count / len(values) for count in counts),
            scope=scope,
            key_kind=kind,
            evidence_count=len(values),
            fallback_path=path,
        )

    def predict_clause_heavy_light(
        self,
        repo: str,
        bin_: str,
        argv: Sequence[str],
        resource: str,
        *,
        ts_start: float | None = None,
    ) -> ClauseHeavyLightPrediction | None:
        """Predict one resource using the same public/local backoff hierarchy."""

        try:
            threshold = CANONICAL_RESOURCE_HEAVY_THRESHOLDS[resource]
        except KeyError as exc:
            raise ValueError(f"unknown Heavy/Light resource {resource!r}") from exc
        if ts_start is not None:
            self._advance(ts_start)
        selected = self._select(repo, resource, bin_, argv)
        if selected is None:
            return None
        values, scope, kind, path = selected
        probability_heavy = sum(value > threshold for value in values) / len(values)
        return ClauseHeavyLightPrediction(
            resource=resource,
            threshold=threshold,
            probability_heavy=probability_heavy,
            label="heavy" if probability_heavy > 0.5 else "light",
            scope=scope,
            key_kind=kind,
            evidence_count=len(values),
            fallback_path=path,
        )

    def predict_clause_resource_classes(
        self,
        repo: str,
        bin_: str,
        argv: Sequence[str],
        *,
        ts_start: float | None = None,
    ) -> dict[str, ClauseHeavyLightPrediction | None]:
        if ts_start is not None:
            self._advance(ts_start)
        return {
            resource: self.predict_clause_heavy_light(repo, bin_, argv, resource)
            for resource in CANONICAL_RESOURCE_HEAVY_THRESHOLDS
        }

    def predict_command_latency_bucket_from_clauses(
        self,
        repo: str,
        clauses: Sequence[Mapping[str, Any]],
        ts_start: float,
        buckets: LatencyBuckets,
        *,
        command: str = "",
        parse_failed: bool = False,
    ) -> CommandLatencyBucketPrediction:
        """Predict only a parsed single clause; never compose bucket IDs."""

        self._advance(ts_start)
        effective = list(clauses)
        clause_bins = tuple(str(c["bin"]) for c in effective)
        reason = None
        prediction = None
        if parse_failed:
            reason = "parse_failed"
        elif len(effective) != 1:
            reason = "compound_command_uncomposed"
        else:
            clause = effective[0]
            prediction = self.predict_clause_latency_bucket(
                repo,
                str(clause["bin"]),
                tuple(clause["argv"]),
                buckets,
            )
        return CommandLatencyBucketPrediction(
            repo=repo,
            command=command,
            parse_failed=parse_failed,
            clause_bins=clause_bins,
            prediction=prediction,
            unavailable_reason=reason,
        )

    def predict_command_latency_bucket(
        self,
        repo: str,
        command: str,
        ts_start: float,
        buckets: LatencyBuckets,
    ) -> CommandLatencyBucketPrediction:
        """Parse a command and predict its bucket when composition is unnecessary.

        Enforces the monotonic-query guard and releases causally-prior repo
        clauses before predicting.
        """

        parsed = parse_command_clauses(command)
        return self.predict_command_latency_bucket_from_clauses(
            repo,
            parsed["clauses"],
            ts_start,
            buckets,
            command=command,
            parse_failed=bool(parsed["parse_failed"]),
        )

    def _advance(self, ts_start: float) -> None:
        if not math.isfinite(ts_start):
            raise ValueError("query ts_start must be finite")
        if self._last_query_ts is not None and ts_start < self._last_query_ts:
            raise ValueError(
                f"backdated query at ts_start {ts_start} after a query at "
                f"{self._last_query_ts}: repo clause state already absorbed "
                "observations completed before the later time"
            )
        self._last_query_ts = ts_start
        self._absorb_completed(ts_start)

    def to_json_obj(self) -> dict[str, Any]:
        """JSON-serializable snapshot of public, repo, and pending state."""

        return {
            "schema": _CLAUSE_SCHEMA,
            "max_prefix_depth": _CLAUSE_MAX_DEPTH,
            "public": {
                source: _nodes_to_json(nodes) for source, nodes in self._public.items()
            },
            "repo": {
                repo: {
                    source: _nodes_to_json(nodes) for source, nodes in sources.items()
                }
                for repo, sources in self._repo.items()
            },
            "pending": [asdict(obs) for _, _, obs in sorted(self._pending)],
            "last_query_ts": self._last_query_ts,
        }

    @classmethod
    def from_json_obj(cls, obj: Mapping[str, Any]) -> ClauseResourceKB:
        """Restore a snapshot produced by :meth:`to_json_obj`."""

        if obj.get("schema") != _CLAUSE_SCHEMA:
            if obj.get("schema") == "runtime_clause_resource_kb_v5":
                raise ValueError(
                    "runtime_clause_resource_kb_v5 lacks Disk and short-null "
                    "resource labels; refit the snapshot"
                )
            raise ValueError(f"unsupported clause schema {obj.get('schema')!r}")
        if obj.get("max_prefix_depth") != _CLAUSE_MAX_DEPTH:
            raise ValueError("snapshot prefix depth differs from module depth")
        kb = cls()
        kb._public = {
            source: {
                key: tuple(values)
                for key, values in _nodes_from_json(obj["public"].get(source, []))
            }
            for source in _CLAUSE_SOURCES
        }
        kb._repo = {
            repo: {
                source: {
                    key: list(values)
                    for key, values in _nodes_from_json(sources.get(source, []))
                }
                for source in _CLAUSE_SOURCES
            }
            for repo, sources in obj.get("repo", {}).items()
        }
        for row in obj.get("pending", []):
            kb.observe_completed_clause(
                ClauseObservation(**{**row, "argv": tuple(row["argv"])})
            )
        last_query_ts = obj.get("last_query_ts")
        kb._last_query_ts = None if last_query_ts is None else float(last_query_ts)
        return kb


__all__ = [
    "CANONICAL_LATENCY_BUCKETS",
    "CANONICAL_LATENCY_BUCKET_EDGES_MS",
    "CANONICAL_RESOURCE_HEAVY_THRESHOLDS",
    "GENERIC_ARGV_CANONICALIZER_VERSION",
    "SHORT_NULL_LIGHT_MAX_LATENCY_MS",
    "ClauseHeavyLightPrediction",
    "ClauseLatencyBucketPrediction",
    "ClauseObservation",
    "ClauseResourceKB",
    "CommandLatencyBucketPrediction",
    "LatencyBuckets",
    "generic_argv_keys",
]

"""Runtime asymmetric two-layer tool resource knowledge base.

Public and repo layers intentionally use different key granularity because
they encode different environment assumptions:

- **Public layer** (frozen after fitting): heterogeneous repositories, so it
  holds only coarse cold-start knowledge — per-binary/head, per outer tool
  name, and global nodes. It never contains exact-command or command-prefix
  nodes; parameter semantics do not transfer across environments.
- **Repo layer** (accumulated causally online): same workspace, recurring
  command templates, so it may key by exact normalized command and ordered
  command prefixes before backing off to the local binary/head.

Backoff order for a query in repo R (hard repo-first, deepest non-empty node
wins — the evaluated baseline policy):

1. repo exact normalized command
2. repo ordered command prefix, deepest to shallowest (max depth 4, the
   frozen depth budget shared with the evaluated lattice)
3. repo binary/head
4. public binary/head
5. public outer tool name (also the honest landing spot for compound
   commands with no single head, untokenizable commands, and non-shell
   tools)
6. public global

Binary/head identity is the generic basenamed command head from
``tool_time.command``. A compound shell call (``make && pytest``) has no
single honest head, so it feeds and matches no binary node — its full-call
label is never attributed to every contained binary. Shell builtins such as
``cd`` are represented by their parsed head; their label remains the
enclosing tool-call observation. No tool-specific option semantics are
implemented; argument order is preserved as-is.

Target semantics (matching the established label definitions):

- ``latency_ms``: observed call latency, skipped for censored calls;
- ``peak_cpu_cores``: eligible peak CPU cores;
- ``peak_memory_mb``: stored as an eligible residual relative to the
  deployment-legal ``ambient_before_mb`` anchor; a prediction adds the
  query's *current* ambient memory to the residual quantile. Queries
  without an ambient anchor get no memory prediction.

Causality: ``observe_completed_call`` buffers observations; an observation
enters repo state only when a later query's ``ts_start`` strictly exceeds
its ``ts_end`` (the evaluated prequential contract). Running, overlapping,
same-start, and future calls never leak into a prediction.

Known limitations (documented, unresolved by design in this phase): under
the hard repo-first baseline a repo node with a single sample overrides
public evidence; arbitration alternatives (shrinkage, calibration) remain
development candidates and are not implemented here.

--------------------------------------------------------------------------
Clause dominant-type classifier (``ClauseResourceKB``)
--------------------------------------------------------------------------

The canonical objective (`analysis/development/tool-resource-canonical-objective.md`)
wants three command-level binary labels predicted before execution and
composed from mvdan clauses:

    shell tool call
      -> mvdan clauses / binary identities (`tool_resource.features`)
      -> per-clause flags from public bin priors + causal repo refinement
      -> command flag per target = OR over its clause flags

``ClauseObservation`` is one *static mvdan clause* (identity = ``bin`` + ordered
``argv``), aggregated by ``tool_resource.clause_bridge`` from the Stage-2 eBPF
windowed sampler (`analysis/development/clause-telemetry-ebpf-stage2-*`). A
static clause may own a same-PID exec chain and descendants
(``env -> nice -> workload`` is ONE clause headed by ``env``); the bridge folds
all owned exec images into a single observation carrying three per-clause
MEASURED metrics (each ``None`` when its coverage is insufficient) plus a raw
``cpu_ns_cumulative`` that is never a flag source.

Per-clause labels at the fixed operating points:

- **latency long** (``> 3500 ms`` and ``> 5000 ms``): from the clause wall
  interval ``latency_ms``.
- **CPU heavy** (``> 2 cores``): from eligible ``peak_cpu_cores`` — the Stage-2
  windowed peak CPU rate, NEVER ``cpu_ns/wall_ns``. Unknown when the clause had
  insufficient CPU-sample coverage.
- **memory heavy** (``> 500 MB``): from eligible ``sampled_peak_rss_mb`` — the
  Stage-2 max aligned distinct-mm RSS sum, NEVER lifetime hiwater or average
  memory. Unknown when RSS coverage is insufficient.

Flag rule: the registered strict-``>`` threshold with a ``>= 0.5`` exceedance
vote over the selected node's observations. Backoff for clause identity
(``bin``, ``argv``) in repo R: repo exact clause -> repo shorter bin-qualified
argv prefixes (depth 4..2) -> repo bin -> public bin -> public global. Prefix
keys are nested under ``bin``, so ``bin`` is queried only after every
more-specific prefix. Public holds only bin and global nodes (coarse
cold-start); exact/prefix nodes are repo-only, mirroring the q90 asymmetry.

Command aggregation is the frozen per-target THREE-VALUED OR of clause flags:
any True -> True; otherwise any Unknown -> Unknown; otherwise False. It never
sums concurrent clause CPU/RSS — two ~2-core clauses each flag False at strict
``> 2`` while the command is concurrently heavy; that is the registered
composer limitation, surfaced on the prediction, not silently patched.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import heapq
import math
from typing import Any, Iterable, Iterator, Mapping, Sequence

from tool_resource.features import parse_command_clauses
from tool_resource.metrics import ecdf_quantile
from tool_time.command import shell_command_heads, shell_command_prefix_tokens

TARGETS = ("latency_ms", "peak_cpu_cores", "peak_memory_mb")
_QUANTILE = 0.9
_MAX_PREFIX_DEPTH = 4  # frozen depth budget, same as the evaluated lattice
_SCHEMA = "runtime_tool_resource_kb_v1"

# (kind, key) — kind is what provenance exposes; key stays internal.
NodeKey = tuple[str, str]


@dataclass(frozen=True)
class CompletedCall:
    """Deployment-legal record of one finished tool call."""

    repo: str
    tool_name: str
    command: str | None
    ts_start: float
    ts_end: float
    censored: bool = False
    peak_cpu_cores: float | None = None
    peak_cpu_cores_eligible: bool = False
    peak_memory_mb: float | None = None
    peak_memory_mb_eligible: bool = False
    ambient_before_mb: float | None = None

    def __post_init__(self) -> None:
        if not (math.isfinite(self.ts_start) and math.isfinite(self.ts_end)):
            raise ValueError("ts_start and ts_end must be finite")
        if self.ts_end < self.ts_start:
            raise ValueError(f"ts_end {self.ts_end} precedes ts_start {self.ts_start}")

    @classmethod
    def from_resource_sample(cls, sample: Any, repo: str) -> CompletedCall:
        """Adapt a ``ResourceCallSample``-shaped row (duck-typed)."""

        tool_args = sample.tool_args or {}
        command = tool_args.get("command")
        return cls(
            repo=repo,
            tool_name=sample.tool_name,
            command=command if isinstance(command, str) else None,
            ts_start=sample.tool_ts_start,
            ts_end=sample.tool_ts_end,
            censored=sample.censored,
            peak_cpu_cores=sample.peak_cpu_cores,
            peak_cpu_cores_eligible=sample.peak_cpu_cores_eligible,
            peak_memory_mb=sample.peak_memory_mb,
            peak_memory_mb_eligible=sample.peak_memory_mb_eligible,
            ambient_before_mb=sample.ambient_before_mb,
        )


@dataclass(frozen=True)
class ToolCallQuery:
    """Pre-call query: repo identity, outer tool call, and current context."""

    repo: str
    tool_name: str
    command: str | None
    ts_start: float
    ambient_before_mb: float | None = None


@dataclass(frozen=True)
class TargetPrediction:
    """q90 estimate plus provenance for one target."""

    target: str
    q90: float | None
    scope: str | None
    key_kind: str | None
    evidence_count: int
    fallback_path: tuple[str, ...]
    note: str | None = None


def _target_values(call: CompletedCall) -> dict[str, float]:
    """Eligible per-target values; an ineligible target is skipped alone."""

    values: dict[str, float] = {}
    if not call.censored:
        values["latency_ms"] = (call.ts_end - call.ts_start) * 1000.0
    if call.peak_cpu_cores_eligible and call.peak_cpu_cores is not None:
        values["peak_cpu_cores"] = float(call.peak_cpu_cores)
    if (
        call.peak_memory_mb_eligible
        and call.peak_memory_mb is not None
        and call.ambient_before_mb is not None
    ):
        residual = float(call.peak_memory_mb) - float(call.ambient_before_mb)
        if not math.isfinite(residual):
            raise ValueError("memory residual must be finite")
        values["peak_memory_mb"] = residual
    return values


def _single_head(command: str | None) -> str | None:
    if not isinstance(command, str) or not command.strip():
        return None
    heads = shell_command_heads(command)
    return heads[0] if len(heads) == 1 else None


def _repo_keys(command: str | None) -> list[NodeKey]:
    """Repo-layer node keys, deepest first: exact, prefixes, binary head."""

    if not isinstance(command, str) or not command.strip():
        return []
    tokens = shell_command_prefix_tokens(command)
    if not tokens:
        return []
    keys: list[NodeKey] = [("exact_command", " ".join(tokens))]
    depth = min(len(tokens), _MAX_PREFIX_DEPTH)
    for length in range(depth, 0, -1):
        keys.append((f"command_prefix_depth_{length}", " ".join(tokens[:length])))
    head = _single_head(command)
    if head is not None:
        keys.append(("binary_head", head))
    return keys


def _public_keys(tool_name: str, command: str | None) -> list[NodeKey]:
    """Public-layer node keys: binary head (if honest), tool name, global."""

    keys: list[NodeKey] = []
    head = _single_head(command)
    if head is not None:
        keys.append(("binary_head", head))
    keys.append(("tool_name", tool_name))
    keys.append(("global", ""))
    return keys


class RuntimeToolResourceKB:
    """Frozen public layer plus causally accumulated per-repo nodes.

    Construct via :meth:`fit_public` or :meth:`from_json_obj`; the public
    layer is immutable afterwards and online observations touch only repo
    and pending state.
    """

    def __init__(self) -> None:
        self._public: dict[str, dict[NodeKey, tuple[float, ...]]] = {
            target: {} for target in TARGETS
        }
        self._repo: dict[str, dict[str, dict[NodeKey, list[float]]]] = {}
        self._pending: list[tuple[float, int, CompletedCall]] = []
        self._pending_seq = 0
        # Queries must be monotonic: absorbing pending calls is irreversible,
        # so a backdated query would see repo state from its future.
        self._last_query_ts: float | None = None

    @classmethod
    def fit_public(cls, calls: Iterable[CompletedCall]) -> RuntimeToolResourceKB:
        """Fit the frozen public layer from historical completed calls."""

        accumulator: dict[str, dict[NodeKey, list[float]]] = {
            target: {} for target in TARGETS
        }
        for call in calls:
            keys = _public_keys(call.tool_name, call.command)
            for target, value in _target_values(call).items():
                for key in keys:
                    accumulator[target].setdefault(key, []).append(value)
        missing = [
            target for target in TARGETS if not accumulator[target].get(("global", ""))
        ]
        if missing:
            raise ValueError(f"fit corpus has no eligible labels for {missing}")
        kb = cls()
        kb._public = {
            target: {key: tuple(values) for key, values in nodes.items()}
            for target, nodes in accumulator.items()
        }
        return kb

    def observe_completed_call(self, call: CompletedCall) -> None:
        """Buffer a finished call; it becomes visible only once causally prior."""

        heapq.heappush(self._pending, (call.ts_end, self._pending_seq, call))
        self._pending_seq += 1

    def query(self, query: ToolCallQuery) -> dict[str, TargetPrediction]:
        """Predict q90 for all targets before the call at ``query.ts_start``."""

        if self._last_query_ts is not None and query.ts_start < self._last_query_ts:
            raise ValueError(
                f"backdated query at ts_start {query.ts_start} after a query at "
                f"{self._last_query_ts}: repo state already absorbed observations "
                "completed before the later time"
            )
        self._last_query_ts = query.ts_start
        self._absorb_completed(query.ts_start)
        return {
            target: self._predict_target(query, target) for target in TARGETS
        }

    def _absorb_completed(self, ts_start: float) -> None:
        # Strictly-completed contract: ts_end < ts_start. Same-start,
        # overlapping, running, and future observations stay pending.
        while self._pending and self._pending[0][0] < ts_start:
            _, _, call = heapq.heappop(self._pending)
            repo_targets = self._repo.setdefault(
                call.repo, {target: {} for target in TARGETS}
            )
            keys = _repo_keys(call.command)
            for target, value in _target_values(call).items():
                for key in keys:
                    repo_targets[target].setdefault(key, []).append(value)

    def _levels(
        self, repo: str, target: str, tool_name: str, command: str | None
    ) -> Iterator[tuple[str, NodeKey, Sequence[float]]]:
        repo_nodes = self._repo.get(repo, {}).get(target, {})
        for key in _repo_keys(command):
            yield "repo", key, repo_nodes.get(key, ())
        public_nodes = self._public[target]
        for key in _public_keys(tool_name, command):
            yield "public", key, public_nodes.get(key, ())

    def _select(
        self, repo: str, target: str, tool_name: str, command: str | None
    ) -> tuple[Sequence[float], str, str, tuple[str, ...]]:
        """Baseline arbitration: first (deepest) non-empty node wins outright.

        This is the single selection point; alternative arbitration policies
        (shrinkage, calibration) would replace this method, not the storage.
        """

        path: list[str] = []
        for scope, (kind, _), values in self._levels(repo, target, tool_name, command):
            path.append(f"{scope}:{kind}")
            if values:
                return values, scope, kind, tuple(path)
        raise ValueError(f"no public global node for target {target!r}")

    def _predict_target(
        self, query: ToolCallQuery, target: str
    ) -> TargetPrediction:
        if target == "peak_memory_mb" and query.ambient_before_mb is None:
            return TargetPrediction(
                target=target,
                q90=None,
                scope=None,
                key_kind=None,
                evidence_count=0,
                fallback_path=(),
                note="memory prediction requires ambient_before_mb anchor",
            )
        values, scope, kind, path = self._select(
            query.repo, target, query.tool_name, query.command
        )
        q90 = ecdf_quantile(values, _QUANTILE)
        note = None
        if target == "peak_memory_mb":
            q90 += float(query.ambient_before_mb)
            note = "residual quantile plus query ambient_before_mb"
        return TargetPrediction(
            target=target,
            q90=q90,
            scope=scope,
            key_kind=kind,
            evidence_count=len(values),
            fallback_path=path,
            note=note,
        )

    def to_json_obj(self) -> dict[str, Any]:
        """JSON-serializable snapshot of public, repo, and pending state."""

        return {
            "schema": _SCHEMA,
            "quantile": _QUANTILE,
            "max_prefix_depth": _MAX_PREFIX_DEPTH,
            "public": {
                target: _nodes_to_json(nodes)
                for target, nodes in self._public.items()
            },
            "repo": {
                repo: {
                    target: _nodes_to_json(nodes)
                    for target, nodes in targets.items()
                }
                for repo, targets in self._repo.items()
            },
            "pending": [
                asdict(call) for _, _, call in sorted(self._pending)
            ],
            "last_query_ts": self._last_query_ts,
        }

    @classmethod
    def from_json_obj(cls, obj: Mapping[str, Any]) -> RuntimeToolResourceKB:
        """Restore a snapshot produced by :meth:`to_json_obj`."""

        if obj.get("schema") != _SCHEMA:
            raise ValueError(f"unsupported schema {obj.get('schema')!r}")
        if obj.get("quantile") != _QUANTILE:
            raise ValueError("snapshot quantile differs from module quantile")
        if obj.get("max_prefix_depth") != _MAX_PREFIX_DEPTH:
            raise ValueError("snapshot prefix depth differs from module depth")
        kb = cls()
        kb._public = {
            target: {
                key: tuple(values)
                for key, values in _nodes_from_json(obj["public"][target])
            }
            for target in TARGETS
        }
        missing = [
            target for target in TARGETS if not kb._public[target].get(("global", ""))
        ]
        if missing:
            raise ValueError(f"snapshot has no public global node for {missing}")
        kb._repo = {
            repo: {
                target: {
                    key: list(values)
                    for key, values in _nodes_from_json(targets.get(target, []))
                }
                for target in TARGETS
            }
            for repo, targets in obj.get("repo", {}).items()
        }
        for row in obj.get("pending", []):
            kb.observe_completed_call(CompletedCall(**row))
        last_query_ts = obj.get("last_query_ts")
        kb._last_query_ts = None if last_query_ts is None else float(last_query_ts)
        return kb


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
# Clause dominant-type classifier
# ==========================================================================

_CLAUSE_SCHEMA = "runtime_clause_resource_kb_v2"
_CLAUSE_MAX_DEPTH = 4  # ordered argv-prefix budget, same frozen depth as q90
_DELIM = "\x00"  # argv tokens may contain spaces; NUL cannot collide

# Aggregated Stage-2 clause-observation value sources. Each is a per-clause
# MEASURED metric (see ``tool_resource.clause_bridge``), not an eBPF exit field:
#   latency_ms          -- clause wall interval (frozen 3500/5000 ms points);
#   peak_cpu_cores       -- windowed peak CPU cores (never cpu_ns/wall_ns);
#   sampled_peak_rss_mb  -- max aligned distinct-mm RSS (never lifetime hiwater).
_LATENCY_MS = "latency_ms"
_PEAK_CPU_CORES = "peak_cpu_cores"
_SAMPLED_PEAK_RSS_MB = "sampled_peak_rss_mb"
_CLAUSE_SOURCES = (_LATENCY_MS, _PEAK_CPU_CORES, _SAMPLED_PEAK_RSS_MB)

# Fixed operating points (canonical objective). Strict `>` thresholds.
CPU_HEAVY_TARGET = "cpu_heavy_2cores"
MEMORY_HEAVY_TARGET = "memory_heavy_500mb"
FLAG_TARGETS: dict[str, tuple[str, float]] = {
    "latency_long_3500ms": (_LATENCY_MS, 3500.0),
    "latency_long_5000ms": (_LATENCY_MS, 5000.0),
    CPU_HEAVY_TARGET: (_PEAK_CPU_CORES, 2.0),
    MEMORY_HEAVY_TARGET: (_SAMPLED_PEAK_RSS_MB, 500.0),
}
CLASSIFIER_TARGETS = tuple(FLAG_TARGETS)

# Command composition is a per-target THREE-VALUED OR over clause flags:
#   any True -> True; else any Unknown(None) -> Unknown(None); else False.
# It never sums concurrent clause CPU/RSS — that limitation is surfaced, not
# silently corrected (see _aggregate_or).
_CONCURRENT_SUM_CAVEAT = (
    "three-valued OR does not sum concurrent clauses: individually "
    "sub-threshold concurrent clauses can jointly cross the threshold "
    "(registered composer limitation, not a telemetry failure)"
)
_SEQUENTIAL_LATENCY_CAVEAT = (
    "OR is sufficient but not complete: sequential short clauses can "
    "accumulate into a long command"
)


@dataclass(frozen=True)
class ClauseObservation:
    """One completed *static mvdan clause*, aggregated from Stage-2 telemetry.

    Identity is the mvdan clause (``bin``, ordered ``argv``) — NOT a runtime
    exec-image occurrence. A single static clause may own an exec chain
    (``env -> nice -> workload``) and descendants; the bridge
    (``tool_resource.clause_bridge``) aggregates all owned exec images into one
    observation. The three flag sources are per-clause MEASURED metrics, each
    ``None`` when its target-specific coverage was insufficient (unknown):

    - ``latency_ms``      -- clause wall interval;
    - ``peak_cpu_cores``  -- windowed peak CPU cores over the owned lineage;
    - ``sampled_peak_rss_mb`` -- max aligned distinct-mm RSS over the lineage.

    ``cpu_ns_cumulative`` is preserved as a separate raw field and is NEVER a
    flag source. ``ts_start``/``ts_end`` are wall-clock seconds for the causal
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
class ClauseFlagPrediction:
    """One target's flag for one clause, with provenance."""

    target: str
    flag: bool | None
    threshold: float | None
    source: str | None
    scope: str | None
    key_kind: str | None
    evidence_count: int
    fallback_path: tuple[str, ...]
    note: str | None = None


@dataclass(frozen=True)
class CommandFlagPrediction:
    """Command-level OR of clause flags for one target."""

    target: str
    flag: bool | None
    clause_flags: tuple[ClauseFlagPrediction, ...]
    note: str | None = None


@dataclass(frozen=True)
class CommandPrediction:
    """Per-target command predictions for a parsed shell command."""

    repo: str
    command: str
    parse_failed: bool
    clause_bins: tuple[str, ...]
    targets: dict[str, CommandFlagPrediction] = field(default_factory=dict)


def _clause_value(obs: ClauseObservation, source: str) -> float | None:
    if source == _LATENCY_MS:
        return obs.latency_ms
    if source == _PEAK_CPU_CORES:
        return obs.peak_cpu_cores
    if source == _SAMPLED_PEAK_RSS_MB:
        return obs.sampled_peak_rss_mb
    raise ValueError(f"unknown clause value source {source!r}")


def _clause_tokens(bin_: str, argv: Sequence[str]) -> tuple[str, ...]:
    # Identity token stream: bin head then the argv tail (argv[0] may be a full
    # path; bin is its basename, already normalized by mvdan).
    return (bin_, *argv[1:])


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


def _exceedance_ge_half(values: Sequence[float], threshold: float) -> bool:
    """Registered rule: strict `>` threshold, `>= 0.5` exceedance vote."""

    if not values:
        raise ValueError("exceedance requires non-empty values")
    return sum(value > threshold for value in values) / len(values) >= 0.5


class ClauseResourceKB:
    """Clause-level dominant-type classifier with the asymmetric KB structure.

    Public bin priors are frozen after construction; repo clause/prefix nodes
    accumulate causally (strict ``ts_end < query ts_start``) under the same
    monotonic-query guard as :class:`RuntimeToolResourceKB`.
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
    def fit_public(cls, observations: Iterable[ClauseObservation]) -> ClauseResourceKB:
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
            raise ValueError("fit corpus has no clause latency (wall_ns) evidence")
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

    def predict_clause(
        self, repo: str, bin_: str, argv: Sequence[str]
    ) -> dict[str, ClauseFlagPrediction]:
        """Per-target flags for one clause; a target with no evidence is unknown.

        CPU-heavy uses eligible ``peak_cpu_cores > 2`` and memory-heavy uses
        eligible ``sampled_peak_rss_mb > 500`` — both target-specific, so a
        clause with no such evidence yields ``flag=None`` (unknown) for that
        target only, never a fabricated value.
        """

        out: dict[str, ClauseFlagPrediction] = {}
        for target, (source, threshold) in FLAG_TARGETS.items():
            selected = self._select(repo, source, bin_, argv)
            if selected is None:
                out[target] = ClauseFlagPrediction(
                    target=target,
                    flag=None,
                    threshold=threshold,
                    source=source,
                    scope=None,
                    key_kind=None,
                    evidence_count=0,
                    fallback_path=(),
                    note=f"no {source} evidence in public or repo layer (unknown)",
                )
                continue
            values, scope, kind, fpath = selected
            out[target] = ClauseFlagPrediction(
                target=target,
                flag=_exceedance_ge_half(values, threshold),
                threshold=threshold,
                source=source,
                scope=scope,
                key_kind=kind,
                evidence_count=len(values),
                fallback_path=fpath,
            )
        return out

    def predict_command_from_clauses(
        self,
        repo: str,
        clauses: Sequence[Mapping[str, Any]],
        ts_start: float,
        *,
        command: str = "",
        parse_failed: bool = False,
    ) -> CommandPrediction:
        """Advance causal state, then OR clause flags into command-level flags."""

        self._advance(ts_start)
        per_clause = [
            self.predict_clause(repo, str(c["bin"]), tuple(c["argv"])) for c in clauses
        ]
        targets: dict[str, CommandFlagPrediction] = {}
        for target in CLASSIFIER_TARGETS:
            clause_flags = tuple(pc[target] for pc in per_clause)
            targets[target] = _aggregate_or(target, clause_flags)
        return CommandPrediction(
            repo=repo,
            command=command,
            parse_failed=parse_failed,
            clause_bins=tuple(str(c["bin"]) for c in clauses),
            targets=targets,
        )

    def predict_command(
        self, repo: str, command: str, ts_start: float
    ) -> CommandPrediction:
        """Parse ``command`` with mvdan then predict command-level flags.

        Enforces the monotonic-query guard and releases causally-prior repo
        clauses before predicting, exactly like the legacy call-level query.
        """

        parsed = parse_command_clauses(command)
        return self.predict_command_from_clauses(
            repo,
            parsed["clauses"],
            ts_start,
            command=command,
            parse_failed=bool(parsed["parse_failed"]),
        )

    def _advance(self, ts_start: float) -> None:
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
                source: _nodes_to_json(nodes)
                for source, nodes in self._public.items()
            },
            "repo": {
                repo: {
                    source: _nodes_to_json(nodes)
                    for source, nodes in sources.items()
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
        if not kb._public[_LATENCY_MS].get(("global", "")):
            raise ValueError("snapshot has no public clause latency global node")
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
            kb.observe_completed_clause(ClauseObservation(**{
                **row, "argv": tuple(row["argv"])
            }))
        last_query_ts = obj.get("last_query_ts")
        kb._last_query_ts = None if last_query_ts is None else float(last_query_ts)
        return kb


def _aggregate_or(
    target: str, clause_flags: Sequence[ClauseFlagPrediction]
) -> CommandFlagPrediction:
    """Frozen per-target three-valued OR over clause flags.

    any True -> True; otherwise any Unknown(None) -> Unknown(None); otherwise
    False (an empty clause set is False, the OR identity). Never sums
    concurrent clause CPU/RSS — that limitation is surfaced in the note.
    """

    flags = [cf.flag for cf in clause_flags]
    if any(f is True for f in flags):
        result: bool | None = True
    elif any(f is None for f in flags):
        result = None
    else:
        result = False

    if not clause_flags:
        note = "no executable clauses parsed; three-valued OR over empty set is False"
    elif target.startswith("latency_long"):
        note = _SEQUENTIAL_LATENCY_CAVEAT
    else:  # cpu-heavy / memory-heavy
        note = _CONCURRENT_SUM_CAVEAT
    return CommandFlagPrediction(
        target=target, flag=result, clause_flags=tuple(clause_flags), note=note
    )


__all__ = [
    "CLASSIFIER_TARGETS",
    "CPU_HEAVY_TARGET",
    "FLAG_TARGETS",
    "MEMORY_HEAVY_TARGET",
    "TARGETS",
    "ClauseFlagPrediction",
    "ClauseObservation",
    "ClauseResourceKB",
    "CommandFlagPrediction",
    "CommandPrediction",
    "CompletedCall",
    "RuntimeToolResourceKB",
    "TargetPrediction",
    "ToolCallQuery",
]

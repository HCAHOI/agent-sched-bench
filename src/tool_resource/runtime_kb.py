"""Causal clause latency and resource-class knowledge base.

Public and repo layers intentionally use different key granularity because
they encode different environment assumptions. The default representation
preserves the reviewed raw-prefix hierarchy; the development-only structured
representation replaces raw prefixes with one role-aware argv signature:

- **Public layer** (frozen after fitting): heterogeneous repositories, so the
  default holds coarse per-binary/global knowledge while the structured arm
  may additionally hold its privacy-preserving argv signature.
- **Repo layer** (accumulated causally online): same workspace, recurring
  command templates, so it may key by exact clause and ordered argument
  prefixes before backing off to the local binary.

Default backoff order for a query in repo R (hard repo-first, deepest non-empty
node wins):

1. repo exact clause
2. repo ordered argument prefix, deepest to shallowest
3. repo binary
4. public binary
5. public global

An opt-in ``repo_binary_first`` arbitration reorders step 3 ahead of steps 1-2,
selecting the repository's binary-granularity node whenever it holds evidence.
The default order is unchanged; see ``REPO_BINARY_FIRST_ARBITRATION``.

Only observations completed strictly before a query become visible. Command
prediction composes empirical clause values by shell execution stage: pipeline
members overlap, while successive stages run sequentially.
"""

from __future__ import annotations

import hashlib
import heapq
import math
import re
from bisect import bisect_left, bisect_right, insort
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass
from decimal import Decimal, InvalidOperation
from functools import lru_cache
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

_CLAUSE_SCHEMA = "runtime_clause_resource_kb_v7"
_CLAUSE_MAX_DEPTH = 4  # frozen ordered argv-prefix depth budget
_DELIM = "\x00"  # argv tokens may contain spaces; NUL cannot collide
RAW_ARGV_REPRESENTATION = "raw-argv-prefix-v1"
GENERIC_ARGV_CANONICALIZER_VERSION = "generic-argv-v3-role"
STRUCTURED_ARGV_REPRESENTATION = GENERIC_ARGV_CANONICALIZER_VERSION
HARD_BACKOFF_ARBITRATION = "hard-first-nonempty-v1"
POSTERIOR_SHRINKAGE_ARBITRATION = "public-local-posterior-v1"
COMMAND_COMPOSITION_ARBITRATION = "empirical-shell-graph-v1"
COMMAND_COMPOSITION_DRAWS = 256
# Prefer the repository's binary-granularity node over deeper repository nodes.
# Deepest-non-empty selects sparse exact/prefix nodes that usually hold no Heavy
# observation, then falls through to a diluted cross-repository prior; the
# binary-granularity repository node is the level with both coverage and purity.
REPO_BINARY_FIRST_ARBITRATION = "repo-binary-first-v1"
SHRINKAGE_ALPHA_GRID = (1.0, 4.0, 16.0, 64.0)

# Heavy/Light decision cut on P(Heavy). 0.5 is optimal only when a false action
# and a missed Heavy cost the same; a caller whose action is asymmetric passes
# C/(B+C) instead. Not a tuned parameter -- it is declared from the action's
# cost ratio, never fitted to evaluation data.
#
# The comparison is strict, so a ratio exactly equal to the cut decides Light.
# Ties are reachable: a 6-observation node with one Heavy is bit-equal to a
# declared 1/6. Light on a tie matches the latency contract, where an exact
# probability tie selects the shorter bucket.
DEFAULT_HEAVY_DECISION_THRESHOLD = 0.5
_SUPPORTED_REPRESENTATIONS = {
    RAW_ARGV_REPRESENTATION,
    STRUCTURED_ARGV_REPRESENTATION,
}
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
_STABLE_SUBCOMMAND = re.compile(r"[A-Za-z][A-Za-z0-9_-]{0,31}")
_STABLE_SUBCOMMAND_MIN_REPOS = 3

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
    2000.0,
    8000.0,
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
    canonicalizer_version: str
    arbitration: str = HARD_BACKOFF_ARBITRATION
    local_key_kind: str | None = None
    local_evidence_count: int = 0
    public_key_kind: str | None = None
    public_evidence_count: int = 0
    shrinkage_alpha: float | None = None


@dataclass(frozen=True)
class ClauseHeavyLightPrediction:
    """Empirical Heavy/Light prediction for one clause resource."""

    resource: str
    threshold: float
    probability_heavy: float
    heavy_decision_threshold: float
    label: str
    scope: str
    key_kind: str
    evidence_count: int
    fallback_path: tuple[str, ...]
    canonicalizer_version: str
    arbitration: str = HARD_BACKOFF_ARBITRATION
    local_key_kind: str | None = None
    local_evidence_count: int = 0
    public_key_kind: str | None = None
    public_evidence_count: int = 0
    shrinkage_alpha: float | None = None


@dataclass(frozen=True)
class CommandLatencyBucketPrediction:
    """One command-level latency result composed from clause evidence."""

    repo: str
    command: str
    parse_failed: bool
    clause_bins: tuple[str, ...]
    prediction: ClauseLatencyBucketPrediction | None
    unavailable_reason: str | None = None


@dataclass(frozen=True)
class CommandResourceClassPrediction:
    """One command-level resource-class result composed from clause evidence."""

    repo: str
    command: str
    parse_failed: bool
    clause_bins: tuple[str, ...]
    classifications: dict[str, ClauseHeavyLightPrediction | None]
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


def _checked_latency(value: float) -> float:
    """Reject a latency a bucket id could not be computed from.

    The bucket histogram used to validate every value on the way past, because
    it called ``bucket_id`` per value. It is a binary search now, so validation
    happens once per value as it enters a node instead. Checking only the ends
    of the sorted node would not do: NaN compares false against everything, so
    it can sort into the middle and slip past both ends.
    """

    if not math.isfinite(value) or value < 0.0:
        raise ValueError("latency_ms must be finite and non-negative")
    return value


def _ordered_node(values: Iterable[float]) -> list[float]:
    """Node order: non-comparable values first, then ascending.

    Only latency is validated on entry, because only latency was validated
    before. The Heavy/Light sources therefore still admit NaN, and NaN has no
    position under ``<``: letting it into the sorted region leaves genuinely
    ordered values out of order, and the Heavy/Light bisect then miscounts
    values that are themselves perfectly fine.

    Holding NaN ahead of the sorted region keeps the binary-search
    precondition -- ``threshold < value`` stays false-then-true across the
    node -- and reproduces the total it replaced: ``threshold < nan`` is false,
    so a NaN counts as light exactly as ``sum(value > threshold)`` counted it,
    and it still counts toward the denominator.
    """

    materialized = list(values)
    return [value for value in materialized if math.isnan(value)] + sorted(
        value for value in materialized if not math.isnan(value)
    )


def _insert_into_node(node: list[float], value: float) -> None:
    """Insert preserving :func:`_ordered_node`'s invariant."""

    if math.isnan(value):
        node.insert(0, value)
    else:
        # bisect skips the NaN prefix on its own: `value < nan` is false, so the
        # search moves right past it into the sorted region.
        insort(node, value)


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


def _option_name(token: str) -> str | None:
    if _LONG_OPTION.fullmatch(token) or _SHORT_OPTION.fullmatch(token):
        return token
    return None


def _structured_argv_parts(
    argv: Sequence[str],
) -> tuple[str | None, list[str], list[str]]:
    """Return the possible subcommand, option multiset, and positional shapes.

    Generic option arity is unknowable without binary-specific schemas. A
    standalone option conservatively consumes one immediately following
    non-option as its shaped value. This can miss a subcommand after a boolean
    flag, but it cannot promote an option value into a raw public subcommand,
    and reordered option/value pairs remain invariant without per-binary rules.
    """

    subcommand: str | None = None
    options: list[str] = []
    positionals: list[str] = []
    operands_only = False
    tail = list(argv[1:])
    index = 0
    while index < len(tail):
        token = tail[index]
        if token == "--":
            positionals.append("boundary:--")
            operands_only = True
            index += 1
            continue
        assignment = _ENV_ASSIGNMENT.fullmatch(token)
        if assignment and not operands_only:
            options.append(
                "env:"
                f"{assignment.group(1)}={_canonical_dynamic_value(assignment.group(2))}"
            )
            index += 1
            continue
        if operands_only:
            positionals.append(_canonical_dynamic_value(token))
            index += 1
            continue
        if token.startswith("--") and "=" in token:
            flag, value = token.split("=", 1)
            name = flag if _LONG_OPTION.fullmatch(flag) else "<OPT>"
            options.append(f"{name}={_canonical_dynamic_value(value)}")
            index += 1
            continue
        attached = _SHORT_ATTACHED_VALUE.fullmatch(token)
        if attached and not _SHORT_OPTION.fullmatch(token):
            options.append(
                f"{attached.group(1)}={_canonical_dynamic_value(attached.group(2))}"
            )
            index += 1
            continue
        option = _option_name(token)
        if option is not None:
            if index + 1 < len(tail):
                following = tail[index + 1]
                if following != "--" and (
                    not following.startswith("-")
                    or _NUMBER.fullmatch(following) is not None
                ):
                    options.append(
                        f"{option}={_canonical_dynamic_value(following)}"
                    )
                    index += 2
                    continue
            options.append(option)
            index += 1
            continue
        if _NUMBER.fullmatch(token):
            if subcommand is None:
                subcommand = token
            else:
                positionals.append(_canonical_dynamic_value(token))
            index += 1
            continue
        if token.startswith("-"):
            if index + 1 < len(tail):
                following = tail[index + 1]
                if following != "--" and (
                    not following.startswith("-")
                    or _NUMBER.fullmatch(following) is not None
                ):
                    options.append(
                        f"<OPT>={_canonical_dynamic_value(following)}"
                    )
                    index += 2
                    continue
            options.append("<OPT>")
            index += 1
            continue
        if subcommand is None:
            subcommand = token
        else:
            positionals.append(_canonical_dynamic_value(token))
        index += 1
    return subcommand, options, positionals


def _structured_argv_tokens(
    bin_: str,
    argv: Sequence[str],
    stable_subcommands: frozenset[tuple[str, str]],
) -> tuple[str, ...]:
    subcommand, options, positionals = _structured_argv_parts(argv)
    if subcommand is None:
        subcommand_token = "<NONE>"
    elif (bin_, subcommand) in stable_subcommands:
        subcommand_token = subcommand
    else:
        subcommand_token = _canonical_dynamic_value(subcommand)
    return (
        f"bin:{bin_}",
        f"subcommand:{subcommand_token}",
        *(f"option:{option}" for option in sorted(options)),
        "positionals:",
        *positionals,
    )


def _fit_stable_subcommands(
    observations: Iterable[ClauseObservation],
) -> frozenset[tuple[str, str]]:
    repositories: dict[tuple[str, str], set[str]] = {}
    for obs in observations:
        candidate, _, _ = _structured_argv_parts(obs.argv)
        if candidate is None or _STABLE_SUBCOMMAND.fullmatch(candidate) is None:
            continue
        repositories.setdefault((obs.bin, candidate), set()).add(obs.repo)
    return frozenset(
        key
        for key, repos in repositories.items()
        if len(repos) >= _STABLE_SUBCOMMAND_MIN_REPOS
    )


def generic_argv_keys(
    bin_: str,
    argv: Sequence[str],
    *,
    stable_subcommands: frozenset[tuple[str, str]] = frozenset(),
) -> list[NodeKey]:
    """Development-only role-aware signature and binary backoff keys."""

    tokens = _structured_argv_tokens(bin_, argv, stable_subcommands)
    return [("structured_argv", _DELIM.join(tokens)), ("bin", bin_)]


@lru_cache(maxsize=8192)
def _clause_repo_keys_cached(bin_: str, argv_tail: tuple[str, ...]) -> tuple[NodeKey, ...]:
    # Keyed on the tail, not the whole argv: _clause_tokens drops argv[0], so
    # ("git", ("/usr/bin/git", "status")) and ("git", ("git", "status")) produce
    # identical keys and would otherwise occupy two entries and miss each other.
    # ponytail: bounded by entry count, not by key length -- argv here comes
    # from parsed agent commands, so the ceiling is fine; add a length guard if
    # argv ever originates from an untrusted source.
    tokens = (bin_, *argv_tail)
    keys: list[NodeKey] = [("exact_clause", _DELIM.join(tokens))]
    depth = min(len(tokens), _CLAUSE_MAX_DEPTH)
    # depth-1 prefix equals the bin node's content, so stop prefixes at 2.
    for length in range(depth, 1, -1):
        keys.append((f"argv_prefix_depth_{length}", _DELIM.join(tokens[:length])))
    keys.append(("bin", bin_))
    return tuple(keys)


def _clause_repo_keys(bin_: str, argv: Sequence[str]) -> tuple[NodeKey, ...]:
    """Repo backoff keys, most-specific first, for clause identity (bin, argv).

    Order: exact clause -> shorter bin-qualified argv prefixes -> bin. Every
    prefix key is nested under ``bin`` (its first token is ``bin``), so ``bin``
    is the LAST, most-general node queried — a more-specific prefix always wins
    before the bare bin node.

    Memoized: ``_select`` rebuilds the identical key list once per value source
    on every query, and the joins dominate an otherwise O(log n) lookup.
    """

    return _clause_repo_keys_cached(bin_, tuple(argv[1:]))


def _clause_public_keys(bin_: str) -> tuple[NodeKey, ...]:
    """Public clause keys: coarse bin prior then global."""

    return (("bin", bin_), ("global", ""))


def _command_stages(
    clauses: Sequence[Mapping[str, Any]],
) -> tuple[tuple[int, ...], ...] | None:
    """Group ordered clauses into sequential stages of concurrent pipelines."""

    stages: list[tuple[int, ...]] = []
    pipeline: list[int] = []
    for index, clause in enumerate(clauses):
        if clause.get("in_subst") is True:
            return None
        if clause.get("in_pipe") is not True:
            if pipeline:
                stages.append(tuple(pipeline))
                pipeline = []
            stages.append((index,))
            continue
        position = clause.get("pipeline_position")
        if position == 0:
            if pipeline:
                stages.append(tuple(pipeline))
            pipeline = [index]
        elif isinstance(position, int) and position == len(pipeline):
            pipeline.append(index)
        else:
            return None
    if pipeline:
        stages.append(tuple(pipeline))
    return tuple(stages)


def _stratified_empirical_draws(
    values: Sequence[float],
    seed_material: str,
) -> tuple[float, ...]:
    """Deterministic marginally uniform draws without runtime RNG state."""

    digest = hashlib.blake2s(seed_material.encode(), digest_size=4).digest()
    offset = int.from_bytes(digest, "big") % COMMAND_COMPOSITION_DRAWS
    return tuple(
        values[
            ((draw + offset) % COMMAND_COMPOSITION_DRAWS)
            * len(values)
            // COMMAND_COMPOSITION_DRAWS
        ]
        for draw in range(COMMAND_COMPOSITION_DRAWS)
    )


class ClauseResourceKB:
    """Causal clause history with latency and resource-class APIs.

    Public priors are frozen after construction; repo nodes accumulate causally
    under a monotonic-query guard. ``representation`` selects either the
    reviewed raw-prefix hierarchy or the frozen Candidate R structured arm.
    """

    def __init__(
        self,
        *,
        representation: str = RAW_ARGV_REPRESENTATION,
        stable_subcommands: frozenset[tuple[str, str]] = frozenset(),
        shrinkage_alpha: float | None = None,
        heavy_decision_threshold: float = DEFAULT_HEAVY_DECISION_THRESHOLD,
        repo_binary_first: bool = False,
    ) -> None:
        if repo_binary_first and shrinkage_alpha is not None:
            # Shrinkage selects its scopes independently and never consults
            # _select, so the two arbitrations cannot be composed coherently.
            raise ValueError(
                "repo_binary_first and posterior shrinkage are exclusive arbitrations"
            )
        # The open-interval test alone rejects NaN, both infinities, and bool
        # (True -> 1.0, False -> 0.0). Unlike shrinkage_alpha, which tests grid
        # membership and would silently accept True as 1.0, no separate bool or
        # isfinite guard is reachable here.
        if not 0.0 < float(heavy_decision_threshold) < 1.0:
            raise ValueError(
                "heavy_decision_threshold must be a finite probability in (0, 1)"
            )
        if representation not in _SUPPORTED_REPRESENTATIONS:
            raise ValueError(f"unsupported clause representation {representation!r}")
        if representation == RAW_ARGV_REPRESENTATION and stable_subcommands:
            raise ValueError("raw argv representation cannot carry subcommand vocabulary")
        if shrinkage_alpha is not None:
            if representation != STRUCTURED_ARGV_REPRESENTATION:
                raise ValueError("posterior shrinkage requires structured argv")
            if (
                isinstance(shrinkage_alpha, bool)
                or not math.isfinite(shrinkage_alpha)
                or float(shrinkage_alpha) not in SHRINKAGE_ALPHA_GRID
            ):
                raise ValueError(
                    f"shrinkage alpha must be one of {SHRINKAGE_ALPHA_GRID}"
                )
        self._representation = representation
        self._heavy_decision_threshold = float(heavy_decision_threshold)
        self._repo_binary_first = bool(repo_binary_first)
        self._stable_subcommands = stable_subcommands
        self._shrinkage_alpha = (
            None if shrinkage_alpha is None else float(shrinkage_alpha)
        )
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
        *,
        representation: str = RAW_ARGV_REPRESENTATION,
        shrinkage_alpha: float | None = None,
        heavy_decision_threshold: float = DEFAULT_HEAVY_DECISION_THRESHOLD,
        repo_binary_first: bool = False,
    ) -> ClauseResourceKB:
        """Fit frozen public priors and any label-free fit vocabulary."""

        if representation == STRUCTURED_ARGV_REPRESENTATION:
            materialized = list(observations)
            stable_subcommands = _fit_stable_subcommands(materialized)
            observations = materialized
        else:
            stable_subcommands = frozenset()
        kb = cls(
            representation=representation,
            stable_subcommands=stable_subcommands,
            shrinkage_alpha=shrinkage_alpha,
            heavy_decision_threshold=heavy_decision_threshold,
            repo_binary_first=repo_binary_first,
        )
        acc: dict[str, dict[NodeKey, list[float]]] = {
            source: {} for source in _CLAUSE_SOURCES
        }
        for obs in observations:
            keys = kb._public_keys(obs.bin, obs.argv)
            for source in _CLAUSE_SOURCES:
                value = _clause_value(obs, source)
                if value is None:
                    continue
                if source == _LATENCY_MS:
                    _checked_latency(value)
                for key in keys:
                    acc[source].setdefault(key, []).append(value)
        if not acc[_LATENCY_MS].get(("global", "")):
            raise ValueError("fit corpus has no clause latency evidence")
        # Nodes are held sorted so predictions are binary searches, not scans.
        kb._public = {
            source: {key: tuple(_ordered_node(values)) for key, values in nodes.items()}
            for source, nodes in acc.items()
        }
        return kb

    @property
    def representation(self) -> str:
        return self._representation

    @property
    def canonicalizer_version(self) -> str:
        return self._representation

    @property
    def arbitration(self) -> str:
        if self._shrinkage_alpha is not None:
            return POSTERIOR_SHRINKAGE_ARBITRATION
        if self._repo_binary_first:
            return REPO_BINARY_FIRST_ARBITRATION
        return HARD_BACKOFF_ARBITRATION

    @property
    def shrinkage_alpha(self) -> float | None:
        return self._shrinkage_alpha

    @property
    def heavy_decision_threshold(self) -> float:
        """Cut on P(Heavy). Result-affecting, so a formal run must record it."""

        return self._heavy_decision_threshold

    def _repo_keys(self, bin_: str, argv: Sequence[str]) -> tuple[NodeKey, ...]:
        if self._representation == RAW_ARGV_REPRESENTATION:
            return _clause_repo_keys(bin_, argv)
        structured, binary = generic_argv_keys(
            bin_,
            argv,
            stable_subcommands=self._stable_subcommands,
        )
        exact = ("exact_clause", _DELIM.join(_clause_tokens(bin_, argv)))
        return (exact, structured, binary)

    def _public_keys(self, bin_: str, argv: Sequence[str]) -> tuple[NodeKey, ...]:
        if self._representation == RAW_ARGV_REPRESENTATION:
            return _clause_public_keys(bin_)
        structured, binary = generic_argv_keys(
            bin_,
            argv,
            stable_subcommands=self._stable_subcommands,
        )
        return (structured, binary, ("global", ""))

    def observe_completed_clause(self, obs: ClauseObservation) -> None:
        """Buffer a completed clause; visible only once strictly causally prior.

        An unusable latency is refused here, on submission, rather than when the
        buffer drains. Draining pops the observation before it could be checked,
        so a rejection there destroyed the evidence it rejected: the query
        raised once and every later query then succeeded over a corpus quietly
        missing that clause. Draining also happens inside every query, so a bad
        latency aborted predictions for unrelated repositories and resources.
        """

        if obs.latency_ms is not None:
            _checked_latency(obs.latency_ms)
        heapq.heappush(self._pending, (obs.ts_end, self._pending_seq, obs))
        self._pending_seq += 1

    def _absorb_completed(self, ts_start: float) -> None:
        while self._pending and self._pending[0][0] < ts_start:
            _, _, obs = heapq.heappop(self._pending)
            repo_sources = self._repo.setdefault(
                obs.repo, {source: {} for source in _CLAUSE_SOURCES}
            )
            keys = self._repo_keys(obs.bin, obs.argv)
            for source in _CLAUSE_SOURCES:
                value = _clause_value(obs, source)
                if value is None:
                    continue
                for key in keys:
                    # Keep each node ordered on insert: absorption happens once
                    # per observation, prediction happens on every clause.
                    _insert_into_node(repo_sources[source].setdefault(key, []), value)

    def _ordered_repo_keys(
        self, bin_: str, argv: Sequence[str]
    ) -> tuple[NodeKey, ...]:
        """Repository keys in consultation order for the active arbitration.

        ``repo_binary_first`` moves the binary-granularity node ahead of the
        deeper exact/prefix nodes. Reordering happens HERE rather than in
        ``_select`` so that ``_select`` remains exactly the first candidate --
        the invariant ``diagnostic_clause_latency_candidates`` documents and the
        latency evaluator asserts.
        """

        keys = self._repo_keys(bin_, argv)
        if not self._repo_binary_first:
            return keys
        binary = ("bin", bin_)
        return (binary, *(key for key in keys if key != binary))

    def _candidate_nodes(
        self, repo: str, source: str, bin_: str, argv: Sequence[str]
    ) -> Iterator[tuple[Sequence[float], str, str, tuple[str, ...]]]:
        """Yield non-empty backoff nodes in runtime selection order."""

        repo_nodes = self._repo.get(repo, {}).get(source, {})
        public_nodes = self._public[source]
        path: list[str] = []
        for key in self._ordered_repo_keys(bin_, argv):
            path.append(f"repo:{key[0]}")
            values = repo_nodes.get(key)
            if values:
                yield values, "repo", key[0], tuple(path)
        for key in self._public_keys(bin_, argv):
            path.append(f"public:{key[0]}")
            values = public_nodes.get(key)
            if values:
                yield values, "public", key[0], tuple(path)

    def _select(
        self, repo: str, source: str, bin_: str, argv: Sequence[str]
    ) -> tuple[Sequence[float], str, str, tuple[str, ...]] | None:
        return next(self._candidate_nodes(repo, source, bin_, argv), None)

    def _select_independent_scopes(
        self,
        repo: str,
        source: str,
        bin_: str,
        argv: Sequence[str],
    ) -> tuple[
        tuple[Sequence[float], str, str, tuple[str, ...]] | None,
        tuple[Sequence[float], str, str, tuple[str, ...]] | None,
    ]:
        repo_nodes = self._repo.get(repo, {}).get(source, {})
        public_nodes = self._public[source]
        repo_path: list[str] = []
        local = None
        for key in self._repo_keys(bin_, argv):
            repo_path.append(f"repo:{key[0]}")
            values = repo_nodes.get(key)
            if values:
                local = (values, "repo", key[0], tuple(repo_path))
                break
        public_path: list[str] = []
        public = None
        for key in self._public_keys(bin_, argv):
            public_path.append(f"public:{key[0]}")
            values = public_nodes.get(key)
            if values:
                public = (values, "public", key[0], tuple(public_path))
                break
        return local, public

    @staticmethod
    def _bucket_probabilities(
        values: Sequence[float],
        buckets: LatencyBuckets,
    ) -> tuple[float, ...]:
        # Nodes are sorted, so the histogram is one binary search per edge.
        counts: list[int] = []
        at_or_below_previous = 0
        for edge in buckets.edges_ms:
            at_or_below = bisect_right(values, edge)
            counts.append(at_or_below - at_or_below_previous)
            at_or_below_previous = at_or_below
        counts.append(len(values) - at_or_below_previous)
        return tuple(count / len(values) for count in counts)

    def _latency_prediction(
        self,
        selected: tuple[Sequence[float], str, str, tuple[str, ...]],
        buckets: LatencyBuckets,
    ) -> ClauseLatencyBucketPrediction:
        values, scope, kind, path = selected
        return ClauseLatencyBucketPrediction(
            probability_by_bucket=self._bucket_probabilities(values, buckets),
            scope=scope,
            key_kind=kind,
            evidence_count=len(values),
            fallback_path=path,
            canonicalizer_version=self.canonicalizer_version,
            arbitration=self.arbitration,
        )

    def _posterior_latency_prediction(
        self,
        local: tuple[Sequence[float], str, str, tuple[str, ...]] | None,
        public: tuple[Sequence[float], str, str, tuple[str, ...]] | None,
        buckets: LatencyBuckets,
        alpha: float,
    ) -> ClauseLatencyBucketPrediction:
        if local is None and public is None:
            raise ValueError("no local or public clause latency node")
        local_values = () if local is None else local[0]
        public_values = () if public is None else public[0]
        if not local_values:
            probabilities = self._bucket_probabilities(public_values, buckets)
        elif not public_values:
            probabilities = self._bucket_probabilities(local_values, buckets)
        else:
            local_probabilities = self._bucket_probabilities(local_values, buckets)
            public_probabilities = self._bucket_probabilities(public_values, buckets)
            denominator = len(local_values) + alpha
            probabilities = tuple(
                (
                    len(local_values) * local_probability
                    + alpha * public_probability
                )
                / denominator
                for local_probability, public_probability in zip(
                    local_probabilities,
                    public_probabilities,
                    strict=True,
                )
            )
        local_kind = None if local is None else local[2]
        public_kind = None if public is None else public[2]
        scope = (
            "repo+public"
            if local is not None and public is not None
            else ("repo" if local is not None else "public")
        )
        return ClauseLatencyBucketPrediction(
            probability_by_bucket=probabilities,
            scope=scope,
            key_kind="+".join(
                kind for kind in (local_kind, public_kind) if kind is not None
            ),
            evidence_count=len(local_values) + len(public_values),
            fallback_path=(
                *(local[3] if local is not None else ()),
                *(public[3] if public is not None else ()),
            ),
            canonicalizer_version=self.canonicalizer_version,
            arbitration=POSTERIOR_SHRINKAGE_ARBITRATION,
            local_key_kind=local_kind,
            local_evidence_count=len(local_values),
            public_key_kind=public_kind,
            public_evidence_count=len(public_values),
            shrinkage_alpha=alpha,
        )

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
        if self._shrinkage_alpha is not None:
            local, public = self._select_independent_scopes(
                repo,
                _LATENCY_MS,
                bin_,
                argv,
            )
            return self._posterior_latency_prediction(
                local,
                public,
                buckets,
                self._shrinkage_alpha,
            )
        selected = self._select(repo, _LATENCY_MS, bin_, argv)
        if selected is None:
            raise ValueError("no public global clause latency node")
        return self._latency_prediction(selected, buckets)

    def diagnostic_clause_latency_shrinkage_predictions(
        self,
        repo: str,
        bin_: str,
        argv: Sequence[str],
        buckets: LatencyBuckets,
        alphas: Sequence[float] = SHRINKAGE_ALPHA_GRID,
        *,
        ts_start: float | None = None,
    ) -> tuple[ClauseLatencyBucketPrediction, ...]:
        """Fit-only alpha candidates using the runtime posterior implementation."""

        if self._representation != STRUCTURED_ARGV_REPRESENTATION:
            raise ValueError("posterior shrinkage requires structured argv")
        if ts_start is not None:
            self._advance(ts_start)
        local, public = self._select_independent_scopes(
            repo,
            _LATENCY_MS,
            bin_,
            argv,
        )
        predictions = []
        for alpha in alphas:
            if float(alpha) not in SHRINKAGE_ALPHA_GRID:
                raise ValueError(
                    f"shrinkage alpha must be one of {SHRINKAGE_ALPHA_GRID}"
                )
            predictions.append(
                self._posterior_latency_prediction(
                    local,
                    public,
                    buckets,
                    float(alpha),
                )
            )
        return tuple(predictions)

    def diagnostic_clause_latency_candidates(
        self,
        repo: str,
        bin_: str,
        argv: Sequence[str],
        buckets: LatencyBuckets,
        *,
        ts_start: float | None = None,
    ) -> tuple[ClauseLatencyBucketPrediction, ...]:
        """Return all non-empty backoff nodes for analysis-only oracle ceilings.

        The first item is exactly the runtime-selected prediction. Callers must
        never use a later item for deployment selection: choosing among these
        candidates requires the observed label and is therefore hindsight.
        """

        if ts_start is not None:
            self._advance(ts_start)
        empirical = tuple(
            self._latency_prediction(selected, buckets)
            for selected in self._candidate_nodes(repo, _LATENCY_MS, bin_, argv)
        )
        if self._shrinkage_alpha is None:
            return empirical
        local, public = self._select_independent_scopes(
            repo,
            _LATENCY_MS,
            bin_,
            argv,
        )
        posterior = self._posterior_latency_prediction(
            local,
            public,
            buckets,
            self._shrinkage_alpha,
        )
        return (posterior, *empirical)

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
        if self._shrinkage_alpha is not None:
            local, public = self._select_independent_scopes(
                repo,
                resource,
                bin_,
                argv,
            )
            if local is None and public is None:
                return None
            local_values = () if local is None else local[0]
            public_values = () if public is None else public[0]
            local_heavy = len(local_values) - bisect_right(local_values, threshold)
            public_probability = (
                0.0
                if not public_values
                else (
                    len(public_values) - bisect_right(public_values, threshold)
                )
                / len(public_values)
            )
            if not local_values:
                probability_heavy = public_probability
            elif not public_values:
                probability_heavy = local_heavy / len(local_values)
            else:
                probability_heavy = (
                    local_heavy + self._shrinkage_alpha * public_probability
                ) / (len(local_values) + self._shrinkage_alpha)
            local_kind = None if local is None else local[2]
            public_kind = None if public is None else public[2]
            return ClauseHeavyLightPrediction(
                resource=resource,
                threshold=threshold,
                probability_heavy=probability_heavy,
                heavy_decision_threshold=self._heavy_decision_threshold,
                label=(
                    "heavy"
                    if probability_heavy > self._heavy_decision_threshold
                    else "light"
                ),
                scope=(
                    "repo+public"
                    if local is not None and public is not None
                    else ("repo" if local is not None else "public")
                ),
                key_kind="+".join(
                    kind
                    for kind in (local_kind, public_kind)
                    if kind is not None
                ),
                evidence_count=len(local_values) + len(public_values),
                fallback_path=(
                    *(local[3] if local is not None else ()),
                    *(public[3] if public is not None else ()),
                ),
                canonicalizer_version=self.canonicalizer_version,
                arbitration=POSTERIOR_SHRINKAGE_ARBITRATION,
                local_key_kind=local_kind,
                local_evidence_count=len(local_values),
                public_key_kind=public_kind,
                public_evidence_count=len(public_values),
                shrinkage_alpha=self._shrinkage_alpha,
            )
        selected = self._select(repo, resource, bin_, argv)
        if selected is None:
            return None
        values, scope, kind, path = selected
        # Sorted node: everything after the threshold's right-insertion point is
        # strictly greater, so the Heavy count is one binary search.
        probability_heavy = (
            len(values) - bisect_right(values, threshold)
        ) / len(values)
        return ClauseHeavyLightPrediction(
            resource=resource,
            threshold=threshold,
            probability_heavy=probability_heavy,
            heavy_decision_threshold=self._heavy_decision_threshold,
            label=(
                "heavy"
                if probability_heavy > self._heavy_decision_threshold
                else "light"
            ),
            scope=scope,
            key_kind=kind,
            evidence_count=len(values),
            fallback_path=path,
            canonicalizer_version=self.canonicalizer_version,
            arbitration=self.arbitration,
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

    def _composed_command_values(
        self,
        repo: str,
        command: str,
        clauses: Sequence[Mapping[str, Any]],
        source: str,
    ) -> tuple[
        tuple[float, ...],
        tuple[tuple[Sequence[float], str, str, tuple[str, ...]], ...],
    ] | None:
        if self._shrinkage_alpha is not None:
            return None
        stages = _command_stages(clauses)
        if stages is None:
            return None
        selected = tuple(
            self._select(
                repo,
                source,
                str(clause["bin"]),
                tuple(str(value) for value in clause["argv"]),
            )
            for clause in clauses
        )
        if any(node is None for node in selected):
            return None
        nodes = tuple(node for node in selected if node is not None)
        draws_by_clause = [
            _stratified_empirical_draws(
                node[0],
                f"{command}\0{source}\0{index}\0{node[1]}\0{node[2]}",
            )
            for index, node in enumerate(nodes)
        ]
        composed: list[float] = []
        for draw in range(COMMAND_COMPOSITION_DRAWS):
            stage_values = [
                (
                    max(draws_by_clause[index][draw] for index in stage)
                    if source == _LATENCY_MS
                    else sum(draws_by_clause[index][draw] for index in stage)
                )
                for stage in stages
            ]
            composed.append(
                sum(stage_values)
                if source in {_LATENCY_MS, _DISK_READ_WRITE_BYTES_TOTAL}
                else max(stage_values)
            )
        return tuple(_ordered_node(composed)), nodes

    def _composed_prediction_provenance(
        self,
        nodes: Sequence[tuple[Sequence[float], str, str, tuple[str, ...]]],
    ) -> dict[str, Any]:
        scopes = {node[1] for node in nodes}
        return {
            "scope": (
                "repo+public"
                if scopes == {"repo", "public"}
                else next(iter(scopes))
            ),
            "key_kind": "shell_execution_graph",
            "evidence_count": min(len(node[0]) for node in nodes),
            "fallback_path": tuple(
                f"clause[{index}]:{node[1]}:{node[2]}"
                for index, node in enumerate(nodes)
            ),
            "canonicalizer_version": self.canonicalizer_version,
            "arbitration": COMMAND_COMPOSITION_ARBITRATION,
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
        """Predict one command, composing empirical clause values when needed."""

        self._advance(ts_start)
        effective = list(clauses)
        clause_bins = tuple(str(c["bin"]) for c in effective)
        reason = None
        prediction = None
        if parse_failed:
            reason = "parse_failed"
        elif not effective:
            reason = "no_executable_clause"
        elif _command_stages(effective) is None:
            reason = "compound_composition_unavailable"
        elif len(effective) == 1:
            clause = effective[0]
            prediction = self.predict_clause_latency_bucket(
                repo,
                str(clause["bin"]),
                tuple(clause["argv"]),
                buckets,
            )
        else:
            composed = self._composed_command_values(
                repo,
                command,
                effective,
                _LATENCY_MS,
            )
            if composed is None:
                reason = "compound_composition_unavailable"
            else:
                values, nodes = composed
                prediction = ClauseLatencyBucketPrediction(
                    probability_by_bucket=self._bucket_probabilities(values, buckets),
                    **self._composed_prediction_provenance(nodes),
                )
        return CommandLatencyBucketPrediction(
            repo=repo,
            command=command,
            parse_failed=parse_failed,
            clause_bins=clause_bins,
            prediction=prediction,
            unavailable_reason=reason,
        )

    def predict_command_resource_classes_from_clauses(
        self,
        repo: str,
        clauses: Sequence[Mapping[str, Any]],
        ts_start: float,
        *,
        command: str = "",
        parse_failed: bool = False,
    ) -> CommandResourceClassPrediction:
        """Predict command CPU/RSS/Disk classes with the shell composer."""

        self._advance(ts_start)
        effective = list(clauses)
        clause_bins = tuple(str(clause["bin"]) for clause in effective)
        if parse_failed:
            reason = "parse_failed"
        elif not effective:
            reason = "no_executable_clause"
        elif _command_stages(effective) is None or (
            len(effective) > 1 and self._shrinkage_alpha is not None
        ):
            reason = "compound_composition_unavailable"
        else:
            reason = None
        classifications: dict[str, ClauseHeavyLightPrediction | None] = {}
        if reason is None:
            for resource, threshold in CANONICAL_RESOURCE_HEAVY_THRESHOLDS.items():
                if len(effective) == 1:
                    clause = effective[0]
                    classifications[resource] = self.predict_clause_heavy_light(
                        repo,
                        str(clause["bin"]),
                        tuple(clause["argv"]),
                        resource,
                    )
                    continue
                composed = self._composed_command_values(
                    repo,
                    command,
                    effective,
                    resource,
                )
                if composed is None:
                    classifications[resource] = None
                    continue
                values, nodes = composed
                probability_heavy = (
                    len(values) - bisect_right(values, threshold)
                ) / len(values)
                classifications[resource] = ClauseHeavyLightPrediction(
                    resource=resource,
                    threshold=threshold,
                    probability_heavy=probability_heavy,
                    heavy_decision_threshold=self._heavy_decision_threshold,
                    label=(
                        "heavy"
                        if probability_heavy > self._heavy_decision_threshold
                        else "light"
                    ),
                    **self._composed_prediction_provenance(nodes),
                )
        return CommandResourceClassPrediction(
            repo=repo,
            command=command,
            parse_failed=parse_failed,
            clause_bins=clause_bins,
            classifications=classifications,
            unavailable_reason=reason,
        )

    def predict_command_latency_bucket(
        self,
        repo: str,
        command: str,
        ts_start: float,
        buckets: LatencyBuckets,
    ) -> CommandLatencyBucketPrediction:
        """Parse a command and predict its command-level latency bucket.

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

    def predict_command_resource_classes(
        self,
        repo: str,
        command: str,
        ts_start: float,
    ) -> CommandResourceClassPrediction:
        parsed = parse_command_clauses(command)
        return self.predict_command_resource_classes_from_clauses(
            repo,
            parsed["clauses"],
            ts_start,
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
            "representation": self._representation,
            "canonicalizer_version": self.canonicalizer_version,
            "arbitration": self.arbitration,
            "shrinkage_alpha": self._shrinkage_alpha,
            "heavy_decision_threshold": self._heavy_decision_threshold,
            "repo_binary_first": self._repo_binary_first,
            "stable_subcommands": [
                [bin_, subcommand]
                for bin_, subcommand in sorted(self._stable_subcommands)
            ],
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

        schema = obj.get("schema")
        if schema not in {_CLAUSE_SCHEMA, "runtime_clause_resource_kb_v6"}:
            if obj.get("schema") == "runtime_clause_resource_kb_v5":
                raise ValueError(
                    "runtime_clause_resource_kb_v5 lacks Disk and short-null "
                    "resource labels; refit the snapshot"
                )
            raise ValueError(f"unsupported clause schema {obj.get('schema')!r}")
        if obj.get("max_prefix_depth") != _CLAUSE_MAX_DEPTH:
            raise ValueError("snapshot prefix depth differs from module depth")
        representation = str(obj.get("representation", RAW_ARGV_REPRESENTATION))
        if schema == "runtime_clause_resource_kb_v6" and (
            representation != RAW_ARGV_REPRESENTATION
            or obj.get("shrinkage_alpha") is not None
            or obj.get("arbitration", HARD_BACKOFF_ARBITRATION)
            != HARD_BACKOFF_ARBITRATION
        ):
            raise ValueError(
                "runtime_clause_resource_kb_v6 cannot identify structured or "
                "shrinkage semantics safely; refit the snapshot"
            )
        canonicalizer_version = str(
            obj.get("canonicalizer_version", representation)
        )
        if canonicalizer_version != representation:
            raise ValueError("snapshot representation and canonicalizer differ")
        stable_subcommands = frozenset(
            (str(row[0]), str(row[1]))
            for row in obj.get("stable_subcommands", [])
        )
        kb = cls(
            representation=representation,
            stable_subcommands=stable_subcommands,
            shrinkage_alpha=(
                None
                if obj.get("shrinkage_alpha") is None
                else float(obj["shrinkage_alpha"])
            ),
            heavy_decision_threshold=float(
                obj.get("heavy_decision_threshold", DEFAULT_HEAVY_DECISION_THRESHOLD)
            ),
            repo_binary_first=bool(obj.get("repo_binary_first", False)),
        )
        if obj.get("arbitration", kb.arbitration) != kb.arbitration:
            raise ValueError("snapshot arbitration and shrinkage alpha differ")
        # Re-sort on load: a snapshot written before nodes were held sorted, or
        # hand-edited, must still satisfy the binary-search invariant. Latency
        # values are revalidated here for the same reason they are validated on
        # insert -- a restored node is never scanned again.
        def _restore(source: str, values: list[float]) -> list[float]:
            if source == _LATENCY_MS:
                for value in values:
                    _checked_latency(value)
            return _ordered_node(values)

        kb._public = {
            source: {
                key: tuple(_restore(source, values))
                for key, values in _nodes_from_json(obj["public"].get(source, []))
            }
            for source in _CLAUSE_SOURCES
        }
        kb._repo = {
            repo: {
                source: {
                    key: _restore(source, values)
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
    "COMMAND_COMPOSITION_ARBITRATION",
    "COMMAND_COMPOSITION_DRAWS",
    "GENERIC_ARGV_CANONICALIZER_VERSION",
    "HARD_BACKOFF_ARBITRATION",
    "POSTERIOR_SHRINKAGE_ARBITRATION",
    "RAW_ARGV_REPRESENTATION",
    "SHRINKAGE_ALPHA_GRID",
    "SHORT_NULL_LIGHT_MAX_LATENCY_MS",
    "STRUCTURED_ARGV_REPRESENTATION",
    "ClauseHeavyLightPrediction",
    "ClauseLatencyBucketPrediction",
    "ClauseObservation",
    "ClauseResourceKB",
    "CommandLatencyBucketPrediction",
    "CommandResourceClassPrediction",
    "LatencyBuckets",
    "generic_argv_keys",
]

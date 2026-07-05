from __future__ import annotations

import re

# Unix epoch timestamps: contemporary seconds (10 digits) and millisecond/
# microsecond/nanosecond variants (13-16 digits) are volatile run metadata.
_EPOCH_TS_RE = re.compile(r"\b[12]\d{9}(?:\.\d+)?\b|\b[12]\d{12,15}\b")

# ISO-like timestamps with a date and clock component are run-specific metadata.
_ISO_TS_RE = re.compile(
    r"\b\d{4}-\d{2}-\d{2}[T ][0-2]\d:[0-5]\d:[0-5]\d"
    r"(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})?\b"
)

# Process paths include volatile PIDs and procfs pseudo-file names.
_PROC_PATH_RE = re.compile(r"/proc/\d+(?:/[^\s\"'`]+)?")

# Temporary paths commonly include randomized directory or file names.
_TMP_PATH_RE = re.compile(r"/tmp/[^\s\"'`]+")

# Hex addresses are process-specific and commonly appear in native traces.
_HEX_ADDRESS_RE = re.compile(r"\b0x[0-9a-fA-F]+\b")

# Remaining digit runs cover PIDs, counters, ports, durations, and IDs after
# the more specific timestamp/path/address patterns above have been removed.
_DIGIT_RE = re.compile(r"\d+")


def normalize_tool_output(text: str) -> str:
    """Normalize volatile, non-benchmark-specific tool-output fragments.

    The normalizer intentionally uses only general runtime volatility patterns:
    Unix epoch timestamps, ISO timestamps, procfs PID paths, temporary paths,
    hexadecimal addresses, and residual digit runs. It does not encode dataset,
    benchmark, command, or task-specific knowledge.
    """
    normalized = _ISO_TS_RE.sub("<TS>", text)
    normalized = _EPOCH_TS_RE.sub("<TS>", normalized)
    normalized = _PROC_PATH_RE.sub("<PROC>", normalized)
    normalized = _TMP_PATH_RE.sub("<TMP>", normalized)
    normalized = _HEX_ADDRESS_RE.sub("<HEX>", normalized)
    return _DIGIT_RE.sub("<N>", normalized)

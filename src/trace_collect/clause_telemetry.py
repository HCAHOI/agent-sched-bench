"""Compatibility alias for the tool-resource SDK telemetry module."""

from __future__ import annotations

import sys

from tool_resource import telemetry as _telemetry

sys.modules[__name__] = _telemetry

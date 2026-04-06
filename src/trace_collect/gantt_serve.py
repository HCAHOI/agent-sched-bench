"""Generate and serve self-contained Gantt HTML from trace data."""

from __future__ import annotations

import tempfile
import webbrowser
from pathlib import Path
from string import Template
from typing import Any

from trace_collect.gantt_data import build_gantt_payload_multi
from trace_collect.trace_inspector import TraceData

_TEMPLATE_PATH = Path(__file__).parent / "gantt_template.html"


def generate_gantt_html(
    traces: list[tuple[str, TraceData]],
) -> str:
    """Render the Gantt template with embedded trace data.

    Args:
        traces: List of (label, TraceData) pairs.

    Returns:
        Complete HTML string with JSON payload injected.
    """
    import json

    payload = build_gantt_payload_multi(traces)
    payload_json = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))

    template_html = _TEMPLATE_PATH.read_text(encoding="utf-8")
    return template_html.replace("__TRACE_JSON__", payload_json)


def write_gantt(html: str, output: Path) -> None:
    """Write generated HTML to a file."""
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(html, encoding="utf-8")


def open_gantt(html: str) -> Path:
    """Write HTML to a temp file and open in the default browser.

    Returns:
        Path to the temp file.
    """
    fd = tempfile.NamedTemporaryFile(
        suffix=".html", prefix="gantt_", delete=False, mode="w", encoding="utf-8"
    )
    fd.write(html)
    fd.close()
    path = Path(fd.name)
    webbrowser.open(f"file://{path}")
    return path


def serve_gantt(html: str, port: int = 0) -> None:
    """Start a local HTTP server serving the Gantt HTML.

    Args:
        port: Port number. 0 = auto-select free port.
    """
    import http.server
    import io
    import socket
    import threading

    encoded = html.encode("utf-8")

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def log_message(self, *args: Any) -> None:
            pass  # suppress request logs

    if port == 0:
        with socket.socket() as s:
            s.bind(("", 0))
            port = s.getsockname()[1]

    server = http.server.HTTPServer(("127.0.0.1", port), Handler)
    url = f"http://127.0.0.1:{port}"
    print(f"Serving Gantt at {url}  (Ctrl+C to stop)")
    webbrowser.open(url)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped.")
    finally:
        server.server_close()

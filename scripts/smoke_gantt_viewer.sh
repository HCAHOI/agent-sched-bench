#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if ! command -v npx >/dev/null 2>&1; then
  echo "npx is required for the browser smoke test" >&2
  exit 1
fi

PORT="${GANTT_VIEWER_PORT:-8765}"
SESSION="gsmoke"
PYTHON_BIN="${PYTHON_BIN:-./.venv/bin/python}"

cleanup() {
  npx --yes --package @playwright/cli playwright-cli -s="$SESSION" close >/dev/null 2>&1 || true
  if [[ -n "${SERVER_PID:-}" ]] && kill -0 "$SERVER_PID" >/dev/null 2>&1; then
    kill "$SERVER_PID" >/dev/null 2>&1 || true
    wait "$SERVER_PID" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT

PYTHONPATH=src:. "$PYTHON_BIN" -m trace_collect.cli gantt-serve \
  --port "$PORT" \
  --no-browser \
  >/tmp/gantt-viewer-smoke.log 2>&1 &
SERVER_PID=$!

for _ in $(seq 1 50); do
  if curl -sf "http://127.0.0.1:${PORT}/api/health" >/dev/null; then
    break
  fi
  sleep 0.2
done

curl -sf "http://127.0.0.1:${PORT}/api/health" >/dev/null

npx --yes --package @playwright/cli playwright-cli -s="$SESSION" open "http://127.0.0.1:${PORT}" >/dev/null

default_loaded="$(
  npx --yes --package @playwright/cli playwright-cli -s="$SESSION" eval --raw \
    "() => [...document.querySelectorAll('strong')].map((el) => el.textContent).includes('1')"
)"
if [[ "$default_loaded" != "true" ]]; then
  echo "expected default loaded count of 1, got: $default_loaded" >&2
  exit 1
fi

npx --yes --package @playwright/cli playwright-cli -s="$SESSION" eval --raw \
  "() => { const lines = [JSON.stringify({type:'trace_metadata', scaffold:'synthetic', model:'demo', trace_format_version:5, max_iterations:1}), JSON.stringify({type:'action', action_type:'llm_call', action_id:'llm_0', agent_id:'agent-1', iteration:0, ts_start:1000, ts_end:1000.25, data:{raw_response:{choices:[{message:{content:'drag upload'}}]}}})].join('\n'); const file = new File([lines], 'drag_trace.jsonl', { type: 'application/json' }); const dt = new DataTransfer(); dt.items.add(file); window.dispatchEvent(new DragEvent('drop', { dataTransfer: dt })); return true; }" \
  >/dev/null

after_drop="$(
  npx --yes --package @playwright/cli playwright-cli -s="$SESSION" eval --raw \
    "async () => { for (let i = 0; i < 40; i += 1) { const values = [...document.querySelectorAll('strong')].map((el) => el.textContent); if (values.includes('2')) return values.join('|'); await new Promise((resolve) => setTimeout(resolve, 100)); } return [...document.querySelectorAll('strong')].map((el) => el.textContent).join('|'); }"
)"

if [[ "$after_drop" != *"|2|"* && "$after_drop" != 2\|* && "$after_drop" != *\|2 ]]; then
  echo "expected loaded count of 2 after synthetic drag-drop, got: $after_drop" >&2
  exit 1
fi

npx --yes --package @playwright/cli playwright-cli -s="$SESSION" click e30 >/dev/null
npx --yes --package @playwright/cli playwright-cli -s="$SESSION" upload tests/fixtures/claude_code_minimal.jsonl >/dev/null

after_upload="$(
  npx --yes --package @playwright/cli playwright-cli -s="$SESSION" eval --raw \
    "async () => { for (let i = 0; i < 40; i += 1) { const values = [...document.querySelectorAll('strong')].map((el) => el.textContent); if (values.includes('3')) return values.join('|'); await new Promise((resolve) => setTimeout(resolve, 100)); } return [...document.querySelectorAll('strong')].map((el) => el.textContent).join('|'); }"
)"

if [[ "$after_upload" != *"|3|"* && "$after_upload" != 3\|* && "$after_upload" != *\|3 ]]; then
  echo "expected loaded count of 3 after button upload, got: $after_upload" >&2
  exit 1
fi

npx --yes --package @playwright/cli playwright-cli -s="$SESSION" eval --raw \
  "() => { const lane = document.querySelector('.lane-label'); if (lane instanceof HTMLElement) lane.click(); return !!lane; }" \
  >/dev/null

pinned_state="$(
  npx --yes --package @playwright/cli playwright-cli -s="$SESSION" eval --raw \
    "async () => { for (let i = 0; i < 20; i += 1) { if (document.querySelector('.tooltip-card.pinned')) return true; await new Promise((resolve) => setTimeout(resolve, 100)); } return false; }"
)"

if [[ "$pinned_state" != "true" ]]; then
  echo "expected pinned tooltip after clicking the first lane label" >&2
  exit 1
fi

npx --yes --package @playwright/cli playwright-cli -s="$SESSION" eval --raw \
  "() => { const buttons=[...document.querySelectorAll('button')]; const loadAll=buttons.find((b)=>b.textContent?.includes('Load all')); if (loadAll) loadAll.click(); return !!loadAll; }" \
  >/dev/null

after_load="$(
  npx --yes --package @playwright/cli playwright-cli -s="$SESSION" eval --raw \
    "async () => { for (let i = 0; i < 40; i += 1) { const values = [...document.querySelectorAll('strong')].map((el) => el.textContent); if (values.includes('14')) return values.join('|'); await new Promise((resolve) => setTimeout(resolve, 100)); } return [...document.querySelectorAll('strong')].map((el) => el.textContent).join('|'); }"
)"

if [[ "$after_load" != *"|14|"* && "$after_load" != 14\|* && "$after_load" != *\|14 ]]; then
  echo "expected loaded count of 14 after drag-drop + upload + Load all, got: $after_load" >&2
  exit 1
fi

echo "gantt viewer smoke passed"

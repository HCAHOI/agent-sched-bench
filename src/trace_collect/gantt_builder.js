/*
 * gantt_builder.js — JavaScript mirror of src/trace_collect/gantt_data.py.
 *
 * Used by the browser-side Gantt viewer to parse JSONL traces the user
 * drops onto the page and build exactly the same payload shape that the
 * Python CLI generates. The Python side remains canonical — this file
 * must stay in sync with gantt_data.py. The parity test at
 * tests/test_gantt_builder_parity.py guards against drift by running
 * both builders against a synthetic v4 trace and asserting deep-equal
 * output.
 *
 * Loaded two ways:
 *   1) From the browser: gantt_serve.py splices this file into
 *      gantt_template.html at build time by replacing the placeholder
 *      ``__GANTT_BUILDER_JS__``. Exposes ``window.GanttBuilder``.
 *   2) From node (tests): required via ``require('./gantt_builder.js')``
 *      which returns the same namespace via module.exports.
 */

(function (root, factory) {
  if (typeof module === 'object' && module.exports) {
    module.exports = factory();
  } else {
    root.GanttBuilder = factory();
  }
}(typeof self !== 'undefined' ? self : this, function () {
  'use strict';

  // ─── Constants ─────────────────────────────────────────────────
  // Mirror of gantt_data.py:_MARKER_CATEGORIES
  // Phase 5 of trace-sim-vastai-pipeline plan: MCP added (paired with
  // gantt_data.py:21).
  const MARKER_CATEGORIES = new Set(['SCHEDULING', 'SESSION', 'CONTEXT', 'MCP']);

  // Mirror of gantt_data.py:ACTION_TYPE_MAP
  // Phase 5: mcp_call → mcp added (paired with gantt_data.py:48-52).
  const ACTION_TYPE_MAP = {
    llm_call: 'llm',
    tool_exec: 'tool',
    mcp_call: 'mcp',
  };

  // Mirror of DEFAULT_SPAN_REGISTRY / DEFAULT_MARKER_REGISTRY
  // Phase 5: mcp span entry added (paired with gantt_data.py:55-60).
  const DEFAULT_SPAN_REGISTRY = {
    llm:        { color: '#00E5FF', label: 'LLM Call',   order: 0 },
    tool:       { color: '#FF6D00', label: 'Tool Exec',  order: 1 },
    scheduling: { color: '#76FF03', label: 'Scheduling', order: 2 },
    mcp:        { color: '#AB47BC', label: 'MCP Call',   order: 3 },
  };

  const DEFAULT_MARKER_REGISTRY = {
    message_dispatch:     { symbol: 'diamond', color: '#76FF03' },
    session_lock_acquire: { symbol: 'diamond', color: '#76FF03' },
    session_load:         { symbol: 'dot',     color: '#76FF03' },
    message_list_build:   { symbol: 'dot',     color: '#4FC3F7' },
    session_turn_save:    { symbol: 'dot',     color: '#76FF03' },
    task_complete:        { symbol: 'flag',    color: '#FF6D00' },
    llm_error:            { symbol: 'cross',   color: '#FF1744' },
    max_iterations:       { symbol: 'cross',   color: '#FF1744' },
    _default:             { symbol: 'dot',     color: '#6b7280' },
  };

  const LLM_CONTENT_MAX = 1000;
  const TOOL_ARGS_MAX = 200;
  const TOOL_PRIMARY_FIELDS = [
    'path', 'file_path', 'filepath',
    'command', 'cmd',
    'pattern', 'query', 'url',
  ];

  // ─── parseJsonl ────────────────────────────────────────────────
  /**
   * Parse a JSONL trace file into typed buckets. Malformed lines are
   * skipped, matching TraceData.load() in trace_inspector.py.
   *
   * Returns: {metadata, actions, events, summaries, agents}
   *   - metadata: the trace_metadata record (or {})
   *   - actions: sorted by (iteration, ts_start) — same key as Python
   *   - events: sorted by ts
   *   - summaries: list of summary records
   *   - agents: ordered list of unique agent_ids, first-seen order
   */
  function parseJsonl(text) {
    let metadata = {};
    const actions = [];
    const events = [];
    const summaries = [];
    const agentSet = new Set();
    const agents = [];

    const lines = text.split(/\r?\n/);
    for (const line of lines) {
      const trimmed = line.trim();
      if (!trimmed) continue;
      let rec;
      try {
        rec = JSON.parse(trimmed);
      } catch (_e) {
        continue;  // skip malformed
      }
      const t = rec && rec.type;
      if (t === 'trace_metadata') {
        metadata = rec;
      } else if (t === 'action') {
        actions.push(rec);
        const aid = rec.agent_id;
        if (aid && !agentSet.has(aid)) { agentSet.add(aid); agents.push(aid); }
      } else if (t === 'event') {
        events.push(rec);
        const aid = rec.agent_id;
        if (aid && !agentSet.has(aid)) { agentSet.add(aid); agents.push(aid); }
      } else if (t === 'summary') {
        summaries.push(rec);
      }
    }

    actions.sort((a, b) => {
      const ai = (a.iteration || 0) - (b.iteration || 0);
      if (ai !== 0) return ai;
      return (a.ts_start || 0) - (b.ts_start || 0);
    });
    events.sort((a, b) => (a.ts || 0) - (b.ts || 0));

    // Strict v5 version check — v4 support dropped in the SWE-rebench
    // plugin refactor. No backfill, no tolerance. Mirror of the Python
    // check in trace_inspector.py::TraceData.load.
    const version = metadata && metadata.trace_format_version;
    if (version !== 5) {
      throw new Error(
        'Unsupported trace_format_version ' + JSON.stringify(version) +
        ': expected 5. v4 support was dropped during the SWE-rebench ' +
        'plugin refactor; regenerate the trace via the current collector ' +
        'to produce a v5 trace.'
      );
    }

    return { metadata, actions, events, summaries, agents };
  }

  // ─── Detail extractors ──────────────────────────────────────────

  /**
   * Mirror of gantt_data.py:_summarize_tool_call. Prefers
   * ``tool_name(primary="value")`` rendering when the args decode as a
   * JSON dict with a known primary field; otherwise falls back to a
   * 200-char raw args preview.
   */
  function summarizeToolCall(tc) {
    if (tc == null || typeof tc !== 'object') return null;
    const fn = (tc.function && typeof tc.function === 'object') ? tc.function : {};
    const name = fn.name || tc.name || '?';

    let rawArgs = fn.arguments;
    if (rawArgs == null) rawArgs = tc.arguments;
    if (rawArgs == null) rawArgs = '';

    let parsed = null;
    if (rawArgs != null && typeof rawArgs === 'object' && !Array.isArray(rawArgs)) {
      parsed = rawArgs;
    } else if (typeof rawArgs === 'string') {
      try {
        const candidate = JSON.parse(rawArgs);
        if (candidate && typeof candidate === 'object' && !Array.isArray(candidate)) {
          parsed = candidate;
        }
      } catch (_e) {
        parsed = null;
      }
    }

    if (parsed !== null) {
      for (const field of TOOL_PRIMARY_FIELDS) {
        if (field in parsed && parsed[field] !== null && parsed[field] !== undefined) {
          let value = String(parsed[field]);
          if (value.length > TOOL_ARGS_MAX) value = value.slice(0, TOOL_ARGS_MAX) + '...';
          return `${name}(${field}="${value}")`;
        }
      }
    }

    // Fallback: raw-args preview
    let argsStr;
    if (rawArgs != null && typeof rawArgs === 'object') {
      argsStr = JSON.stringify(rawArgs);
    } else {
      argsStr = String(rawArgs);
    }
    let preview = argsStr.slice(0, TOOL_ARGS_MAX);
    if (argsStr.length > TOOL_ARGS_MAX) preview += '...';
    return `${name}(${preview})`;
  }

  /** Mirror of gantt_data.py:_extract_detail_from_action */
  function extractDetailFromAction(act) {
    const data = Object.assign({}, act.data || {});

    const rawResp = data.raw_response;
    delete data.raw_response;

    if (rawResp && typeof rawResp === 'object') {
      const choices = Array.isArray(rawResp.choices) ? rawResp.choices : [];
      if (choices.length > 0) {
        const msg = (choices[0].message && typeof choices[0].message === 'object')
          ? choices[0].message : {};
        const content = msg.content || '';
        if (content) {
          if (content.length > LLM_CONTENT_MAX) {
            data.llm_content = content.slice(0, LLM_CONTENT_MAX) + '...';
          } else {
            data.llm_content = content;
          }
        }
        const toolCalls = Array.isArray(msg.tool_calls) ? msg.tool_calls : [];
        if (toolCalls.length > 0) {
          const summaries = toolCalls
            .map(summarizeToolCall)
            .filter(s => s !== null);
          if (summaries.length > 0) {
            data.tool_calls_requested = summaries;
          }
        }
      }
    }

    delete data.messages_in;
    for (const key of ['tool_result', 'tool_args', 'args_preview', 'result_preview']) {
      if (typeof data[key] === 'string' && data[key].length > 100) {
        data[key] = data[key].slice(0, 100) + '...';
      }
    }
    return data;
  }

  /** Mirror of gantt_data.py:_extract_detail_from_event */
  function extractDetailFromEvent(ev) {
    const data = Object.assign({}, ev.data || {});
    delete data.messages_in;

    const rawResp = data.raw_response;
    delete data.raw_response;
    if (rawResp && typeof rawResp === 'object') {
      const choices = Array.isArray(rawResp.choices) ? rawResp.choices : [];
      if (choices.length > 0) {
        const msg = (choices[0].message && typeof choices[0].message === 'object')
          ? choices[0].message : {};
        const content = msg.content || '';
        if (content) {
          data.llm_content = content.slice(0, 200) + (content.length > 200 ? '...' : '');
        }
      }
    }

    for (const key of ['args_preview', 'result_preview', 'tool_args', 'tool_result']) {
      if (typeof data[key] === 'string' && data[key].length > 100) {
        data[key] = data[key].slice(0, 100) + '...';
      }
    }
    return data;
  }

  // ─── Spans + markers ───────────────────────────────────────────

  function roundTo(x, decimals) {
    const f = Math.pow(10, decimals);
    return Math.round(x * f) / f;
  }

  function buildSpansAndMarkers(actions, events, t0) {
    const spans = [];
    const markers = [];

    for (const act of actions) {
      const atype = act.action_type;
      if (!(atype in ACTION_TYPE_MAP)) continue;
      const spanType = ACTION_TYPE_MAP[atype];
      spans.push({
        type: spanType,
        start: (act.ts_start || 0) - t0,
        end: (act.ts_end || 0) - t0,
        start_abs: act.ts_start || 0,
        end_abs: act.ts_end || 0,
        iteration: act.iteration || 0,
        detail: extractDetailFromAction(act),
      });
    }

    // Scheduling spans — event-gated only, no duration threshold.
    if (spans.length > 0 && events.length > 0) {
      const sortedSpans = spans.slice().sort((a, b) => a.start_abs - b.start_abs);
      for (let i = 0; i < sortedSpans.length - 1; i++) {
        const gapStart = sortedSpans[i].end_abs;
        const gapEnd = sortedSpans[i + 1].start_abs;
        if (gapEnd <= gapStart) continue;
        const evsInGap = events.filter(e =>
          MARKER_CATEGORIES.has(e.category) &&
          gapStart < (e.ts || 0) && (e.ts || 0) < gapEnd
        );
        if (evsInGap.length === 0) continue;
        spans.push({
          type: 'scheduling',
          start: gapStart - t0,
          end: gapEnd - t0,
          start_abs: gapStart,
          end_abs: gapEnd,
          iteration: sortedSpans[i + 1].iteration || 0,
          detail: {
            gap_ms: roundTo((gapEnd - gapStart) * 1000, 1),
            events: evsInGap.map(e => e.event || '?'),
          },
        });
      }
    }

    for (const ev of events) {
      const category = ev.category || '';
      if (MARKER_CATEGORIES.has(category)) {
        markers.push({
          type: category.toLowerCase(),
          event: ev.event || 'unknown',
          t: (ev.ts || 0) - t0,
          t_abs: ev.ts || 0,
          iteration: ev.iteration || 0,
          detail: extractDetailFromEvent(ev),
        });
      }
    }

    spans.sort((a, b) => a.start - b.start);
    markers.sort((a, b) => a.t - b.t);
    return { spans, markers };
  }

  // ─── t0 / elapsed ──────────────────────────────────────────────

  function computeT0(parsed) {
    let t0 = Infinity;
    for (const ev of parsed.events) {
      const ts = ev.ts || 0;
      if (ts && ts < t0) t0 = ts;
    }
    for (const act of parsed.actions) {
      const ts = act.ts_start || 0;
      if (ts && ts < t0) t0 = ts;
    }
    return Number.isFinite(t0) ? t0 : 0.0;
  }

  function getElapsed(parsed) {
    for (const s of parsed.summaries) {
      if ('elapsed_s' in s) return s.elapsed_s;
    }
    return null;
  }

  // ─── Payload builders ──────────────────────────────────────────

  function buildPayload(parsed, label) {
    const meta = parsed.metadata || {};
    const scaffold = meta.scaffold || 'unknown';
    const instanceId = meta.instance_id || '';
    const traceId = label || `${scaffold}/${instanceId}` || 'trace';

    const t0 = computeT0(parsed);

    let agents = parsed.agents.slice();
    if (agents.length === 0) agents = ['default'];

    const lanes = agents.map(agentId => {
      const agentEvents = parsed.events.filter(e => e.agent_id === agentId);
      const agentActions = parsed.actions.filter(a => a.agent_id === agentId);
      const { spans, markers } = buildSpansAndMarkers(agentActions, agentEvents, t0);
      return { agent_id: agentId, spans, markers };
    });

    const distinctIters = new Set(parsed.actions.map(a => a.iteration || 0));

    return {
      id: traceId,
      metadata: {
        scaffold,
        model: meta.model || null,
        instance_id: instanceId,
        mode: meta.mode || null,
        max_iterations: meta.max_iterations || meta.max_steps || null,
        n_actions: parsed.actions.length,
        n_iterations: distinctIters.size,
        n_events: parsed.events.length,
        elapsed_s: getElapsed(parsed),
      },
      t0,
      lanes,
    };
  }

  function buildPayloadMulti(labeledParsedList, opts) {
    opts = opts || {};
    const spanRegistry = opts.spanRegistry || DEFAULT_SPAN_REGISTRY;
    const markerRegistry = opts.markerRegistry || DEFAULT_MARKER_REGISTRY;
    return {
      registries: { spans: spanRegistry, markers: markerRegistry },
      traces: labeledParsedList.map(({ label, parsed }) => buildPayload(parsed, label)),
    };
  }

  return {
    // Constants (for tests and extension)
    MARKER_CATEGORIES,
    ACTION_TYPE_MAP,
    DEFAULT_SPAN_REGISTRY,
    DEFAULT_MARKER_REGISTRY,
    LLM_CONTENT_MAX,
    TOOL_ARGS_MAX,
    TOOL_PRIMARY_FIELDS,
    // Public API
    parseJsonl,
    summarizeToolCall,
    extractDetailFromAction,
    extractDetailFromEvent,
    buildSpansAndMarkers,
    computeT0,
    getElapsed,
    buildPayload,
    buildPayloadMulti,
  };
}));

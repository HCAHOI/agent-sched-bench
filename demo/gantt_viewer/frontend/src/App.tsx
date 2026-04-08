import { Match, Show, Switch, createMemo, onMount } from "solid-js";

import {
  getPayload,
  getTraces,
  uploadTrace,
  type GanttPayload,
  type TraceDescriptor,
  type TracePayload,
} from "./api/client";
import CanvasStage from "./components/CanvasStage";
import DropZone from "./components/DropZone";
import Header from "./components/Header";
import Legend from "./components/Legend";
import Sidebar from "./components/Sidebar";
import Tooltip from "./components/Tooltip";
import TraceChipBar from "./components/TraceChipBar";
import { sameHit } from "./canvas/hit";
import { enablePersistence } from "./state/persist";
import {
  appError,
  descriptors,
  hoverCard,
  loadedTraces,
  loadingIds,
  pinnedCard,
  registries,
  scrollTop,
  setAppError,
  setDescriptors,
  setHoverCard,
  setLoadedTraces,
  setLoadingIds,
  setPinnedCard,
  setRegistries,
  setScrollTop,
  setTimeMode,
  setViewMode,
  setVisibility,
  setZoom,
  timeMode,
  viewMode,
  visibility,
  zoom,
} from "./state/signals";

function mergeTraces(existing: TracePayload[], incoming: TracePayload[]): TracePayload[] {
  const byId = new Map(existing.map((trace) => [trace.id, trace]));
  incoming.forEach((trace) => byId.set(trace.id, trace));
  return [...byId.values()];
}

function mergeDescriptors(existing: TraceDescriptor[], incoming: TraceDescriptor[]): TraceDescriptor[] {
  const byId = new Map(existing.map((trace) => [trace.id, trace]));
  incoming.forEach((trace) => byId.set(trace.id, trace));
  return [...byId.values()].sort((left, right) => left.id.localeCompare(right.id));
}

export default function App() {
  enablePersistence();

  const payload = createMemo<GanttPayload | null>(() =>
    registries()
      ? {
          registries: registries()!,
          traces: loadedTraces(),
        }
      : null,
  );

  const loadedIds = createMemo(() => loadedTraces().map((trace) => trace.id));
  const headerSummary = createMemo(() => {
    const traces = loadedTraces();
    if (traces.length === 0) {
      return "No traces loaded yet.";
    }
    return traces
      .map((trace) => {
        const metadata = trace.metadata;
        return `${trace.label}: ${metadata.scaffold} ${metadata.model ?? ""}`.trim() +
          ` (${metadata.n_actions} actions / ${metadata.n_iterations} iters, ${(metadata.elapsed_s ?? 0).toFixed(1)}s)`;
      })
      .join(" · ");
  });

  async function initialize(): Promise<void> {
    try {
      const traceList = await getTraces();
      setDescriptors(traceList.traces);
      setRegistries(traceList.registries);
      const firstV5 = traceList.traces.find((trace) => trace.source_format === "v5");
      if (firstV5) {
        await loadTraceIds([firstV5.id]);
      }
    } catch (error) {
      setAppError(String(error));
    }
  }

  async function loadTraceIds(ids: string[]): Promise<void> {
    const currentlyLoaded = new Set(loadedIds());
    const inFlight = loadingIds();
    const nextIds = ids.filter((id) => !currentlyLoaded.has(id) && !inFlight[id]);
    if (nextIds.length === 0) {
      return;
    }

    setLoadingIds((current) => ({
      ...current,
      ...Object.fromEntries(nextIds.map((id) => [id, true])),
    }));

    try {
      const nextPayload = await getPayload(nextIds);
      setRegistries(nextPayload.registries);
      setLoadedTraces((current) => mergeTraces(current, nextPayload.traces));
      setVisibility((current) => ({
        ...current,
        ...Object.fromEntries(nextPayload.traces.map((trace) => [trace.id, true])),
      }));
      setAppError(null);
    } catch (error) {
      setAppError(String(error));
    } finally {
      setLoadingIds((current) => {
        const next = { ...current };
        nextIds.forEach((id) => delete next[id]);
        return next;
      });
    }
  }

  function removeTrace(id: string): void {
    setLoadedTraces((current) => current.filter((trace) => trace.id !== id));
    setVisibility((current) => {
      const next = { ...current };
      delete next[id];
      return next;
    });
    if (pinnedCard()?.hit.traceId === id) {
      setPinnedCard(null);
    }
    if (hoverCard()?.hit.traceId === id) {
      setHoverCard(null);
    }
  }

  function toggleVisibility(id: string): void {
    setVisibility((current) => ({
      ...current,
      [id]: current[id] === false,
    }));
  }

  async function handleUpload(files: File[]): Promise<void> {
    for (const file of files) {
      try {
        const response = await uploadTrace(file);
        setDescriptors((current) => mergeDescriptors(current, [response.descriptor]));
        setLoadedTraces((current) => mergeTraces(current, [response.payload_fragment]));
        setVisibility((current) => ({
          ...current,
          [response.descriptor.id]: true,
        }));
        setAppError(null);
      } catch (error) {
        setAppError(String(error));
      }
    }
  }

  onMount(() => {
    void initialize();
    document.addEventListener("keydown", (event) => {
      if (event.key === "Escape") {
        setPinnedCard(null);
      }
    });
  });

  return (
    <main class="app-shell">
      <Header
        loadedCount={() => loadedTraces().length}
        summary={headerSummary}
        onTimeModeChange={setTimeMode}
        onViewModeChange={setViewMode}
        timeMode={timeMode}
        viewMode={viewMode}
        zoom={zoom}
      />

      <TraceChipBar
        descriptors={descriptors()}
        loadedIds={loadedIds()}
        loadingIds={loadingIds()}
        onLoad={loadTraceIds}
        onUpload={handleUpload}
        onRemove={removeTrace}
        onToggleVisibility={toggleVisibility}
        visibility={visibility()}
      />

      <Show when={appError()}>
        <section class="toolbar-card error-banner">{appError()}</section>
      </Show>

      <section class="workspace-card">
        <Sidebar
          onPinLane={(card) => setPinnedCard(card)}
          scrollTop={scrollTop()}
          traces={loadedTraces()}
          viewMode={viewMode()}
          visibility={visibility()}
        />
        <CanvasStage
          onClick={(card) =>
            setPinnedCard((current) => {
              if (!card) {
                return null;
              }
              return current && sameHit(current.hit, card.hit) ? null : card;
            })
          }
          onHover={(card) => {
            if (!pinnedCard()) {
              setHoverCard(card);
            }
          }}
          onPinnedReanchor={(card) =>
            setPinnedCard((current) => {
              if (!current || current.hit.kind === "lane") {
                return current;
              }
              return card;
            })
          }
          onScroll={setScrollTop}
          onZoom={setZoom}
          payload={payload()}
          pinnedHit={pinnedCard()?.hit ?? null}
          timeMode={timeMode()}
          viewMode={viewMode()}
          visibility={visibility()}
          zoom={zoom()}
        />
      </section>

      <Legend registries={registries()} />
      <DropZone onUpload={handleUpload} />

      <Switch>
        <Match when={pinnedCard()}>
          <Tooltip
            card={pinnedCard()}
            onClose={() => setPinnedCard(null)}
            pinned={true}
            registries={registries()}
          />
        </Match>
        <Match when={hoverCard()}>
          <Tooltip
            card={hoverCard()}
            onClose={() => setHoverCard(null)}
            pinned={false}
            registries={registries()}
          />
        </Match>
      </Switch>
    </main>
  );
}

import { For, Show, createMemo } from "solid-js";

import type { TraceDescriptor } from "../api/client";

interface TraceChipBarProps {
  descriptors: TraceDescriptor[];
  loadedIds: string[];
  loadingIds: Record<string, boolean>;
  onLoad: (ids: string[]) => Promise<void> | void;
  onUpload: (files: File[]) => Promise<void> | void;
  onRemove: (id: string) => void;
  onToggleVisibility: (id: string) => void;
  visibility: Record<string, boolean>;
}

export default function TraceChipBar(props: TraceChipBarProps) {
  const loadedSet = createMemo(() => new Set(props.loadedIds));
  let fileInputEl!: HTMLInputElement;

  return (
    <section class="toolbar-card trace-strip">
      <div class="trace-strip-header">
        <div>
          <span class="metric-label">traces</span>
          <strong>Load individual traces or pull all discovered entries.</strong>
        </div>
        <div class="trace-actions">
          <button
            class="secondary-btn"
            onClick={() => fileInputEl.click()}
            type="button"
          >
            Add JSONL
          </button>
          <button
            class="primary-btn"
            onClick={() => props.onLoad(props.descriptors.map((descriptor) => descriptor.id))}
            type="button"
          >
            Load all
          </button>
        </div>
      </div>

      <input
        accept=".jsonl"
        class="visually-hidden"
        multiple
        onChange={(event) => {
          const files = Array.from(event.currentTarget.files ?? []);
          if (files.length > 0) {
            void props.onUpload(files);
          }
          event.currentTarget.value = "";
        }}
        ref={fileInputEl}
        type="file"
      />

      <div class="trace-chip-grid">
        <For each={props.descriptors}>
          {(descriptor) => {
            const loaded = () => loadedSet().has(descriptor.id);
            const visible = () => props.visibility[descriptor.id] !== false;
            return (
              <div
                class="trace-chip"
                classList={{
                  hidden: loaded() && !visible(),
                  loading: props.loadingIds[descriptor.id] === true,
                  loaded: loaded(),
                }}
              >
                <button
                  class="trace-chip-main"
                  onClick={() =>
                    loaded()
                      ? props.onToggleVisibility(descriptor.id)
                      : props.onLoad([descriptor.id])
                  }
                  type="button"
                >
                  <span>{descriptor.label}</span>
                  <small>{descriptor.source_format}</small>
                </button>
                <Show when={loaded()}>
                  <button
                    class="trace-remove"
                    onClick={() => props.onRemove(descriptor.id)}
                    type="button"
                  >
                    ×
                  </button>
                </Show>
              </div>
            );
          }}
        </For>
      </div>
    </section>
  );
}

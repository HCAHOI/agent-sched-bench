import { For, Show } from "solid-js";

import type { Registries } from "../api/client";

interface LegendProps {
  registries: Registries | null;
}

export default function Legend(props: LegendProps) {
  return (
    <section class="toolbar-card legend-card">
      <span class="metric-label">legend</span>
      <div class="legend-row">
        <Show when={props.registries}>
          <For each={Object.entries(props.registries?.spans ?? {})}>
            {([key, value]) => (
              <div class="legend-item" title={key}>
                <span class="legend-swatch" style={{ background: value.color }} />
                <span>{value.label}</span>
              </div>
            )}
          </For>
        </Show>
      </div>
    </section>
  );
}

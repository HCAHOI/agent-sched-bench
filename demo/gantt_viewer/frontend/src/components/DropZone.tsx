import { Show, createSignal, onCleanup, onMount } from "solid-js";

interface DropZoneProps {
  onUpload: (files: File[]) => Promise<void>;
}

export default function DropZone(props: DropZoneProps) {
  const [active, setActive] = createSignal(false);
  let dragDepth = 0;

  async function onDrop(event: DragEvent) {
    event.preventDefault();
    dragDepth = 0;
    setActive(false);
    const files = Array.from(event.dataTransfer?.files ?? []).filter((file) =>
      file.name.toLowerCase().endsWith(".jsonl"),
    );
    if (files.length > 0) {
      await props.onUpload(files);
    }
  }

  function onDragEnter(event: DragEvent) {
    event.preventDefault();
    dragDepth += 1;
    setActive(true);
  }

  function onDragLeave(event: DragEvent) {
    event.preventDefault();
    dragDepth = Math.max(0, dragDepth - 1);
    if (dragDepth === 0) {
      setActive(false);
    }
  }

  function onDragOver(event: DragEvent) {
    event.preventDefault();
    setActive(true);
  }

  onMount(() => {
    window.addEventListener("dragenter", onDragEnter);
    window.addEventListener("dragleave", onDragLeave);
    window.addEventListener("dragover", onDragOver);
    window.addEventListener("drop", onDrop);
  });

  onCleanup(() => {
    window.removeEventListener("dragenter", onDragEnter);
    window.removeEventListener("dragleave", onDragLeave);
    window.removeEventListener("dragover", onDragOver);
    window.removeEventListener("drop", onDrop);
  });

  return (
    <Show when={active()}>
      <div class="dropzone-overlay">
        <div class="dropzone-card">
          <p class="eyebrow">Drop JSONL</p>
          <strong>Release to upload and render the trace immediately.</strong>
        </div>
      </div>
    </Show>
  );
}

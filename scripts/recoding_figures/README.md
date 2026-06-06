# Recording Figure Scripts

These scripts read existing `attempt_*/recordings` artifacts and generate the
first three method-driving figures:

- Plot 1: pairwise iteration distance matrices for attention and MoE.
- Plot 2: layer specialization maps for attention roles and MoE experts.
- Plot 3: attention-vs-MoE layer specialization scatter.

Example:

```bash
python scripts/recoding_figures/make_figures.py \
  traces/terminal-bench/Qwen_Qwen3-Coder-30B-A3B-Instruct/<run>/<task>/attempt_1 \
  --output-dir docs/recording_figures/<run>/<task>
```

The directory name intentionally follows the user-requested spelling
`recoding_figures`.

## BlockTopK Absolute-Position Figures

`plot_head_span_grid.py --mode block_position` renders the selected middle
blocks from the block_topk sparse-attention recorder on the absolute KV token
axis. The static output is a PNG thumbnail plus CSV/JSON summaries; this mode no
longer writes PDFs because the full block axis is meant to be inspected with the
interactive HTML renderer.

Recommended order:

```bash
python scripts/recoding_figures/plot_head_span_grid.py \
  /path/to/task/attempt_1 \
  --mode block_position \
  --layers 0,9,18,27,38,47 \
  --output-dir docs/recording_figures/block_position_YYYYMMDD

python scripts/recoding_figures/detokenize_selected_blocks.py \
  /path/to/task/attempt_1 \
  --layers 0,9,18,27,38,47 \
  --output-dir docs/recording_figures/block_position_YYYYMMDD

python scripts/recoding_figures/plot_block_position_html.py \
  /path/to/task/attempt_1 \
  --layers 0,9,18,27,38,47 \
  --output-dir docs/recording_figures/block_position_YYYYMMDD \
  --workers 4
```

The HTML step writes `block_position.html` in each task directory. It is a
self-contained Canvas viewer with two synchronized heatmaps, layer/head
selectors, box zoom, pan, call/token range controls, segment focus, and
de-tokenized selected-block text in hover/details panels when
`selected_blocks_detok.csv` is present. Head selection affects the attention
panel only; selection frequency is head-independent. Missing cells are gray; the
dark orange band marks sink tokens, and the dark green staircase marks the
per-call recent window.

The HTML builder defaults to one worker to avoid accidental memory spikes on
large multi-task runs. Use `--workers N` explicitly when each task writes to its
own output directory and the node has enough memory; `--workers 0` auto-selects
at most four workers.

For private Hugging Face recordings on a temporary node, keep tokens in
environment variables or transient cache files only. Do not write secrets or
dataset-specific prefixes into repository scripts.

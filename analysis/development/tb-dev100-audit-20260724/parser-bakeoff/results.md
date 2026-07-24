# Terminal-Bench dev-100 Bash parser bake-off

Date: 2026-07-24

## Recommendation

Use **mvdan.cc/sh/v3 v3.13.1 as the primary structure/bin-feature
parser**. Use **tree-sitter-bash 0.25.1 partial recovery only when the primary
parser rejects malformed input**, and retain an explicit partial/error flag.
Do not treat a recovered CST as a strict parse.

Mvdan and brush are tied on correctness in this bake-off: both parse all
754/754 Bash-valid unique commands (775/775 rows), score 25/25 on the generic
contract, and have zero head or construct disagreements with each other.
Mvdan is the better primary because its typed AST needs 314 adapter lines
versus 490 for brush's serialized-AST traversal, and its measured hot parse
median is 14.46 us versus 65.39 us. Brush remains a credible alternative if a
Rust integration is already available.

Tree-sitter is the best malformed fallback because the sole Bash-invalid
command was its sole partial parse: strict=false, partial=true, with recovered
heads and spans. It is not the primary because it misses the generic
same-statement multi-heredoc case and disagrees with mvdan/brush on seven
sequence flags, one pipeline flag, one heredoc count, and 22 unique head sets.

Native bashlex is not suitable here. It strictly parses only 431/776 rows
(55.5%) and 413/755 unique commands (54.7%), with zero recovery of the quoted
heredoc bodies that dominate this corpus. The earlier
bashlex-plus-heredoc-replacement workaround reached a reported 98.5% parse
coverage, but replacement destroys native heredoc structure and still leaves
arithmetic unsupported. Mvdan/brush reach 99.87% overall and 100% on
Bash-valid input without that workaround.

Prefix-trie keys still use the existing `shlex` normalization. This bake-off
does **not** claim to fix prefix transfer.

## Boundary and protocol

- Only the 100 task IDs in
  `analysis/development/tb-dev100-audit-20260724/tb-dev100-split.json` were
  opened, by constructing
  `traces/terminal-bench/tb-all/canonical/<allowlisted-id>/trace.jsonl`
  directly. The 139-task confirmation set was neither enumerated nor opened.
- Corpus: 776 exec rows, 755 exact-unique command strings, and 21 duplicate
  rows. Every metric below reports unique-command and row counts.
- `bash -n` is a syntax-validity reference only: 754/755 unique commands and
  775/776 rows are valid under GNU Bash 5.1.16 with `LC_ALL=C.UTF-8`.
- Strict tree-sitter success requires zero `ERROR` and zero `MISSING` nodes.
  Its one error-containing CST is reported only as partial recovery.
- Heredoc body recovery is exact raw-byte-span plus quoted/`<<-` flag equality
  against an independent quote-aware scanner. Two incomplete references are
  reported and excluded rather than counted as failures.
- “Coverage” below uses the union of parser-positive detections as its
  denominator; it measures cross-parser agreement, not semantic truth. The
  25 generic cases provide expected-feature truth.
- `U` means exact-unique commands; `R` means rows.

The complete machine-readable output, including every row/unique denominator,
pairwise matrix, focused slice, error class, timing sample, and ten sampled
commands, is in [results.json](results.json).

## Primary outcomes

| Parser | Strict overall U | Strict overall R | Strict on Bash-valid U | Strict on Bash-valid R | Partial U / R | Exact complete heredocs U / R | Generic |
|---|---:|---:|---:|---:|---:|---:|---:|
| bashlex | 413/755 (54.7%) | 431/776 (55.5%) | 413/754 (54.8%) | 431/775 (55.6%) | 0 / 0 | 0/326 / 0/329 | 20/25 |
| mvdan | 754/755 (99.87%) | 775/776 (99.87%) | 754/754 (100%) | 775/775 (100%) | 0 / 0 | 325/326 / 328/329 | 25/25 |
| brush | 754/755 (99.87%) | 775/776 (99.87%) | 754/754 (100%) | 775/775 (100%) | 0 / 0 | 325/326 / 328/329 | 25/25 |
| tree-sitter | 754/755 (99.87%) | 775/776 (99.87%) | 754/754 (100%) | 775/775 (100%) | 1 / 1 | 325/326 / 328/329 | 24/25 |

The one overall strict failure is the same genuinely malformed command for
mvdan, brush, and tree-sitter. It is also the only shortfall in the
parser-independent complete-heredoc denominator.

## Construct coverage

Each cell is detected/union-positive as `U; R`.

| Construct | bashlex | mvdan | brush | tree-sitter |
|---|---:|---:|---:|---:|
| Sequence | 336/568; 354/589 | 567/568; 588/589 | 567/568; 588/589 | 562/568; 583/589 |
| Pipeline | 158/227; 160/229 | 225/227; 227/229 | 225/227; 227/229 | 226/227; 228/229 |
| Background | 6/9; 6/9 | 9/9; 9/9 | 9/9; 9/9 | 9/9; 9/9 |
| Loop | 30/45; 30/45 | 45/45; 45/45 | 45/45; 45/45 | 45/45; 45/45 |
| C-style `for` | 0/0; 0/0 | 0/0; 0/0 | 0/0; 0/0 | 0/0; 0/0 |
| Subshell | 7/13; 7/13 | 13/13; 13/13 | 13/13; 13/13 | 13/13; 13/13 |
| Command substitution | 42/73; 42/76 | 73/73; 76/76 | 73/73; 76/76 | 73/73; 76/76 |
| Nested substitution | 0/0; 0/0 | 0/0; 0/0 | 0/0; 0/0 | 0/0; 0/0 |
| Heredoc | 0/337; 0/340 | 336/337; 339/340 | 336/337; 339/340 | 337/337; 340/340 |
| Arithmetic expansion | 0/7; 0/7 | 7/7; 7/7 | 7/7; 7/7 | 7/7; 7/7 |

There are no dev-100 C-style loops or nested substitutions, so those claims
come only from the generic suite. Mvdan and brush have identical detection on
every construct and every command.

## Pairwise disagreements

Strict/head cells are counts as `U/R`.

| Pair | Strict verdict | Ordered head list |
|---|---:|---:|
| bashlex vs mvdan | 341/344 | 341/344 |
| bashlex vs brush | 341/344 | 341/344 |
| bashlex vs tree-sitter | 341/344 | 350/353 |
| mvdan vs brush | 0/0 | 0/0 |
| mvdan vs tree-sitter | 0/0 | 22/22 |
| brush vs tree-sitter | 0/0 | 22/22 |

Nonzero construct disagreements, also `U/R`:

| Pair | Sequence | Pipeline | Background | Loop | Subshell | Command subst. | Heredoc | Arithmetic |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| bashlex vs mvdan | 231/234 | 69/69 | 3/3 | 15/15 | 6/6 | 31/34 | 336/339 | 7/7 |
| bashlex vs brush | 231/234 | 69/69 | 3/3 | 15/15 | 6/6 | 31/34 | 336/339 | 7/7 |
| bashlex vs tree-sitter | 232/235 | 70/70 | 3/3 | 15/15 | 6/6 | 31/34 | 337/340 | 7/7 |
| mvdan vs brush | 0/0 | 0/0 | 0/0 | 0/0 | 0/0 | 0/0 | 0/0 | 0/0 |
| mvdan vs tree-sitter | 7/7 | 1/1 | 0/0 | 0/0 | 0/0 | 0/0 | 1/1 | 0/0 |
| brush vs tree-sitter | 7/7 | 1/1 | 0/0 | 0/0 | 0/0 | 0/0 | 1/1 | 0/0 |

## Focused slices

| Slice | Size U / R | bashlex strict U / R | mvdan strict U / R | brush strict U / R | tree strict U / R |
|---|---:|---:|---:|---:|---:|
| Quoted heredoc | 328 / 331 | 0/328 / 0/331 | 327/328 / 330/331 | 327/328 / 330/331 | 327/328 / 330/331 |
| Multiple heredocs | 23 / 23 | 0/23 / 0/23 | 23/23 / 23/23 | 23/23 / 23/23 | 23/23 / 23/23 |
| Arithmetic | 9 / 9 | 0/9 / 0/9 | 9/9 / 9/9 | 9/9 / 9/9 | 9/9 / 9/9 |
| `<<-` | 0 / 0 | — | — | — | — |
| Nested substitution | 0 / 0 | — | — | — | — |

For exact body boundaries, the quoted-heredoc complete denominator is 326 U /
329 R: bashlex recovers 0/326 U and 0/329 R; each other parser recovers 325/326
U and 328/329 R. Two incomplete references are excluded. For multi-heredoc
commands, the complete denominator is 22 U / 22 R: mvdan, brush, and
tree-sitter recover all 22/22, while bashlex recovers none. The generic suite
separately catches tree-sitter's failure on two heredocs attached to one
simple command, plus quoted EOF and `<<-` behavior absent from the dev slice.

## Performance and integration

Performance is one final batch on this host. Cold is the median of seven
process-spawn-plus-init runs; hot is the adapter-reported per-command median.
Peak RSS comes from `wait4` and is tens of MiB (order `10^1` MiB) for all four.

| Parser | Cold ms | Hot us/command | 755-command batch s | Commands/s | Peak RSS KiB | Adapter LoC | Integration |
|---|---:|---:|---:|---:|---:|---:|---|
| bashlex | 154.38 | 442.39 | 0.8140 | 927 | 23,776 | 181 | Python library; in-process is available |
| mvdan | 2.49 | 14.46 | 0.0319 | 23,703 | 25,564 | 314 | Compiled Go subprocess |
| brush | 1.27 | 65.39 | 0.0872 | 8,656 | 29,788 | 490 | Compiled Rust subprocess |
| tree-sitter | 79.57 | 93.68 | 0.3424 | 2,205 | 33,748 | 219 | Isolated Python subprocess |

| Parser | Pinned version | License | Last release checked | Maintenance/integration note |
|---|---|---|---|---|
| bashlex | [0.18](https://pypi.org/project/bashlex/) | GPL-3.0-or-later | 2023-01-18 | Oldest release; already importable in project venv |
| mvdan | [v3.13.1](https://github.com/mvdan/sh/releases/tag/v3.13.1) | BSD-3-Clause | 2026-04-06 | Current typed Go AST; runtime build info verifies the module version |
| brush-parser | [0.4.0](https://docs.rs/brush-parser/0.4.0/brush_parser/) | MIT | 2026-05-03 | Active Rust parser; locked source rebuilt before the run |
| tree-sitter-bash | [0.25.1](https://pypi.org/project/tree-sitter-bash/) | MIT | 2025-12-02 | Active CST grammar with useful error recovery |

No project dependency was added or modified. The pinned Go/Rust sources are
rebuilt by [build_adapters.sh](build_adapters.sh); toolchains and binaries
remain under `/tmp/tb-parser-bakeoff`.

## Representative dev-100 disagreements

The machine-readable result contains ten sampled disagreements. Five distinct
mechanisms are shown here; every sample occurs once at both U and R level.

### 1. Genuinely malformed input

```bash
python3 - <<'PY'
from pathlib import Path
import re
p=Path('exp_data/datasets/tokenized/rw_v2_fasttext_openhermes_vs_rw_v2_bigram_0.1_arcade100k.json')
text=p.read_text()
rx=re.compile(r'\bhf_[A-Za-z0-9]{20,}\b')
ms=rx.findall(text)
if len(ms) != 2 or len(set(ms)) != 1:
    raise SystemExit(f'Expected two copies of one token; got count={len(ms)}, unique={len(set(ms))}')
p.write_text(rx.sub('<your-huggingface-token>', text))
print('Replaced both copies of the embedded Hugging Face token.')
PY
python3 -m json.tool exp_data/datasets/tokenized/rw_v2_fasttext_openhermes_vs_rw_v2_bigram_0.1_arcade100k.json >/dev/null
git status --short
git diff --stat
git diff -- ray_processing/process.py ray_processing/ray_cluster.yaml | sed -E 's/(AWS_ACCESS_KEY_ID[=\" ]+)[^ '\"<]+/\1<REDACTED>/g; s/(AWS_SECRET_ACCESS_KEY[=\" ]+)[^ '\"<]+/\1<REDACTED>/g; s#https://[^@ <]+@github.com#https://<REDACTED>@github.com#g; s/(--token )[A-Za-z0-9_<>-]+/\1<REDACTED>/g'
```

**OBSERVATION:** `bash -n`, bashlex, mvdan, and brush reject it. Tree-sitter is
strict=false/partial=true and recovers coarse heads, including error-region
noise.

**JUDGMENT:** This is the evidence for tree-sitter as a flagged malformed
fallback, not evidence that its recovered structure is fully trustworthy.

### 2. Quoted heredoc

```bash
python3 - <<'PY'
import inspect
import mteb.cli
print(inspect.getsource(mteb.cli))
PY
```

**OBSERVATION:** Bash accepts it. Bashlex rejects it; mvdan, brush, and
tree-sitter strictly parse it with head `python3`, one heredoc, and exact body
boundaries.

**JUDGMENT:** The dominant bashlex failure is parser support, not malformed
Terminal-Bench commands.

### 3. Arithmetic expansion

```bash
cp input.csv /tmp/input-test.csv && vim -Nu NONE -n -Es /tmp/input-test.csv -S /app/apply_macros.vim; status=$?; echo "vim_exit=$status"; cmp -s /tmp/input-test.csv expected.csv; cmpstatus=$?; echo "cmp_exit=$cmpstatus"; wc -c /tmp/input-test.csv expected.csv; exit $((status || cmpstatus))
```

**OBSERVATION:** Bashlex raises `NotImplementedError: arithmetic expansion`.
The other three strictly parse it with the same seven heads and both sequence
and arithmetic flags.

**JUDGMENT:** Heredoc replacement cannot repair this independent bashlex
coverage gap.

### 4. Tree-sitter sequence flag

```bash
if command -v bundle >/dev/null 2>&1; then bundle exec jekyll build; else echo 'Bundler is not installed; skipped Jekyll build.'; exit 0; fi
```

**OBSERVATION:** All four strictly parse the command and agree on heads.
Bashlex, mvdan, and brush set `sequence=true`; tree-sitter does not.

**JUDGMENT:** Tree-sitter recovery is useful, but its construct flags should
not define primary feature semantics.

### 5. Tree-sitter head recovery around a heredoc

```bash
LIB=/usr/local/lib/python3.13/site-packages/numpy.libs/libscipy_openblas64_-56d6093b.so
nm -D "$LIB" | grep -Ei 'dgeev' | head -30
python3-config --includes --ldflags
python3 - <<'PY'
import numpy
print(numpy.get_include())
PY
```

**OBSERVATION:** Mvdan and brush agree on `nm`, `grep`, `head`,
`python3-config`, and `python3`. Tree-sitter strictly parses the command but
omits the trailing `python3` head. Bashlex rejects the quoted heredoc.

**JUDGMENT:** This is another reason to keep tree-sitter behind the primary
parser rather than combine their outputs on valid input.

## Artifacts

- [driver.py](driver.py): allowlist enforcement, Bash reference, aggregation,
  focused scanners, performance, and JSON writer.
- [microcases.json](microcases.json): 25 generic expected-feature cases.
- [bashlex_adapter.py](bashlex_adapter.py) and
  [treesitter_adapter.py](treesitter_adapter.py): Python JSONL adapters.
- [go_adapter/main.go](go_adapter/main.go): mvdan JSONL adapter.
- [rust_adapter/src/main.rs](rust_adapter/src/main.rs): brush JSONL adapter.
- [results.json](results.json): complete aggregate.


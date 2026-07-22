# Five-fold prequential profile-update screen (2026-07-21)

**DEVELOPMENT-ONLY / EXPLORATORY. This is not an online activation gate, deployment certificate, or unopened-stream result.**

`swe-rebench-qwen3.7-max-seed42-offset50-100-complete-v2` initializes one profile and fixed gate (100 tasks, 4640 calls). `swe-rebench-qwen3.7-max-fresh-seed42-skip150-n277` supplies the five-fold prequential stream (277 tasks, 13410 calls). The task sets are disjoint.

Each Fresh-277 task is scored exactly once in its held-out fold. Its lane first consumes the other four folds. `warmup snapshot` freezes at test entry; `task` and `call` continue past-only updates inside the held-out fold. `fresh4 static` is a same-100-derived-gate matched reference.

Positive utility delta favors the treatment named by the comparison.

## Primary exact point estimates

| Comparison | KV ms | utility delta ms | mean/task ms | + / - / 0 tasks |
|---|---:|---:|---:|---:|
| frozen 100 vs deadline | 3500 | 22559.11 | 81.44 | 6 / 0 / 271 |
| frozen 100 vs deadline | 5000 | 110726.55 | 399.73 | 118 / 15 / 144 |
| fresh4 static vs deadline | 3500 | 58391.51 | 210.80 | 28 / 6 / 243 |
| fresh4 static vs deadline | 5000 | 28787.87 | 103.93 | 45 / 12 / 220 |
| warmup snapshot vs deadline | 3500 | 47539.86 | 171.62 | 23 / 4 / 250 |
| warmup snapshot vs deadline | 5000 | 151042.85 | 545.28 | 97 / 16 / 164 |
| task vs deadline | 3500 | 47901.80 | 172.93 | 24 / 4 / 249 |
| task vs deadline | 5000 | 160093.05 | 577.95 | 109 / 18 / 150 |
| call vs deadline | 3500 | 47901.80 | 172.93 | 24 / 4 / 249 |
| call vs deadline | 5000 | 160093.05 | 577.95 | 109 / 18 / 150 |
| fresh4 static vs frozen 100 | 3500 | 35832.41 | 129.36 | 28 / 8 / 241 |
| fresh4 static vs frozen 100 | 5000 | -81938.68 | -295.81 | 29 / 89 / 159 |
| warmup snapshot vs frozen 100 | 3500 | 24980.75 | 90.18 | 23 / 10 / 244 |
| warmup snapshot vs frozen 100 | 5000 | 40316.29 | 145.55 | 38 / 30 / 209 |
| task vs frozen 100 | 3500 | 25342.69 | 91.49 | 24 / 10 / 243 |
| task vs frozen 100 | 5000 | 49366.49 | 178.22 | 42 / 18 / 217 |
| call vs frozen 100 | 3500 | 25342.69 | 91.49 | 24 / 10 / 243 |
| call vs frozen 100 | 5000 | 49366.49 | 178.22 | 42 / 18 / 217 |
| task vs warmup snapshot | 3500 | 361.94 | 1.31 | 1 / 0 / 276 |
| task vs warmup snapshot | 5000 | 9050.20 | 32.67 | 12 / 4 / 261 |
| call vs warmup snapshot | 3500 | 361.94 | 1.31 | 1 / 0 / 276 |
| call vs warmup snapshot | 5000 | 9050.20 | 32.67 | 12 / 4 / 261 |
| call vs task | 3500 | 0.00 | 0.00 | 0 / 0 / 277 |
| call vs task | 5000 | 0.00 | 0.00 | 0 / 0 / 277 |

## Timing and publication

Ready before the next within-task causally eligible tool call: 12847/13133 (0.9782).

| Fold | update p50 / p95 / p99 / max ms | call readiness |
|---:|---:|---:|
| 1 | 0.0709 / 0.1536 / 0.4153 / 1.9653 | 2690/2728 (0.9861) |
| 2 | 0.0713 / 0.1492 / 0.5525 / 1.8595 | 2596/2672 (0.9716) |
| 3 | 0.0707 / 0.1432 / 0.4668 / 1.7344 | 2434/2496 (0.9752) |
| 4 | 0.0730 / 0.1679 / 0.5848 / 2.5922 | 2611/2648 (0.9860) |
| 5 | 0.0717 / 0.1603 / 0.4714 / 1.2696 | 2516/2589 (0.9718) |

Host timing drives this replay's publication schedule but is not a deployment-latency guarantee.

No bootstrap CI or sign-flip p-value is reported: held-out-fold updates make later task decisions path-dependent.

Complete calibration, fold membership, decisions, update timings, publication timestamps, model versions/state hashes, and input/source hashes are stored in:

- `prequential-profile-update-2026-07-21-records.jsonl.zst` — 181059 records, SHA-256 `fa78a151949d201fd46510c9a740f45eca24bb32ad3be8662f7c22be09fbc5e7`
- `prequential-profile-update-2026-07-21-f1-records.jsonl.zst` — 83755 records, SHA-256 `8da1c3a40b57565f2a13bc2a20cf8aed3a92d28ddc74206440a19c3c65c7e038`
- `prequential-profile-update-2026-07-21-f2-records.jsonl.zst` — 83419 records, SHA-256 `0cd525102a95d38ffe6aa9837131b26476b4467cf9d8c33100aebedee0388e0b`
- `prequential-profile-update-2026-07-21-f3-records.jsonl.zst` — 82357 records, SHA-256 `e2e0625ad253077f9ddfde1b8284279acbcb5e5d17972ad48d76ed4f8a24ccf5`
- `prequential-profile-update-2026-07-21-f4-records.jsonl.zst` — 83269 records, SHA-256 `8691e9ef65cdcf0ba2a3ca4dd03d607acda9ba4b69dac2a5c20473993706f6b1`
- `prequential-profile-update-2026-07-21-f5-records.jsonl.zst` — 82915 records, SHA-256 `17db295f1a99d389f7e56940f5d4e314b98a2071a68f8e8728881cd30da13d5b`

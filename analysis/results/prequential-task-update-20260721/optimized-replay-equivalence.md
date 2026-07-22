# Optimized replay equivalence

This receipt replaces the scientifically duplicate optimized replay artifacts. Both the reviewed original and optimized replay were recovered from checkpoint tree `950ef8c11fe3dab3ba2dcfc4025de096c3aef9ba` for comparison; only the original sidecars are retained.

## Provenance

- Original producer commit: `e6cca14e00c98353891cc6fe735d9bf3fa12c783`
- Optimized replay commit: `2c8232af10460bc2c35a3af365f12c36ccf9ea7b`
- Shared config SHA-256: `f0b49926af9a4d8cf5029e53df24f27e700ee003876d9738555fd7b60b30bf20`
- Original scorer SHA-256: `45122bf7d694260142b3207fcf1b67bd7422fe53dee21b57a0cee0da4612b5e9`
- Optimized scorer SHA-256: `174da20e6e29da074c1db1c3c00f7fd70d6cf9d93e910d11ecc7561da0c207fd`
- Shared driver SHA-256: `3feeca2f317bde2c0f85f8ccce19407e7d036ca3d3ac7098baa1f1f4227cf0b9`
- Optimized compact report hashes: Markdown `f8e7853669ab43016d7f75a1d80fc956ad18fb57302f5e028166f307e249d6d1`; JSON `87df1a932b58322d112bcc8ced5553d9fb149bde9e0500fb82710c8a1559bb3e`.
- Optimized sidecar hashes, initialization then folds 1–5: `583c1f98a5664833419b1a9fc3daf8ef74502ff06b2173d2fac173fd487efa89`, `2ea32efdf4867cbd38629082276dcc2f66b166bee2614a38bff85924ad41b7d4`, `74a3cf2f6c721bdefc04742eb34d0749496e46ac7213559a27da3459eb2f246d`, `6bc6e076c4358873b9d26226cb73fc219dfd3d3acda2786533e1f2256eecba08`, `bc7db78481744ecd82b49719205a2e1844c49b860540df45af2cfb19531ba1d4`, `fa5b6c020a9b5116a528fdcd11152d21be9257848b4447e47a0864ee18d82e2b`.

## Equivalence result

The streaming comparison covered 596,598 non-metadata records and found zero trigger, utility, score, or policy-decision differences. Scientific point estimates and conclusions are unchanged.

Accepted runtime-only differences were eight rows whose `call_prior_task_count` differed by one, 88 rows differing only in call publication version/count/hash metadata, timing changes from the faster scorer, and aggregate readiness of 12,843/13,133 instead of 12,847/13,133. Readiness is a wall-clock publication race, and none of these differences changed a resulting decision.

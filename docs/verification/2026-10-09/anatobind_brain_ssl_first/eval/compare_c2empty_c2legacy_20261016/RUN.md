# Saved-test descriptive comparison provenance

**NOT_EVIDENCE.** Executed 2026-10-16 using the comparison specified before the new run in `../../u_empty_c2_20261014/README.md`. This reads existing evaluation JSON; it does not rerun model inference or select thresholds. Legacy C2 crossed its failed Q25 check and remains exploratory. Each arm retains its own validation-selected score threshold, so U differences include the operating-point change. The intervals describe patient resampling of these fixed runs, not training-seed variation.

Code base: `77fd6ebc81ec945f77f7c9316f05119294b1aca7`. The two comparison implementation files below were unchanged when the command ran. All 1,056 test rows were paired. Output: `COMPARE.md` and `compare.json`, anonymous aggregates only.

```sh
CUDA_VISIBLE_DEVICES='' PYTHONNOUSERSITE=1 PYTHONPATH=. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 nice -n 19 /home/congcongliu/anaconda3/envs/nvgen/bin/python scripts/aur_compare.py \
  --a /data0/congcong/code/Project_Doing/foundation_model-conformance/docs/verification/2026-10-09/anatobind_brain_ssl_first/eval/c2_test_20261014_1344 \
  --b /data0/congcong/code/Project_Doing/foundation_model/docs/verification/2026-10-09/anatobind_brain_ssl_first/eval/c2_test_20261011_0500 \
  --name-a C2-empty --name-b C2-legacy \
  --out /tmp/compare_c2empty_c2legacy_20261016_reproduction --n-boot 2000
```

Run from the repository root; use a fresh output directory. The recorded run used `docs/verification/2026-10-09/anatobind_brain_ssl_first/eval/compare_c2empty_c2legacy_20261016` as output. Default bootstrap seed: 0.

| Input | SHA256 |
|---|---|
| C2-empty cases.json | `ef4a835877d42b2e62fd50546f0eedd950423a0a8a084a5ba0fc6bd93f9fb754` |
| C2-empty aggregate.json | `36eb9a657696f4e01b1f4a9e3781835e9de676a373e2daf266ac3c5522612b09` |
| C2-legacy cases.json | `0b14cba65940158f553cf0097a83fb6c5415921830598384dbfbfe4b357ea815` |
| C2-legacy aggregate.json | `e2728f098eaf80f661d289ba8341d2fe279fd8579a5a361ab244e8ef7ca5492d` |
| scripts/aur_compare.py | `40e335962ff6c78f6f95f1a37b969bd2402deddd1e1f1f5c24f2938a3eac0a0d` |
| anatobind/aur/compare.py | `178eb0a70bff8a870641b823429b9205fefa2ad0df8283f919356fbc6d1afb7a` |

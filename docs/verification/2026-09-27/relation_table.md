# Relation feature table build (2026-09-27)

## Command

```bash
D=docs/verification/$(date +%F); mkdir -p $D
PYTHONNOUSERSITE=1 PYTHONPATH=. nice -n 19 ~/anaconda3/envs/nvgen/bin/python scripts/build_relation_table.py \
    --out /data2/congcong/data/FM_data/derived/relation/v1 | tee $D/relation_table_output.txt
```

## Manifest (as printed)

```json
{
 "version": "v1",
 "built_at": "2026-09-27T15:37:23+00:00",
 "git_commit": "1474c627e196e25d8ca510e13c4e5f0d4ac441c5",
 "registry": "docs/verification/2026-09-24/gate05/lesions.csv",
 "registry_sha256": "7d40d27b7e5dc64db97a2f5db60aa7e1d448a3d81bd6e4c0ce442f46750538f0",
 "folds": "data/level_r/folds.json",
 "folds_sha256": "99599c72f6332977d31e38a91024941ba1c03e68d32021db2c54a28e1b699a55",
 "seg_root": "/data2/congcong/data/FM_data/derived/synthseg/fastmri_brain/seg_native",
 "kspace_root": "/data2/congcong/data/FM_data/fastMRI_lh_brain_knee/kspace/brain",
 "parameters": {
  "candidate_mm": 15.0,
  "distance_cap_mm": 30.0,
  "soft_sigma_mm": 1.0,
  "patch_mm": 36.0,
  "pixel_mm": 0.75,
  "patch_slices": 3,
  "min_c1_agreement": 0.99
 },
 "n_lesions": 1297,
 "n_patients": 165,
 "per_fold": {
  "0": {
   "patients": 33,
   "lesions": 280
  },
  "1": {
   "patients": 33,
   "lesions": 250
  },
  "2": {
   "patients": 33,
   "lesions": 276
  },
  "3": {
   "patients": 33,
   "lesions": 334
  },
  "4": {
   "patients": 33,
   "lesions": 157
  }
 },
 "c1_distribution": {
  "white_matter": 985,
  "cortex": 310,
  "basal_ganglia": 2
 },
 "c1_source_counts": {
  "overlap": 1278,
  "nearest": 19
 },
 "n_candidate_fallback": 1
}
```

Registry sha256: `7d40d27b7e5dc64db97a2f5db60aa7e1d448a3d81bd6e4c0ce442f46750538f0`.

The one candidate fallback is lesion 1021: `c1_class=cortex`, `c1_source=nearest`, `c1_overlap=0.0`, `dist_cortex_mm=17.61624339210832` (nearest class 17.6 mm away), `registry_lookup_class=cortex` (agrees with C1 despite the fallback).

## C1 agreement

`c1_agreement 0.9953739398612182 mismatches [64, 147, 327, 836, 971, 1086]` — all six above the 0.99 refusal threshold.

Per-mismatch detail (`lesion_id`, registry lookup class, C1 class, `c1_source`), read from `table.csv`:

| lesion_id | registry_lookup_class | c1_class | c1_source | c1_overlap |
|---|---|---|---|---|
| 64 | white_matter | cortex | overlap | 0.4470588235294118 |
| 147 | white_matter | cortex | overlap | 0.5324675324675324 |
| 327 | white_matter | cortex | overlap | 0.34701492537313433 |
| 836 | white_matter | cortex | overlap | 0.41130434782608694 |
| 971 | cortex | white_matter | overlap | 0.5024232633279483 |
| 1086 | white_matter | cortex | overlap | 0.3084636504715292 |

All six are white matter / cortex boundary lesions whose class-level argmax (C1) differs from the label-level registry lookup, consistent with the brief.

## Elapsed time

Command started ~23:31:20, `manifest.json` written 2026-09-27 23:37:24 +0800 (local) — about 6 minutes wall clock, matching the brief's estimate.

每个数字来自上面的命令输出；C1 与注册表标签级查表的差异逐条列出(P6)。

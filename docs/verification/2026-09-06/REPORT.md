# Current-state verification — 2026-09-06

Observed on 2026-09-06, approximately 22:34–22:40 Asia/Shanghai. This is a live filesystem, metadata, and test audit, not a restatement of the research plan.

**Assessment:** the data engine is implemented and its tests pass. The recorded pseudo-label outputs exist and their headers are readable. Data acceptance remains incomplete: fastMRI outputs mix preprocessing geometry versions; SKM-TEA annotations contain records requiring review. No M1 model or training implementation is present.

## Repository and implementation

- Current branch: `plan-v3`.
- Local `plan-v2`, local `plan-v3`, and both live GitHub branch heads: `b0d1b72e3a7af745bb4758f5a5a286a1048b748b`.
- Tracked source had no local modifications at audit start.
- Two pre-existing untracked files contain old proposal text: `AnatoBind-MRI_cui.md` and `粘贴的 markdown (1)。md(20260809-075619)(1)`.
- Inspected tracked, untracked, and ignored contents. `project/`, `.agents/`, and `.codex/` are empty. Ignored files are Python/pytest caches.
- Implemented: fastMRI RSS-to-NIfTI conversion, label resampling, SynthSeg staging/batching/resume, and the runner CLI. The architecture renderer draws diagrams; it does not implement the model.
- Absent from this workspace: SKM-TEA dataset interface, Swin model, A/S/U_B/U_Q heads, relation/E modules, k-space intervention engine, training loops, M1 results, and checkpoints.

The plan at this commit is v2.1. Its 1A/2A/3B/4B choices are design decisions, not completed experiments. Its M0 status includes older inventory statements and cannot be used as the current execution status.

## Tests and active jobs

Executed from the repository root:

```sh
env PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. \
  /home/congcongliu/anaconda3/envs/nvgen/bin/python \
  -m pytest -q -p no:cacheprovider
```

Result: **31 passed in 0.65s**, exit 0. The documentation's count of 24 is stale. These are data-engine tests, including injected substitute SynthSeg runners; they do not validate segmentation quality or M1 performance.

A host-level `ps -ef` search found no matching `run_synthseg`, `SynthSeg_predict`, `run_nii_chain`, `run_isles`, `nnUNetv2_train`, or Python processes naming `anatobind`/`skm_tea`. This is a point-in-time process-name check, not proof that no differently named or remote job exists.

## SynthSeg artifacts

Source root: `/data2/congcong/data/FM_data/derived/synthseg`.

| Dataset | Manifest rows | Successful outputs | Readable/native-shape-valid headers | Affine discrepancies against current reference rule |
|---|---:|---:|---:|---:|
| fastMRI brain annotated subset | 997 | 996 | 996 | 139 |
| PDGM | 501 | 501 | 501 | 0 |
| BMSR | 461 | 461 | 461 | 0 |
| HCP | 1113 | 1113 | 1113 | 0 |
| ISLES | 250 | 250 | 250 | 0 |
| Total | 3322 | 3321 | 3321 | 139 |

All successful manifest paths correspond to output files: no missing, unlisted, zero-byte outputs or duplicate stems. All headers have valid dimensions, positive spacing, finite/invertible affines, and integer label types. The only skipped record is `file_brain_AXT1_201_6002824`, with fewer than four slices.

For PDGM/BMSR/HCP/ISLES, output shape and affine were compared with each source NIfTI header. For fastMRI, output geometry was compared with original RSS shape and the current encoded-FOV/matrix conversion convention. All three padded thin-stack outputs have the original slice count and match the current native-grid convention.

Eighteen deterministic output samples were fully decompressed with gzip CRC verification and voxel loading: first/middle/last manifest-success rows per dataset plus the three fastMRI thin-stack outputs. All samples contain foreground and only expected SynthSeg label IDs. This is a payload sample, not full-volume anatomical review or full-payload verification of all 3321 outputs.

Evidence: [synthseg.json](synthseg.json), including manifest SHA-256 values and sampled identifiers.

### fastMRI preprocessing version mismatch

Independent comparison of all 996 fastMRI raw headers and staged/native NIfTI headers confirmed:

- 857 native outputs match the current encodedSpace spacing formula.
- 139 native outputs and their staged inputs match the previous reconSpace formula instead.
- All 996 native outputs are internally consistent with their stored staged reference, accounting for thin-stack padding.
- All 996 output shapes are correct under the conversion convention.

Affected series: AXFLAIR_201 **10**, AXFLAIR_202 **65**, AXFLAIR_206 **2**, AXT1_202 **62**.

The differences are small: relative spacing differences approximately **0.0128%–0.3271%**, with maximum absolute affine translation-coefficient differences per affected volume approximately **0.0127–0.3219 mm**. This is not evidence of a twofold size error or proof that these segmentations are anatomically incorrect.

Example: `file_brain_AXFLAIR_201_6002888` stores column spacing `0.6875 mm`; the current formula yields `0.689756489 mm`. The corresponding centered affine x translation changes from `89.03125` to `89.323465267 mm`.

Likely cause, supported by file timestamps, old-formula agreement, and source inspection: cached staged/output files survived the geometry-rule change. Staging and resume skip existing files without geometry-version validation. Both the physical correctness of the chosen convention and the documented left/right handedness assumption still need independent reference validation.

Evidence: [affected-file CSV](fastmri_affine_mismatches.csv), [independent geometry comparison](fastmri_affine_details.json).

## SKM-TEA inputs

Roots under `/data2/congcong/data/FM_data`:

- `SKM-TEA_ltr/annotations/v1.0.0/`
- `SKM-TEA_ltr/segmentation_masks/dicom-track/`
- `SKM-TEA/files_recon_calib-24/`

All **155/155 raw HDF5 files** open and expose `kspace`, `maps`, `masks`, and `target`. All **155/155 segmentation NIfTI headers** load with finite/invertible affines, positive spacing, and shapes matching the corresponding JSON dimensions and raw spatial dimensions. Raw tensor payloads were not loaded.

The official splits contain **86/33/36 scans** and **242/104/130 annotations**, totaling **155 distinct subjects/scans and 476 annotations**. Subjects and scans do not overlap between splits. Numeric image/annotation IDs are split-local and require namespacing when combined.

### Annotation findings

Under the plan's declared `[x,y,z,w,h,d]` interpretation, **9 boxes have negative extents**: 7 in the nominal main group, 1 effusion, 1 ligament. Two remain out of bounds even after sorting the signed start/end coordinates:

| Split / annotation ID | Scan | Finding |
|---|---|---|
| train / 191 | MTR_161 | z endpoint 271 exceeds declared depth 160 |
| val / 29 | MTR_047 | x endpoint 2478 exceeds declared width 512 |

Two additional records have suspicious category/tissue combinations:

| Split / annotation ID | Scan | Category → tissue |
|---|---|---|
| train / 48 | MTR_145 | Ligament Tear → Meniscus |
| test / 117 | MTR_156 | Cartilage Lesion → ACL |

These two records do not overlap the nine box findings. The nominal category-based **319/117/40** groups are reproducible, but are not an accepted final cohort. Applying only strict box validity and category/tissue consistency screens leaves **311/116/38** records; that is a diagnostic count, not an adopted exclusion policy. Original annotations were not corrected or removed.

### Geometry and ontology interfaces still require acceptance

- Meniscus and Tibial Cartilage tissue labels lack medial/lateral identity; they cannot be treated as a direct one-to-one mapping to all six anatomy queries.
- 151 volumes have depth 160; four have depths 168, 152, 156, and 144.
- JSON orientation metadata contains 137 `SI/AP/LR` and 18 `SI/AP/RL` cases. NIfTI affine axis codes are `P/I/R`. Header shape agreement does not establish target/mask/box physical alignment.
- Eight volumes have 16 coils; the remaining 147 have 8. All have two echoes.
- Each of six provided Poisson masks has shape `416 × (depth−80)`, requiring an explicit embedding rule for the raw ky–kz grid.
- Two unreferenced `.h5.truncated` remnants exist; they are not the valid referenced inputs and were not modified.

Evidence: [skmtea.json](skmtea.json), including all nine box records, metadata distributions, and file-read exceptions (none).

## Next work justified by this verification

1. Resolve the fastMRI geometry-version policy and resume/cache validation; decide whether affected artifacts need regeneration after reference validation.
2. Review the SKM-TEA box semantics and suspicious category/tissue records; preserve source annotations and freeze an explicit inclusion/handling policy.
3. Implement and accept the SKM-TEA manifest, ontology mapping, target/mask/box coordinate mapping, Poisson-mask embedding, and subject-level five-fold split.
4. Validate small multi-coil intervention examples, then build the two M1 arms and run the gate experiment.

No training-readiness claim follows from file counts or passing plumbing tests alone. This audit added only verification documents/evidence to the repository; it did not change source code or data, regenerate labels, launch training, commit, or push.

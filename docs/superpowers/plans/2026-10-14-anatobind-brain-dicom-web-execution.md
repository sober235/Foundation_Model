# AnatoBind-Brain DICOM A/U/R 公共交互原型实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans after this plan is approved. Native inline execution; no subagents.

**Goal（完整产品的后续目标）:** 从公开 HTTPS URL 接收一张 2D 或一个 3D brain MRI DICOM input，输出可交互的 A/U/R 结果并支持下载。当前优先完成模型与离线验收，执行顺序以下列 Q28 更新为准。

**Architecture:** 3D DICOM series 经 DICOM intake 转成带真实 geometry 的 volume，进入现有 3D AUR；孤立 2D DICOM 进入独立训练的 2D AUR checkpoint。FastAPI 提供临时 job API，vanilla JS 展示切片、A/U overlays 与 R host；Caddy 对公网提供 HTTPS。两种模型共享 A/U/R 标签和结果 schema，不共享未经验证的 checkpoint。

**Tech Stack:** Python, PyTorch, SimpleITK, FastAPI, python-multipart, Uvicorn, vanilla HTML/CSS/JS, Caddy.

**Spec:** `docs/superpowers/specs/2026-10-14-anatobind-brain-dicom-web-design.md`

**2026-10-19训练准备复核：** 本文保留原产品范围与任务分解；Task4/5代码尚未实现，所列命令不是已执行或可直接启动的入口。旧2D配方仅含II/III，兼容初始化及2D Stage I/表征验收未定义；batch/LR/240k与80k曝光仍待实测冻结。正式开训前按[新的路线与就绪性分析](../../plans/2026-10-19-aur-dual-input-route-and-training-readiness.md)补齐数据/几何/目标、架构初始化、可恢复训练与评价闭环。DICOM产品就绪、Dataset909最终比较结果和3D全部质量门不是第一个独立2D optimizer step的技术前提；最终评价/发布仍执行各自门槛。架构与训练配方须在新实施设计中明确，本文不作为未经复核的长训练启动命令。

**参照资格澄清：** 下文“相同2D样本/患者”指相同锁定分区、相同U可监督子集及相同实例/ignore定义。AUR可额外使用A-only平面训练A/S，但U不可见或未标注平面不能作为Dataset909的健康阴性。

**Status（2026-10-16 收尾更新）:** 以下保留 2026-10-14 完整产品范围和任务设计。用户 Q28（2026-10-13 09:53 UTC）已要求“先不考虑这里显示的问题，仅仅考虑模型是否能否成功即可,显示后续再说”：**模型成功优先；DICOM 解码、离线 A/U/R 验收及独立 2D 模型任务保留，API/UI/公开 URL 为后续阶段。** 本文件不表示这些任务已实施。3D 三目标修正 Task 1–6 已完成，Task 7 新 U 损失 C2 的训练/val/test 已完成，预登记的 legacy 描述性配对也已收尾；`c2_test_20261014_1344` 的 G2-A/G2-U/G3 全部失败（[结果](../../verification/2026-10-09/anatobind_brain_ssl_first/u_empty_c2_20261014/RESULTS.md)）。当前应记录失败、完成分析并提出新模型计划，不因门失败转去继续做显示。

**阅读本计划的阶段约束:** Tasks 1–5 的模型、DICOM 几何和离线任务仍在范围内，Task 6 只保留离线结果契约。Tasks 7–10 的服务、显示、公网部署和公开 URL 验收全部后移；其原文是后续产品设计，不是当前可执行的并行任务。独立 2D 需要自己的数据、参照、checkpoint 和 A/U/R 验收，不能用 3D 或网页成果替代。

## Global Constraints

- 本计划只支持脑部 MRI，单 sequence/每个 job；不做 CT、X-ray、其他器官、多序列融合或疾病诊断。
- 两个输入模式：单张 2D DICOM (DICOM geometry must be valid) 和一个 3D DICOM series / 3D multi-frame object。用户可上传多个文件；混合 series 时必须选择其一。
- A=32 anatomy masks；U=generic abnormality instances from the labeled training distribution；R=每个 U instance 的 13 类分侧 host + no-host。2D/3D 共用输出语义，使用各自模型和各自验证门。
- 2D 模型由 3D volume labels 派生切片训练；先按 patient split，再扩增平面/视图。不得切片后再随机拆 train/val/test。
- 所有现有标签和内部关系真值含伪标签，结果必须标 `NOT_EVIDENCE`。页面固定标注 research prototype / not a diagnosis；不要声称识别任意未知病变。
- DICOM uploads 必须去标识。服务不持久保留 DICOM、mask、patient tags 或病例级 outputs；默认 job 完成 30 分钟清除。
- URL 无需登录且公网可达。原始 API 只监听 `127.0.0.1`，由 Caddy 提供 HTTPS；单个 GPU worker、限长队列。
- 3D 训练/验证依赖 `docs/superpowers/plans/2026-10-13-anatobind-aur-three-goal-conformance.md`。该计划的 G2-A/G2-U/G3 若失败，记录失败并停止当前模型实验，后续修改须新计划；不以 API/UI 工作接续失败模型。preview URL 或 UI smoke 不代表 A/U/R 达标。
- 数据/模型路径默认从 `/data2/congcong/data/FM_data` 配置；DICOM 输入 job 放独立临时目录，绝不挂载整个 `FM_data` 到公网服务。

## Product contract

```text
single 2D DICOM ──> DICOM Input2D ──> AUR2D checkpoint ──> 2D A/U masks + per-instance R
3D DICOM series ──> DICOM Input3D ──> existing AUR3D checkpoint ──> 3D A/U masks + per-instance R
                                                     │
                                                     └──> FastAPI job API
                                                            │
                                                            └──> public browser viewer
```

Single-slice output is 2D only (area mm²); do not invent slice thickness or report lesion volume. 3D output preserves the source grid and reports volume mm³. Both return the same lesion record fields: `instance_id`, `host`, `host_side`, `host_probs` (14 classes), `evidence`, and a mask asset.

---

### Task 1: Freeze and verify the 3D A/U/R model dependency

**Files:**
- Read: `docs/superpowers/plans/2026-10-13-anatobind-aur-three-goal-conformance.md`
- Create: `docs/verification/2026-10-14/anatobind_brain_dicom_web/model_readiness.md`

**Interfaces:**
- Consumes: the 3D training sample manifest, validation-patient manifest, G1 checkpoint and report from the existing conformance plan.
- Produces: selected 3D checkpoint path/hash, exact sample split hashes, G2-A/G2-U/G3 verdicts and `NOT_EVIDENCE` label.

- [ ] Consolidate the actual 2026-10-13 model-correction results: Tasks 1–6 and the new-loss C2 training/val/test are complete; preserve the failed gates, completed descriptive comparison, historical legacy Q25 deviation and original run artifacts. Any further model change needs a new reviewed plan. Preserve fail-closed Q25 for the legacy recipe; the new `unmatched_empty_v1` profile has no legacy early-stop gate by preregistration.
- [ ] Record the actual 3D checkpoint, run IDs, test reports, all three gate results and manifest hashes in `model_readiness.md`.
- [ ] If a gate fails, leave the result marked NOT PASSED, preserve the complete reports, and stop the current model experiment. Complete its declared analysis and propose a new model plan. Retain the DICOM decoding/offline acceptance and independent 2D work in scope; API/UI/public-URL work stays deferred under Q28.
- [ ] Commit the readiness record with `git commit -m "docs: record 3D AUR readiness for DICOM product"`.

### Task 2: DICOM intake and geometry-preserving decoding

**Files:**
- Create: `anatobind/aur/dicom.py`
- Create: `tests/test_aur_dicom.py`
- Create: `scripts/inspect_aur_dicom.py`

**Interfaces:**
- `Input2D(pixels_hw, origin_ras_mm, row_axis_ras, col_axis_ras, spacing_hw_mm, sequence_type)`
- `Input3D(pixels_zyx, affine_ras, spacing_xyz_mm, sequence_type)`
- `inspect_dicom_files(files) -> list[SeriesInfo]`
- `read_dicom_input(files, series_id=None, sequence_type=None) -> Input2D | Input3D`
- `SeriesInfo` contains a random response ID, dimension, modality, shape, spacing and safe series description; never returns patient name/ID or raw UID.

- [ ] **Step 1: Inventory true DICOM inputs** under `/data2/congcong/data/FM_data`. Record whether representative deidentified 2D and 3D brain MR DICOMs exist; if absent, use a permitted deidentified public sample for runtime validation. Keep pixels outside Git.
- [ ] **Step 2: Add synthetic DICOM fixtures** for one 2D MR instance, a multi-file 3D series, a 3D multi-frame object, a 4D object and a directory with two different Series UIDs. Use only generated pixel data and fake identifiers.
- [ ] **Step 3: Add failing geometry tests** for Modality LUT/RescaleSlope/Intercept, array order, LPS-to-RAS origin/direction, pixel spacing, slice spacing/order, dimension routing, mixed-series selection, duplicate slices, missing geometry, non-MR and 4D rejection.
- [ ] **Step 4: Implement DICOM loading** with SimpleITK `ImageFileReader` for a single object and `ImageSeriesReader`/GDCM series IDs for multi-file series. Sort by DICOM geometry through the reader API; do not sort lexically by filename. Read only required metadata; convert LPS geometry to RAS once.
- [ ] **Step 5: Sanitize metadata** before returning `SeriesInfo`; error messages and logs may include random job ID and error code only.
- [ ] **Step 6: Run** `pytest tests/test_aur_dicom.py -q -p no:cacheprovider`.
- [ ] **Step 7: Commit** with `git add anatobind/aur/dicom.py tests/test_aur_dicom.py scripts/inspect_aur_dicom.py && git commit -m "feat: add geometry-safe brain DICOM intake"`.

### Task 3: Route 3D DICOM through existing AUR3D inference

**Files:**
- Modify: `anatobind/aur/infer.py`
- Create: `anatobind/aur/dicom_infer.py`
- Create: `scripts/infer_anatobind_brain_dicom.py`
- Create: `tests/test_aur_dicom_infer.py`

**Interfaces:**
- `infer_3d_input(input_volume: Input3D, checkpoint, out_dir, device) -> InferenceResult3D`
- CLI: `scripts/infer_anatobind_brain_dicom.py --input-files FILE [FILE ...] --checkpoint PATH --out NEW_DIR --gpu ID [--series-id OPAQUE_ID] [--seq T1|T1c|T2|FLAIR|DWI|ADC]`.
- Reuse the current 3D model, crop/window logic and thresholds from its evaluation report.
- Output masks/overlays must map back to the input volume’s original DICOM-derived geometry.

- [ ] Test `Input3D` versus its equivalent NIfTI volume: apply identical Modality LUT/RescaleSlope/Intercept, normalized pixels, affine, sequence override, segmentation array shape, lesion ids and JSON records within numerical tolerance. Pixel display windowing remains a browser-only setting.
- [ ] Test reversed slice order, oblique direction cosines, anisotropic spacing and a volume whose z depth is smaller than the training crop.
- [ ] Refactor inference internals to consume `(pixels, affine)`; keep the existing `--nifti` CLI backward compatible and add `--dicom`/series mode through the new entry.
- [ ] Confirm no direct DICOM tag or original filename enters output JSON. Run targeted CPU tests and a GPU smoke on an approved idle card after the CPU contract passes.
- [ ] Run `pytest tests/test_aur_dicom_infer.py tests/test_aur_infer.py -q -p no:cacheprovider`; commit adapter, CLI and tests with `git commit -m "feat: run AUR3D on DICOM volumes"`.

### Task 4: Build patient-locked 2D A/U/R training data

**Files:**
- Create: `anatobind/aur2d/samples.py`
- Create: `anatobind/aur2d/targets.py`
- Create: `scripts/aur2d_prepare.py`
- Create: `scripts/aur2d_reference_prepare.py`
- Create: `tests/test_aur2d_samples.py`
- Create: `tests/test_aur2d_targets.py`
- Create: `tests/test_aur2d_reference.py`

**Interfaces:**
- Input manifest: `/data2/congcong/data/FM_data/derived/aur/samples_1mm_v1.json`.
- Output: `derived/aur2d/samples_2d_v1.json`, `splits_2d_v1.json`, `inventory_2d.csv`, leakage report, manifest SHA-256; nnU-Net raw `Dataset909_AUR2DReference` with matching `splits_final.json`.
- One row is one 2D plane from one sequence volume; records source, patient, sequence, plane orientation, image/anatomy/lesion paths and A/U/R supervision flags.
- U reference dataset: `Dataset909_AUR2DReference` with binary lesion masks, 2D nnU-Net configuration, and the same locked patient train/validation/test split.

- [ ] Split by original patient first, then extract slices. Every axial/coronal/sagittal and random oblique view of one patient inherits the same split.
- [ ] Derive A labels from the existing SynthSeg maps and U labels from each source's existing visible lesion values; ignore small-lesion regions under the existing 3D volume floor.
- [ ] Derive R pseudo-targets from the same-slice 13-host map using overlap then nearest in-plane host; add no-host as class 14.
- [ ] Add tests that all image/label transforms share one geometry transform, no patient/slice leakage occurs, left/right survives orientation changes and U-unsupervised slices are not negative labels.
- [ ] Build the Dataset909 reference from the exact same 2D manifest. Its training patients and image rows must equal the AnatoBind 2D train partition; val/test patient IDs must equal the AnatoBind 2D partitions.
- [ ] Run `python scripts/aur2d_prepare.py --samples /data2/congcong/data/FM_data/derived/aur/samples_1mm_v1.json --out /data2/congcong/data/FM_data/derived/aur2d`; inspect inventory and QC montages before training.
- [ ] Prepare the nnU-Net environment and run the Dataset909 reference:

```bash
export FM_DATA_ROOT=/data2/congcong/data/FM_data
export nnUNet_raw=$FM_DATA_ROOT/derived/nnunet/raw
export nnUNet_preprocessed=$FM_DATA_ROOT/derived/nnunet/preprocessed
export nnUNet_results=$FM_DATA_ROOT/derived/nnunet/results
python scripts/aur2d_reference_prepare.py --samples $FM_DATA_ROOT/derived/aur2d/samples_2d_v1.json --out $nnUNet_raw/Dataset909_AUR2DReference
nnUNetv2_plan_and_preprocess -d 909 --verify_dataset_integrity
nnUNetv2_train 909 2d 0 -tr nnUNetTrainer_250epochs
```

- [ ] Run `pytest tests/test_aur2d_samples.py tests/test_aur2d_targets.py tests/test_aur2d_reference.py -q -p no:cacheprovider`; commit the preparation code and tests with `git commit -m "feat: build patient-locked 2D AUR samples"`.

### Task 5: Train and evaluate the dedicated 2D AnatoBind model

**Files:**
- Create: `anatobind/aur2d/model.py`
- Create: `anatobind/aur2d/dataset.py`
- Create: `anatobind/aur2d/losses.py`
- Create: `anatobind/aur2d/train.py`
- Create: `anatobind/aur2d/infer.py`
- Create: `anatobind/aur2d/eval.py`
- Create: `scripts/aur2d_train.py`
- Create: `scripts/aur2d_eval.py`
- Create: `tests/test_aur2d_model.py`
- Create: `tests/test_aur2d_train.py`
- Create: `tests/test_aur2d_losses.py`
- Create: `tests/test_aur2d_infer.py`
- Create: `tests/test_aur2d_eval.py`

**Interfaces:**
- A head: 32 anatomy masks and structure presence.
- U head: 64 2D event queries, presence + per-instance mask; unmatched queries learn empty masks under the same `unmatched_empty_v1` semantics.
- R head: one 14-class host/no-host distribution per U instance; physical 2D coordinates and orientation are inputs.
- `infer_2d_input(input_frame: Input2D, checkpoint, device) -> InferenceResult2D`.
- Training entry: `scripts/aur2d_train.py --samples PATH --stage II|III --seen-slices INT --out NEW_DIR --seed INT`; evaluation entry: `scripts/aur2d_eval.py --checkpoint PATH --samples PATH --split val|test --out NEW_DIR`.
- Stage II exports `aur2d_stage2_best.pt`; Stage III exports `aur2d_stage3_best.pt`.
- Stage II trains A/U/S and freezes R; Stage III resumes Stage II and adds R, matching the 3D stage semantics.

- [ ] Implement `AnatoBindBrain2D`: 2D Swin stages with channels [64,128,256,512], depths [2,2,6,2], patch stem (4,4), 32 anatomy queries, 64 event queries and one 14-class relation output per U instance. Input includes image, valid mask, 3D patient-space coordinates on the 2D plane, local coordinates and optional sequence type. Do not load 3D weights as if they were 2D weights.
- [ ] Add losses with explicit `a_supervised`, `u_supervised`, `r_supervised`; mask invalid points and ignore unsupported labels.
- [ ] Preprocess Dataset909 with `nnUNetv2_plan_and_preprocess -d 909 --verify_dataset_integrity`, then train the reference with `nnUNetv2_train 909 2d 0 -tr nnUNetTrainer_250epochs`; its `splits_final.json` must keep the locked validation/test patients out of training. Select the reference FP/image point and AnatoBind thresholds using validation only.
- [ ] Freeze the proposed 2D recipe after the smoke: seed 0, global batch 24, microbatch 12 on two cards, grad accumulation 1, 160×160 mm crop at 1 mm, 24 workers/rank, Stage II/III 240k/80k seen slices. Use script-scale `II_BB=5e-5); global-batch scaling gives Stage II backbone/head 6.124e-5/6.124e-4 with 667-step warmup, Stage III backbone/head/relation 6.124e-5/3.062e-4/6.124e-4 with 133-step warmup. Save the full run config and manifest hash before main training. Test is evaluated once and is never used to change the model or threshold.
- [ ] Run AUR2D into new, non-existing run directories:

```bash
RUNS=/data2/congcong/data/FM_data/derived/aur/ssl_runs/aur2d
STAMP=$(date +%Y%m%d_%H%M%S)
STAGE2_DIR=$RUNS/stage2_240k_$STAMP
STAGE3_DIR=$RUNS/stage3_80k_$STAMP
python scripts/aur2d_train.py --samples /data2/congcong/data/FM_data/derived/aur2d/samples_2d_v1.json --stage II --seen-slices 240000 --seed 0 --global-batch 24 --microbatch 12 --grad-accum 1 --out "$STAGE2_DIR"
python scripts/aur2d_train.py --samples /data2/congcong/data/FM_data/derived/aur2d/samples_2d_v1.json --stage III --resume-stage2 "$STAGE2_DIR/aur2d_stage2_best.pt" --seen-slices 80000 --seed 0 --global-batch 24 --microbatch 12 --grad-accum 1 --out "$STAGE3_DIR"
```
- [ ] Evaluate: A 13-host macro Dice ≥ 0.80 (report all 32); U per-source sensitivity ≥ same 2D reference at its validation-selected FP/image point − 0.05; R controlled ABA(R) ≥ B0. Report orientation/resolution strata separately.
- [ ] Mark pseudo-label metrics `NOT_EVIDENCE`. If any gate fails, preserve reports and keep the public inference release disabled until a new model plan passes.
- [ ] Run `pytest tests/test_aur2d_model.py tests/test_aur2d_train.py tests/test_aur2d_losses.py tests/test_aur2d_infer.py tests/test_aur2d_eval.py -q -p no:cacheprovider`; run one CPU synthetic forward/backward smoke and one approved idle-GPU smoke before training. Record model config, seed, manifest hash, checkpoint hash and all gates; commit source/tests with `git commit -m "feat: add 2D AnatoBind AUR model"`.

### Task 6: Shared offline inference contract and mask assets

**Files:**
- Create: `anatobind/aur/service_schema.py`
- Create: `anatobind/aur/results.py`
- Create: `tests/test_aur_service_schema.py`

**Interfaces:**
- Current model/offline scope: `InferenceResult2D` and `InferenceResult3D` serialize to the same JSON schema version 1. The future API may consume this contract after model work; defining it does not authorize service/UI implementation now.
- A result has mode, geometry, sequence/sequence source, evidence status, anatomy asset, lesion list and per-lesion `host`, `host_side`, 14-class `host_probs`.
- 2D assets are PNG masks on the original 2D pixel grid; 3D assets are NIfTI masks on the original 3D grid plus viewer-ready slice overlays.

- [ ] Add schema validators and tests for both modes, unknown labels, NaN probabilities, empty lesion list and output-grid mismatch.
- [ ] Preserve `NOT_EVIDENCE`, checkpoint SHA, code SHA and operating point in the result; never include source file path or DICOM identifiers.
- [ ] Run `pytest tests/test_aur_service_schema.py -q -p no:cacheprovider`; commit with `git commit -m "feat: define shared 2D and 3D inference result schema"`.

### Task 7: Public inference API and bounded job worker（后续阶段，Q28 后移）

**Files:**
- Create: `anatobind/web/__init__.py`
- Create: `anatobind/web/api.py`
- Create: `anatobind/web/jobs.py`
- Create: `anatobind/web/worker.py`
- Create: `scripts/serve_anatobind_web.py`
- Create: `tests/test_aur_web_api.py`
- Modify: `requirements-ci.txt`
- Create: `requirements-web.txt` with FastAPI, Uvicorn and `python-multipart`; pin the exact versions tested in the existing nvgen Python environment.

**Interfaces:**
- `POST /api/intakes`: accepts one DICOM object, multiple DICOM files or a ZIP containing only DICOM files; validates them and returns safe series options plus a random intake id/token pair.
- `POST /api/jobs`: requires intake id/token, selected opaque series id (mapped to the internal UID server-side only), mode and optional sequence override; returns job id and one-time job token.
- Intake/job tokens travel in the `Authorization: Bearer` header, never in URLs or logs.
- `GET /api/jobs/{job_id}` returns progress; `GET /api/jobs/{job_id}/result` returns JSON; `GET /api/jobs/{job_id}/assets/{name}` streams overlay/mask; `DELETE /api/jobs/{job_id}` erases job data.
- Upload endpoint uses FastAPI `UploadFile` + `python-multipart`; the app binds only to `127.0.0.1:8765`.

- [ ] Keep one bounded GPU inference queue and one resident worker; load only the selected 2D or 3D checkpoint. Queue size ≤ 2 waiting jobs; return 429 if full.
- [ ] Enforce 1 GiB total upload and uncompressed ZIP size, 2048 files, one selected series/job, request timeout and maximum decoded volume dimensions. Reject ZIP path traversal, symlinks, non-DICOM files, malformed DICOM, non-MR modality and 4D time series. Rate-limit each remote IP to 5 intakes per 10 minutes and one active job; keep rate-limit state in memory only and never log the raw IP.
- [ ] Create a random per-job token; require it to poll/read/delete that job. No login required to open the public page.
- [ ] Delete intake, pixels, masks and results 30 minutes after completion or immediately on DELETE. A startup sweeper removes expired job directories.
- [ ] Add tests for upload, multi-series selection, ZIP traversal rejection, job states, token isolation, upload/rate limits, timeout, cleanup and metadata/filename suppression.
- [ ] Run `pytest tests/test_aur_web_api.py -q -p no:cacheprovider`; commit with `git commit -m "feat: add temporary DICOM inference job API"`.

### Task 8: Interactive browser viewer（后续阶段，Q28 后移）

**Files:**
- Create: `web/aur_demo/index.html`
- Create: `web/aur_demo/app.js`
- Create: `web/aur_demo/style.css`
- Create: `tests/test_aur_web_static.py`

- [ ] Implement drag/drop and file selection for one DICOM, multiple DICOM files or a DICOM ZIP; show safe series summaries and mode/sequence selection.
- [ ] Show queue/progress/error state; poll by job token; make reset/delete visible.
- [ ] Render 2D image with toggleable A/U masks and a clickable lesion list. Render 3D with slice slider, window/level, A/U toggles and per-lesion R details.
- [ ] Provide downloads: JSON + mask PNG for 2D; JSON + NIfTI anatomy/lesion masks for 3D.
- [ ] Keep all UI assets local; no external analytics, fonts, scripts or remote image calls. Add a persistent `Research prototype / NOT_EVIDENCE / not a diagnosis` banner.
- [ ] Add static contract tests for required modes, result schema fields, error states, deletion and download routes.
- [ ] Run `pytest tests/test_aur_web_static.py -q -p no:cacheprovider`; commit with `git commit -m "feat: add interactive AUR DICOM viewer"`.

### Task 9: Public HTTPS deployment（后续阶段，Q28 后移）

**Files:**
- Create: `deploy/Caddyfile`
- Create: `deploy/systemd/anatobind-brain-web.service`
- Create: `scripts/web_healthcheck.py`
- Create/modify verification records under `docs/verification/2026-10-14/anatobind_brain_dicom_web/`

- [ ] Build the service in the existing nvgen runtime plus FastAPI, Uvicorn and `python-multipart`; keep the GPU inference worker single-process.
- [ ] Configure `AUR_2D_CHECKPOINT` and `AUR_3D_CHECKPOINT` from the passing mode-specific model reports; startup refuses a missing checkpoint or a SHA-256 mismatch.
- [ ] Caddy serves `PUBLIC_HOST` on HTTPS and reverse-proxies to `127.0.0.1:8765`. Configure host DNS and ports 80/443; do not expose the API port directly.
- [ ] Add service restart policy, health endpoint, bounded log rotation and a deployment secret for job-token hashing. API/Caddy logs contain job id/status/duration/error code only; disable access-log fields that expose raw client IP, DICOM filename or tokens.
- [ ] Verify public URL from an external network with no login; verify HTTP redirects to HTTPS, health endpoint passes, 2D and 3D uploads reach inference, and the temporary files disappear after TTL.
- [ ] Record the actual public URL, commit SHA, checkpoint hashes, service host and smoke outputs in `public_url.md`. No URL is considered delivered until external-network verification passes.
- [ ] Run `python scripts/web_healthcheck.py --base-url "$PUBLIC_BASE_URL"`; inspect systemd/Caddy logs for PHI leakage and commit deploy files/runbook with `git commit -m "ops: deploy public brain AUR DICOM demo"`.

### Task 10: Final end-to-end acceptance（后续公网产品阶段）

**Files:**
- Modify: `STATUS.md`
- Create: `docs/verification/2026-10-14/anatobind_brain_dicom_web/REPORT.md`

- [ ] Run targeted DICOM, 2D model, shared schema, API and static UI tests; then run the full CPU suite. Run one GPU smoke per dimension on approved idle hardware.
- [ ] Test two deidentified fixtures through the public URL: one standalone 2D DICOM and one multi-slice 3D series. Confirm A/U/R masks map back to exact input geometry and the viewer reports 2D area vs 3D volume correctly.
- [ ] Confirm model gates for each dimension, no cross-patient leakage, public URL access, upload limits, token isolation, error handling and automatic cleanup.
- [ ] If a model gate fails, preserve reports, mark the corresponding A/U/R target NOT PASSED, and stop the model plan and release claims. Do not tune on test or use preview/UI work as the next model task; API/UI remains deferred under Q28. A further model change needs a new reviewed plan.
- [ ] Update `STATUS.md` with the actual public URL, mode-specific model/checkpoint IDs, gate results, privacy/retention behavior and `NOT_EVIDENCE` boundary.
- [ ] Run `git diff --check`; commit the final E2E report and status with `git commit -m "docs: record public AUR DICOM end-to-end acceptance"`.

## Execution order and release gate

Current order under Q28/Q29/Q30: the existing 3D conformance training/val/test and declared descriptive analysis are complete; preserve the failed A/U/R gates; complete failure diagnosis, then propose and review the next model plan before any further training or model change. Preserve DICOM input/geometry decoding, offline 3D acceptance, independent 2D data/reference/model/acceptance, and the shared offline inference contract. Model failure does not authorize a detour into display or service work.

API/UI, public deployment and external-network acceptance are later phases after the model work. Their task text above preserves the original product requirements; it is not authorization for parallel display work while training proceeds. Public model inference requires the applicable 2D/3D gates to pass. The eventual product target is complete only when both input modes work end-to-end at the public URL and those existing A/U/R gates pass; no medical validation or arbitrary unknown-abnormality gate is added here.

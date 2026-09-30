# Whole-branch code review 1 (most capable model), code at a22da8b, returned 2026-09-29 16:05

Saved by the controller from the reviewer's hand-back, shortened only where marked. Scratch of the reviewer:
`/tmp/claude-1002/-home-congcongliu--claude/614ccd9c-f8e8-435b-afc8-47ae37fe3721/scratchpad/review_final/`.

## Verdict

Ready for the evaluations — on the dry run's real nnU-Net outputs the branch's counts equal an independent count, the
predictions it reads are out-of-fold, and every hand-over agrees; what remains is a rule for the early stop (no code),
the queue's handling of failed trainings, and the sentence's wording before Task 12 writes the records.

## What was checked across modules (verified = run, read = code reading)

- Hand-overs 1–11 (label file → array → components → rows → voxel volume → nnU-Net output → score → component ids →
  binding → threshold → matching → record): all agree. Verified on 6 real outputs of the dry run (orientations LPS, RAS,
  LAS): npz is float32 (2, z, y, x), agreement with the label map 1.000000, the prediction's affine equals the label's
  and the anatomy's (largest deviation 3e-8); every predicted voxel has foreground probability > 0.5 (smallest 0.5002);
  on 6 real SynthSeg maps the left labels lie on the side the header calls left (labels and headers agree; this does
  not show the headers are right).
- Arithmetic: false-positive denominator = scans; sensitivity denominator = rows not ignored; pooled over folds, no
  per-fold averaging; sensitivity_family == sensitivity on every row; repeat scans counted per scan (248 of the 461
  metastasis scans come from the 101 patients with several scans). An independent count (separate code) equals the
  branch's (n_gt, n_hit, n_fp) at thresholds 0.50, 0.70, 0.90, 0.95 for the three diseases. The head's
  eval_brain_disease.py run in a sandbox gives the same verdicts, including "no operating point" for the infarct dry run.
- Splits: splits_final.json reproduces from case_folds; 0 patients in train and val of one fold in all 15 folds; the
  logs of the six running folds say "Using splits from existing split file" with the expected counts; in all 314
  metastasis patient groups the sex is constant, the age span at most 5 years, the letters run in order. jobs reads
  fold_f/validation for splits[f]["val"] only. Cross run: disjoint datasets by construction; patient overlap between
  the two UCSF cohorts not checked.
- Writes: no call that deletes, renames or signals in any file of the branch.
- Not checked: the full suite; the 491 other glioma anatomy headers; a training that hangs at exit; predictions of
  250-epoch models (none exist yet).

## Findings

### Critical
None.

### Important
1. An early stop can be decided by the last step of the threshold grid, and a skip file cannot be taken back.
   Example: 2.4 FP per scan and sensitivity 0.34 at 0.90, 1.6 and 0.27 at 0.95 → operating point 0.95, flag true, skip
   file written. Steps of that size are real (S2 3d_fullres froc.csv: 0.90 → 0.95 takes sensitivity 0.314 → 0.229,
   FP 0.98 → 0.51). Smallest fix, no code: take the row with the highest threshold whose false positives exceed 2 per
   scan; write the skip file only if there is no such row or its sensitivity is under 0.3 too; otherwise report both
   rows and ask.
2. The queue reports a failed training in one line, moves on, and ends with "done" and return code 0 (verified by
   simulation: six jobs with exit code 3 → "queue empty, nothing running: done", return code 0). A fault that hits
   every start uses up all pending jobs, each leaving a log that blocks its restart. A relaunch with the documented
   command truncates queue.log. Now: require exit code 0 in queue.log for every fold evaluated; at the first non-zero
   code read that job's log, create the stop file if the cause can hit every start, report; any relaunch writes to a
   new log name. In code later: a final line with counts of succeeded, failed and refused jobs, return code 1 if any
   failed, no new start after a job that failed within ten minutes of its start.
3. The sentence states more than the record holds in two places (before Task 12 Step 1):
   (a) lesion_clause ignores host_rule and the nearest rule has no distance limit: a lesion entirely inside the left
   lateral ventricle gives "左侧大脑白质存在转移瘤样异常，体积约 64 mm³。疑似脑转移瘤。";
   (b) a study whose only predicted component scores 0.54 under a threshold of 0.55 gives "未见梗死样异常。" and
   "未见相关异常".
   Smallest fix (wording needs the user's nod): for host_rule == "nearest" write "邻近<侧><结构>（未与任何结构重叠）存在…";
   for an empty record "本模型未检出<类型>（阈值 0.xx）" and "未检出相关异常"; report the share of detections bound by the
   nearest rule beside no_host_rate.

### Minor
1. Volume floor differs by one voxel between evaluation and inference entry for two metastasis scans (the inference
   entry reads the anatomy's header: 100201B zooms (0.49999997, 0.5, 2.0) give floor 21 against 20; 100203A gives 41
   against 40). Fix: lesion_components.py:19 → math.ceil(min_mm3 / float(voxel_mm3) * (1 - 1e-6)); verified: 77 tests
   pass, none of the 1212 evaluation floors changes.
2. The side word can rest on a tenth of the lesion: a brainstem lesion with 10 % of its voxels in the left cerebellum
   gives "左侧脑干存在梗死样异常…累及小脑". Fix: no side word when the main structure has no sided label; the record
   keeps side.
3. Records are computed inside the write loop of eval_brain_disease.py. Fix: build the texts before the mkdir.
4. The operating threshold is measured on single-fold models and applied to ensembles (inference entry, cross run).
   Fix: one sentence in the cross run's REPORT.md and the README.
5. Matching keeps the assignment with the largest total IoU and drops pairs under 0.1 afterwards (pre-existing, shared
   with S2): 47 of 20000 random crowded scans give fewer hits than the best one-to-one matching, never more. Keep for
   comparability with S2; name it in the README.
6. Two build checks come after the writes (past for this run): label values are checked while labels are written; the
   expected counts after the build.
7. The guard against an axis mix-up cannot see small lesions (99 % agreement is accepted). Fix: detections raises when
   a component's smallest foreground probability is under 0.5.
8. Records carry no provenance. Fix: fields anatomy_source and model_folds.
9. Report-only lines worth adding: false positives per scan as median, maximum and number of scans over 2; false
   positives that touch labelled voxels; share of detections with score >= 0.95.

## Judgement calls for the user (the reviewer's Chinese, unchanged)

1. 达标数字是"每例误报 ≤ 2 的各行里灵敏度最高的一行"。分数是预测块内前景概率的均值，必然 > 0.5，所以阈值 0.05–0.50 是同一行，19 行里最多 10 个不同的点。nnU-Net 的分数又集中在 1 附近（S2 小病灶检测器 3d：34% 的误报、62% 的命中分数 ≥ 0.95；阈值从 0.90 到 0.95 这一步，灵敏度 0.314 → 0.229）。
2. 因此要分清两种情形。误报多的模型在 0.95 仍超过每例 2 个，就是"无工作点"，与它找到多少病灶无关（干跑的梗死模型，只训 5 个 epoch、2 例：0.95 处找到 6 个里的 3 个，每例 2.5 个误报）。误报少的模型用不满预算，达标数字就是不设阈值时的灵敏度。
3. 我会和达标数字一起报告，达标线不动：阈值 0.50 与 0.95 两行的灵敏度和每例误报；分数 ≥ 0.95 的检出与误报各占多少；每例误报的中位数、最大值、超过 2 个的扫描数；按病灶大小分层的灵敏度。
4. 均值分数反映不了"这一处是不是病灶"的把握。大块的核心体素多，分数总是高，大的误报（如术后残腔）任何阈值都去不掉。小块只有几个体素，干跑输出里从 0.65 到 0.9996 都有。提高阈值的代价主要落在 < 5 mm 这一层，转移瘤和梗死约一半的病灶在这一层。沿用均值是为了和 S2 可比，我不建议现在换。
5. 句子里的"疑似X"只说明跑的是 X 模型，不是鉴别的结果。1212 例里只有 3 例没有病灶，模型没有在正常脑上训练或测试过，每例误报是在有病的扫描上测的。所以"未见…"只表示该模型在阈值以上没有检出，不能当阴性结论。
6. 解剖词来自 SynthSeg 伪标签上的体素计票。大肿瘤的主结构可能只是 54% 对 46%（干跑例 UCSF-PDGM-0287 写成"右侧大脑皮层…累及大脑白质"）。不与任何结构重叠的病灶会被写在最近的结构上。左右取决于文件头（抽查 6 例，标签的左右与文件头一致，文件头本身未核实）。胶质瘤的"体积"是整瘤标注，含水肿。

## Tests that would not notice (each change applied to a scratch copy: 77 passed before and after)

1. The score statistic: `.mean()` → `.max()` or `.min()` in detections. Test to add: one component with 8 voxels at
   0.875 and 8 at 0.625, score 0.75.
2. Voxel volume and spacing in case_scan: `1.0` in place of voxel_mm3, or the zooms reversed. Test to add: a case_scan
   case with voxel volume 2.5 and zooms (1, 1, 2.5), a 5-voxel lesion counted, a 3-voxel one ignored, a 3-voxel
   detection dropped, and a nearest host that depends on the long axis.
3. The threshold of the records: `0.5` in place of thr in eval_brain_disease.py. Test to add: a detection at 0.625 in
   a fold 0 case that must be absent from the record, and threshold == 0.85.
Also unnoticed: `>=` → `>` in study_record and overlap_counts; the budget `<=` → `<` and IOU = 0.25 in
detection_metrics (they also pass the 29 tests of the S2 and knee evaluations); the queue ignoring the skip file or the
stop file (only the dry-run path of main is tested).

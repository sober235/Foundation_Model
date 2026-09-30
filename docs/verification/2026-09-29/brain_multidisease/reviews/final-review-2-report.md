# Whole-branch review 2: build/brain-multidisease at 1e2e95f (base 93962c9; most attention on a22da8b..HEAD)

Reviewer scratch: `/tmp/claude-1002/-home-congcongliu--claude/614ccd9c-f8e8-435b-afc8-47ae37fe3721/scratchpad/review_s7_final2/`
(`recount.py`, `compare_recount.py`, `check_records.py`, `read_sentences.py`, `derive_docs.py`, `recount_cross.py`, `mutate.sh`, and their `*.txt` outputs).
Nothing in the repository or under `/data2` was changed; every command ran read-only, CPU only, `nice -n 19`, at most 8 threads.

## Verdict

Mergeable as is. Everything the branch claims about the three detectors re-derives from the files with code I wrote
myself: on all 1212 out-of-fold scans, at every one of the 19 grid thresholds, my component/score/matching recount gives
exactly the branch's `n_hit` and false positives per scan (infarct 1208 hits / 1.556 FP per scan at 0.50, glioma 549 /
0.3493 at 0.60, metastasis 2847 / 0.5748 at 0.65), with the spec-literal one-to-one matching and with the branch's
Hungarian variant alike; the ignored counts (238 / 259 / 531), the size strata, the false-positive spread, the Dice means,
the 15 training durations, the cross-run counts (per case) and all 1212 records (folds, thresholds, lesion counts, boxes,
scores, volumes and every sentence against my own implementation of the §6 rules) agree. The 831 tests pass. Every
Important and Minor finding of review 1 is fixed in a22da8b..HEAD, except three report-only lines that were partly
added and the build-order point that was left as "past for this run". What the numbers do not show: the false-positive
budget is never reached (scores are means > 0.5), so the three gates are the un-thresholded sensitivities of these
models on their own diseased cohorts; nothing was measured on normal brains; the anatomy words are SynthSeg
pseudo-labels (NOT_EVIDENCE); the records were written with two sentence rules (largest-first, "还见于") that the spec
still marks 待用户确认; and the five-fold averaging in the cross runs rests on the command line, since nnU-Net's
prediction folder does not record the folds. The findings below are all Minor (docs, a wording/set-logic detail, three
test gaps); none changes a number.

## What was checked

Command prefix everywhere: `cd /data0/congcong/code/Project_Doing/foundation_model-multidisease && OMP_NUM_THREADS=8 PYTHONNOUSERSITE=1 PYTHONPATH=. nice -n 19 ~/anaconda3/envs/nvgen/bin/python …` (recounts used `OMP_NUM_THREADS=1` with a pool of 8).

1. **Independent recount (verified, all cases, not a spot check).** `recount.py` reads each case's `labelsTr/<case>.nii.gz`,
   `fold_<f>/validation/<case>.{nii.gz,npz}` for the fold whose `val` list holds the case, labels 26-connected
   components with `scipy.ndimage.label(structure=ones(3,3,3))`, takes the voxel volume from the label header (equal to
   `cases.json` in all 1212 cases), ignores GT components under 10 mm³, drops predicted ones under 10 mm³, scores by the
   mean of `probabilities[1]` transposed (Z,Y,X)→(X,Y,Z), and matches with my own box IoU in two ways: (a) spec §5
   literally — one-to-one on IoU ≥ 0.1 with as many pairs as possible (`scipy.sparse.csgraph.maximum_bipartite_matching`),
   and (b) the branch's rule — `linear_sum_assignment(-iou)` then drop pairs < 0.1 — both two-stage (counted GT first,
   leftovers against ignored GT, excused). Key output (`compare_recount.txt`):
   - infarct 250 cases in 3.3 s; glioma 501 in 51.3 s; metastasis 461 in 62.0 s.
   - `argmax agreement min 1.0`, `affine dev max 7.2e-06 / 0.0 / 3.0e-08`, `npz dtypes {'float32'}`,
     `pred minprob 0.50002 / 0.500004 / 0.50001` (the `PROB_TOL` check never fires on correct outputs).
   - GT components 2349 / 936 / 4340, ignored **238 / 259 / 531**, counted **2111 / 677 / 3809**; predicted components
     1824 / 860 / 3784, dropped < 10 mm³ 223 / 113 / 625; kept detections at the operating thresholds **1601 / 725 / 3156**
     (= `n_detections` of the binding sections).
   - All 19 rows of the three `froc.csv`: `diff(hu-froc) +0/+0.000000` and `diff(mc-hu) +0/+0` everywhere (n_hit and
     FP per scan). At the operating thresholds no case differs between the two matchings (0 of 1212).
   - Strata at the operating point (Hungarian) equal the REPORTs: infarct 345/1026, 508/682, 355/403; glioma 4/64, 5/36,
     540/577; metastasis 1108/1936, 1166/1283, 573/590. With the max-cardinality matching the infarct strata read
     345/1026, **509/682, 354/403** (same 1208 hits; one equal-cardinality assignment moves a hit between strata) — the
     strata attribution depends on which of several equal-size matchings is chosen; the totals do not.
   - FP spread at the operating point: infarct median 1 / max 15 / 54 scans over 2; glioma 0 / 7 / 10; metastasis 0 / 8 / 26 — equal to the REPORTs.
   - Extra numbers (not in the reports): smallest kept score 0.514 / 0.544 / 0.583; share of kept detections with score ≥ 0.95: infarct 0.626, glioma 0.726, metastasis 0.795.
   Difference from the branch's implementation: none in the counts on these data; the spec-literal matching and the
   Hungarian-then-threshold matching give the same totals at every threshold for all three diseases (review 1's
   "fewer hits in crowded scans" did not occur here).

2. **Out-of-fold integrity (verified).** `check_records.py`: for all 1212 records `model_folds == [f]` with f the fold whose
   `val` list holds the case; train ∩ val = ∅ and train ∪ val = all cases in every fold; patients (glioma with `_FU…`
   grouped, metastasis by the digits before the letter) in train and val of one fold: **0 in all 15 folds**. The
   validation folders hold exactly the val lists (file names and `summary.json` cases equal the lists, all 15 folds;
   `derive_docs.txt`), every training log has `Using splits from existing split file` and `This split has N training
   and M validation cases` equal to the split sizes, `Training done`, no traceback. My recount read exactly
   `fold_f/validation/<case>.npz` and reproduces the records' boxes and scores, so those are the files the records came from.

3. **Records vs sentences (verified, all 1212 files).** `check_records.py` rebuilds every sentence from the record's
   fields with my own code (five largest by volume, largest first; clause = place + 存在 + type + 体积 + 累及 with the
   §6 side rules; "另有 N 处" = count beyond five; "还见于" = places of the remaining lesions not among the five, first
   occurrence order, `HOST_SHORT_ZH`; nearest rule → "邻近…（未与任何结构重叠）" at ≤ 10 mm and "未能定位的区域" beyond;
   empty → "本模型未检出<类型>（阈值 0.xx）。" / "未检出相关异常"), checks threshold = operating threshold, every score ≥
   threshold, lesions sorted by score, key set, `host_sides[host] == host_side`, brainstem `host_side == midline`,
   overlap ⇒ distance 0 and fractions sum to 1, nearest ⇒ no fractions and distance > 0, no nested bracket inside
   "（还见于…）", nearest/未能定位 wording only where the rule says, and, against my recount, lesion count = kept
   components at the threshold with equal boxes, scores (1e-4) and volumes (0.05 mm³). Result: **0 violations** in
   glioma (501), metastasis (461), infarct (250). README record statistics re-derived and equal: lesions per record
   1/1/10, 0/4/69, 0/4/48; no lesion 0/3/5; more than five 2/174/101; with 还见于 1 (2) / 127 (8) / 62 (6); longest
   183/221/220 chars; nearest 6 of 725 / 9 of 3156 / 20 of 1601; unlocated 0/3/0 (= `unlocated_rate` 0.00095 × 3156).
   Note: in 2 metastasis and 10 infarct records the 5th and 6th largest lesions have equal volume; the higher score is
   named (stable sort) — deterministic, and my rebuild used the same rule, so this is a note, not a violation.

4. **Sentence quality (read, `sentences_sample.txt`, 60+ real sentences).** See Judgement calls; no rule violation.

5. **Post-review fixes (read: `git diff a22da8b..HEAD -- anatobind scripts`).**
   - Important 1 (early stop decided by the last grid step): `anatobind/eval/brain_disease.py:75-86` `beyond_budget`
     (highest-threshold row with FP > 2) and `within_reach` (`n_hit + 2 ≥ 0.3·n_gt`), `verdict` :97-110 stops only when
     the operating point is < 0.3 **and** that row is out of reach (or does not exist); otherwise `early_stop_undecided`;
     `scripts/eval_brain_disease.py:70-80` prints both rows. Does what was asked, with a stricter two-lesion margin; the
     skip file stays a manual step (STATUS §3). Never exercised: all verdicts have `beyond_budget: null`.
   - Important 2 (queue): `scripts/gpu_queue.py:119-122` a non-zero exit within 600 s drops all pending jobs; :131-134
     counts line and return code 1 unless every job succeeded; :15-19 a timestamped log name in the documented command
     (the script itself does not choose the name). Tests `test_a_training_that_fails_at_once…`, `…counts_them`,
     `…refused_job…` cover it. The real run used the old code (README deviation 7; the log's last line has no counts).
   - Important 3 (sentence): `anatobind/infer/brain_disease.py:68-74` `place` (nearest → 邻近, > `NEAR_MM = 10` →
     未能定位), :84 the bracket, :122 the empty wording and impression; `nearest_rate`/`unlocated_rate` beside
     `no_host_rate` in `binding_agreement` :136-157. Done.
   - Minor 1 floor tolerance: `anatobind/eval/lesion_components.py:20` `ceil(min_mm3 / voxel * (1 - 1e-6))`; `checks/floor_boundary.txt` shows no real case sits on the boundary. Done.
   - Minor 2 side word: `anatobind/bind/brain_lookup.py:52-54` `host_side` from the lesion's voxels inside the main structure, `host_sides` per structure; brainstem gives "midline" → no side word (verified on 100286A, 100241A, sub-strokecase0107). Done.
   - Minor 3 records before mkdir: `scripts/eval_brain_disease.py:99-104`. Done.
   - Minor 4 threshold measured on single folds: cross-run REPORT paragraph (`scripts/brain_disease_crossrun.py:123-125`) and README deviation 5. Done.
   - Minor 5 matching rule: README deviation 4. Done (and, per item 1, no effect on these data).
   - Minor 6 build checks after writes: not changed (past run; both builds' counts are asserted after the fact in `stage_raw`). Accepted, not fixed.
   - Minor 7 probability check: `anatobind/infer/brain_disease.py:44-46` (`PROB_TOL = 1e-3`). Done; test `test_the_score_is_the_mean_and_a_foreign_probability_map_is_refused`.
   - Minor 8 provenance: `anatomy_source`, `model_folds` :123-125. Done; all 1212 records carry them.
   - Minor 9 report-only lines: FP spread added (`false_positive_spread`); "false positives touching labelled voxels" and "share of detections ≥ 0.95" **not added** (my numbers for the latter are in item 1).
   - `code_version`: `anatobind/eval/brain_disease.py:23-34`, `git --no-optional-locks`, timeout 30 s, "+" only for tracked changes under anatobind/scripts. All three verdict REPORTs and the three cross-run REPORTs say `Code: commit 9a89ff1`; `git diff 9a89ff1..HEAD --stat` touches only `docs/verification/…` (22 files), so the code of the reports is HEAD's code.

6. **Cross runs (verified).** `overlap_counts` (`scripts/brain_disease_crossrun.py:36-55`): detections kept at score ≥ thr;
   "on GT" = shares a voxel with `gt_comp > 0` (fragments included); "claimed" = a counted GT component (≥ 10 mm³) that
   shares a voxel with the union of kept detections — as the README and REPORTs describe. Channel mapping `CROSS`
   (:30-32) follows the model's channel meaning (infarct DWI/ADC; metastasis T1pre/T1post/FLAIR ← PDGM T1/T1c/FLAIR;
   glioma T1/T1c/T2/FLAIR ← BMSR T1pre/T1post/T2Synth/FLAIR) and is pinned by a test. `recount_cross.py` recomputed all
   305 cases from the work-dir predictions with my own code: infarct→glioma 46/25/144/16, metastasis→glioma
   167/153/144/95, glioma→metastasis 378/164/729/222, **0 per-case differences**; the README table equals the three
   `crossrun.json` summaries and their per-case sums (`jq`). Not verifiable from outputs: that five folds were averaged —
   nnU-Net's `predict_from_raw_data_args.json` has no folds key; it rests on the command (no `--folds` → default
   `[0,1,2,3,4]` → `-f 0 1 2 3 4`) and `model_folds` written from the arguments.

7. **Docs vs files (verified by re-derivation, `derive_docs.txt`).** Timing table: all 15 durations and GPU ids from
   `logs/brain_disease/queue.log` (e.g. glioma fold 0 13:46:14→00:51:15 = 11.08 h on GPU 0), wall 16.03 h, 85.8
   GPU-hours, 15 jobs, exit codes {0}, last line `queue empty, nothing running: done`. Dice from the 15 `summary.json`
   (cases with `n_ref > 0`): 0.9279 (501), 0.8072 (461), 0.7847 (247) = README 0.928/0.807/0.785 and the REPORTs.
   Build last lines: 501/495/0, 461/314/0, 250/250/3 (infarct "3 cases without label voxels" = deviation 1);
   `cases.json` prior_surgery yes = 137 scans, 314 patients; glioma 495 patients with the 6 `_FU` cases grouped.
   Verdict table = `verdict.json`/`REPORT.md`/`froc.csv` (thresholds, sensitivities with hit counts, FP per scan, Dice,
   FP spread, strata, surgery strata 0.7496 (2839) / 0.7412 (970)); binding table = the three REPORTs (values copied,
   agreement rates themselves not recomputed — see "not checked"). S2 rows: nnU-Net 2d 0.3662 (475/1297) 1.636 @ 0.55
   and 3d 0.3678 (477/1297) 1.431 @ 0.60 from `docs/verification/2026-09-28/brain_detector/README.md:3`; nnDetection
   0.0607 (17/280) 0.7255 @ 0.50 and swept 0.2429 / 1.8431 @ 0.85 from `build/brain-nndet:docs/verification/2026-09-29/brain_nndet/fold0/REPORT.md` (see Minor 1).
   `infer_smoke.md` / `checks/infer_smoke_consistency.txt`: 1/11/26 lesions, box diff 0, score diff ≤ 1.0e-4, sentences
   equal — the smoke records exist and the three out-of-fold records `UCSF-PDGM-0008`, `100101A`, `sub-strokecase0001`
   have `model_folds [0]` and the same sentences (read; the consistency script was not re-run). Evaluation, cross-run
   and smoke clock times in the README are within 1 s of the files' mtimes (09:55:27 / 09:57:02 / 09:57:23;
   10:04:30 / 10:15:05 / 10:32:51; 09:59:33 / 09:59:53 / 10:00:15). `checks/*.txt` read: floors, grids, cross-run inputs, host voxels as summarised in `checks/README.md`.

8. **Safety (verified by grep + read).** `git diff 93962c9..HEAD | grep` for unlink/rmtree/remove/rename/replace/
   truncate/shutil/kill/signal/terminate/rm/mv: only test fixtures under `tmp_path` (one `.rename` of a fixture, empty
   `write_text("")` fixtures) and string `.replace`. Write sites (listed in the transcript): `build_raw` refuses an
   existing dataset folder, `write_splits` an existing `splits_final.json`, `eval_brain_disease.py` an existing `--out`
   or `--records`, `brain_disease_crossrun.py` an existing `--work` or `--out` (and checks checkpoints and channel files
   first), `infer run()` an existing `--out` (grid check before `link_inputs`), the queue refuses a job whose result
   folder or log exists (so `> log` never truncates), `LOG_DIR.mkdir(exist_ok=True)`. Targets: `nnUNet_raw`/
   `nnUNet_preprocessed` (`derived/nnunet`), `derived/brain_disease/…`, the worktree's `docs/` and `logs/`. `code_version`
   runs git with `--no-optional-locks`. Nothing signals a process.

9. **Tests (verified).** `… -m pytest tests/ -q -p no:cacheprovider` → `831 passed, 1 skipped in 88.37s`. Mutation probe
   (`mutate.sh`: a fresh `git archive HEAD` export per mutation, the mutated file written from `git show`, targeted test
   file run): 26 mutations, 22 caught, 1 uninformative (my sed produced a syntax error), **3 not caught** — see the last section.

Not checked: the binding agreement rates (host/side/host+side) were not recomputed with an independent binder; the
five-fold averaging of the cross runs (above); `checks/*.py` and `infer_smoke_consistency.py` were not re-run (outputs
read); patient overlap between the two UCSF cohorts; the correctness of SynthSeg's left/right (spec M10 assumption).

## Findings

### Critical
None.

### Important
None.

### Minor
1. **README cites a file that is on neither this branch nor main.** `docs/verification/2026-09-29/brain_multidisease/README.md:35`
   names `docs/verification/2026-09-29/brain_nndet/fold0/REPORT.md` as the source of the two nnDetection rows; that
   file exists only on `build/brain-nndet` (commit 3a8a431; `git ls-tree main` has no `fold0/`). After this branch is
   merged and before the nnDetection branch is, the link is dead and the two rows (0.0607 @ 0.7255, 0.2429 @ 1.8431)
   cannot be checked from main. Fix: add "(branch `build/brain-nndet`, 3a8a431, not yet on main)" after the path, or
   merge the nnDetection branch first. The numbers themselves are right.
2. **"还见于邻近X" is written although X is already named.** `anatobind/infer/brain_disease.py:114-115` compares
   `place()` strings, so "邻近左侧大脑皮层" (a nearest-rule lesion) counts as a place different from "左侧大脑皮层". Real
   records: sub-strokecase0107, 0153, 0193, 0208 ("还见于…邻近左侧大脑皮层" while a 左侧大脑皮层 lesion is among the
   five), sub-strokecase0248 ("还见于邻近左侧大脑白质"), 100154D, 100180A. A reader learns nothing new from these places.
   Smallest fix (wording call for the user): build `named` and `also` from the place without the "邻近" prefix
   (`place(l, HOST_SHORT_ZH).removeprefix("邻近")`), or leave nearest-rule lesions out of "还见于". Records would need regenerating into a new folder.
3. **README line 9 over-generalises the code stamp.** "all evaluation reports carry `Code: commit 9a89ff1`" is true of the
   six reports whose numbers the README copies; `infarct_fold0/REPORT.md` has no `Code:` line (it predates 565e75a) and
   `metastasis_fold1_preliminary/REPORT.md` says `7c4ae27`. Fix: "the three verdict reports and the three cross-run reports carry …".
4. **`logs/` is untracked and not ignored** (`git status` → `?? logs/`; `.gitignore` has no `logs` entry). The README says
   the queue and training logs are "not in the repository"; a `git add -A` at merge time would add 1.7 MB of nnU-Net logs.
   Fix: one line `logs/` in `.gitignore` (or add `logs/brain_disease/` deliberately).
5. **Review-1 Minor 9 only partly applied**: the FP spread is reported; "false positives that touch labelled voxels" and
   "share of detections with score ≥ 0.95" are not. Not required by the spec; my values for the latter are 0.626 / 0.726 / 0.795 (infarct / glioma / metastasis).
6. **Records written before the user's confirmation of two sentence rules.** Spec §6 (line 133) still marks
   "largest five by volume" and "还见于" 待用户确认; README line ~"Sentence rules" says "implemented as recommended";
   STATUS §2 items 3–4 are open. Not a code defect; but `eval_brain_disease.py:40-42` refuses an existing `--records`
   folder and the project deletes nothing, so a change of wording means a second folder (e.g. `records_v2`) and a README note.
7. **Task 13 not done on this branch** (expected: this review is part of it): `STATUS.md` and `CLAUDE.md` still describe the
   19:40 pause ("进行中", trainings running); they need the final verdicts, the record folder, the tag and the pointer to this report before the merge.

## Judgement calls for the user（中文）

1. 三个病种都过线，但预算没用满：分数是块内概率均值，必然大于 0.5（实测最小 0.514 / 0.544 / 0.583），每例误报最多 1.556，所以"达标数字"就是这三个模型在各自病人队列上不设阈值的灵敏度。梗死 0.572 只比 0.5 高 0.07；小于 5 mm 的梗死只找到三分之一（345/1026），转移瘤一半多（1108/1936）。
2. 句子先写最大的五处、后面补"还见于"，这两条规格里仍标"待用户确认"，1212 条记录已经按建议写好。要改写法，只能写到新目录（脚本拒绝覆盖，项目不删文件），README 里记一笔。
3. 主结构按体素多数投票，大肿瘤会被写到一个占比只有 29% 的结构上。例：UCSF-PDGM-0483 写成"双侧丘脑存在肿瘤样异常，体积约 116 mL，累及左侧大脑白质、左侧大脑皮层、左侧基底节"（丘脑 28.6%、白质 27.0%、皮层 22.3%）。放射科医生读到"丘脑肿瘤 116 mL"会觉得不对；实际是左侧大半球的大肿瘤累及丘脑。建议主结构占比低于某个值（如 40%）时改写为"累及…多个结构"，或把占比写进句子。这是措辞判断，不是代码错。
4. "双侧"落在很小的病灶上：100196A（小脑，14 mm³）、100196B（17.7 mm³）、sub-strokecase0244（16 mm³）、100310C（白质，87.5 mm³）——两三个体素正好跨在 SynthSeg 左右小脑或胼胝体的分界上，句子写"双侧小脑"。读者会以为是两侧多发。建议小于某体积的病灶不写"双侧"，写"中线旁"或只写结构。
5. "双侧"前缀按主结构内的体素算，整个病灶的 side 另存。UCSF-PDGM-0174："双侧大脑白质…累及左侧大脑皮层"，而记录里整个病灶 side = left。两种口径都在记录里，句子用的是用户 09-29 定的那一种，读的时候要知道这一点。
6. "还见于邻近左侧大脑皮层"——前五处已经写了左侧大脑皮层，再补"邻近左侧大脑皮层"没有新信息（7 条记录，见 Minor 2）。要不要去掉"邻近"类位置，或去掉前缀再比较，由用户定。
7. "未能定位的区域"：100126B 有一处 3.5 mL 的检出离任何结构 29 mm，另一处 18 mm³ 离 66 mm。前者很可能在术腔或伪标签标成脑脊液的地方，后者大概在颅外。写"未能定位"是老实的，但对读者没用；可以考虑把最近结构和距离一起写出（"距左侧小脑约 29 mm"）。
8. "疑似 X"只说明跑的是 X 模型。交叉运行给出了第一份证据：转移瘤模型在胶质瘤上 167 个检出里 153 个落在肿瘤上，胶质瘤模型在转移瘤数据上每例 3.6 个检出。阶段 B 之前，这个印象不能当鉴别。
9. 交叉运行用的是五折平均模型，但阈值是单折模型上测的，而且 nnU-Net 的输出目录不记录用了哪几折；"五折平均"只能从命令行推出。要留证据的话，下次把 `-f` 参数原样写进 crossrun.json（现在写的是 `model_folds`，来自同一个参数，够用但不是 nnU-Net 自己的记录）。
10. 匹配规则：在这 1212 例上，规格写的"IoU ≥ 0.1 一对一、尽量多配"与分支用的"总 IoU 最大再截 0.1"在每个阈值上命中数、误报数完全相同；只有分层表会因等价配法不同而挪动一处（梗死 5–10 mm 层 508 或 509）。不必改代码，但分层数字有 ±1 的口径余地。
11. 时间表里胶质瘤 fold 0 11.08 h、fold 4 4.95 h，差两倍多，是别的会话共用显卡造成的，不是数据；用这些时间做预算要按 5 小时算，不按 11 小时。

## Tests that would not notice (mutation probe, each on a scratch export of HEAD; the rest of the 26 mutations were caught)

1. **`named` built with the long names** (`anatobind/infer/brain_disease.py:114`: `place(l, HOST_SHORT_ZH)` → `place(l)`):
   `tests/test_brain_disease_record.py` 15 passed. Effect: a deep-grey lesion among the five would be repeated in
   "还见于左侧深部灰质". Smallest test: five named lesions, one of them `other_deep_grey` (left), plus a sixth
   `other_deep_grey` (left) → the sentence must end "；另有 1 处同类异常。疑似缺血性梗死。" with no "还见于".
2. **Cross-run union overwritten instead of OR-ed** (`scripts/brain_disease_crossrun.py:48`: `det_any[sl] |= m` →
   `det_any[sl] = m`): `tests/test_brain_disease_crossrun.py` 7 passed. Effect: when a later detection's box covers an
   earlier detection's voxels, the earlier one's mark is erased and the lesion it claimed is no longer counted as
   claimed (common in crowded metastasis scans). Smallest test: detection 1 on lesion A (box 2:4,2:4,1:4), detection 2
   off A but with a box that contains A's box (e.g. a thin L-shaped component spanning 1:9,1:9,1:4 whose voxels avoid
   A) → `n_gt_claimed` must be 1.
3. **Record fold fixed to [0]** (`scripts/eval_brain_disease.py:102`: `[fold_of[s["case"]]]` → `[0]`):
   `tests/test_brain_disease_eval_script.py` 6 passed. Effect: every record would say `model_folds [0]`. Smallest test:
   in `test_fold_report_records_and_refusals` (two folds), assert the record of a fold-1 validation case has `model_folds == [1]`.
4. Not testable by mutation here but worth a test: `also_set_order` (order of the "还见于" places) — my mutation did not compile; the existing test `test_the_lesions_that_are_only_counted_still_name_their_places` does pin the order of five places, so a `set()` would most likely be caught.

Caught by existing tests (for the record): sentence order by score or smallest-first; "还见于" without the `not in named`
filter; `others` taken from the score order; "另有 N" off by one; `> NEAR_MM` → `>=`; `INVOLVED_MIN` 0.05; `within_reach`
`>=`→`>`, `REACH_LESIONS` 3, `reach` when no row is beyond the budget, `beyond_budget` `>`→`>=`, `low` `<`→`<=`, stop
ignoring reach; cross-run "on GT" counted GT only, "claimed" over ignored rows, `>=`→`>` on the threshold, counting
voxels instead of detections; queue return code always 0, fast-failure inverted; probability check disabled; floor
tolerance removed; record threshold fixed at 0.5.

# Review of the brain multi-disease records v2 (AnatoBind-MRI, code c8b1a11) — 2026-09-30

Saved by the controller from the reviewer's hand-back (most capable model; the harness refused the reviewer's own report
file, so this message is the report; the reviewer's scripts and raw outputs were in the session scratch:
`check_v2.py/.out`, `diff_v1_v2.py/.out`, `sample_sentences.txt`, `run_mutations.sh`, `mutations.out`, `mut_impact.py/.out`).
Read-only review; nothing in the repository or under /data2 was changed.

## Verdict

**The v2 records are fit to replace v1 as the delivered records.** Every one of the 1212 v2 sentences is reproduced
character for character by an independent implementation of the v2 rules written from the spec text, and by the
repository's own `study_record` at c8b1a11 fed with each record's lesion fields (whole-record equality, 0 mismatches).
All 26 changed sentences are what the four rules intend. Apart from the sentences, the only v1→v2 field changes are the
side fields of 5 lesions, all explained by rule 2. `verdict.json`, `froc.csv` and `output.txt` are byte-identical to v1
for all three diseases. No High or Medium finding. Low findings: test-coverage gaps (four mutations pass the suite, two
of which would change real sentences), one latent wording path no record triggers, and a handful of wording judgement
calls (§7). None is a reason to hold v2 back.

## 1. Programmatic checks over all 1212 v2 records

Results: SENTENCE_MISMATCH: 0 of 1212 (independent implementation) and 0 of 1212 whole-record mismatches (repo oracle).

| | glioma | metastasis | infarct |
|---|---|---|---|
| records / with lesions / without | 501 / 501 / 0 | 461 / 458 / 3 | 250 / 245 / 5 |
| lesions | 725 | 3156 | 1601 |
| records with "另有" / with "还见于" | 2 / 1 | 174 / 127 | 101 / 59 |
| "跨多个结构" clauses (= sentences) | 5 | 2 | 6 |
| nearest-rule lesions / of which > 10 mm | 6 / 0 | 9 / 3 | 20 / 0 |
| sentences with "距最近的" | 0 | 2 | 0 |
| lesions with `side` = bilateral / `host_sides` entries bilateral | 3 / 14 | 8 / 12 | 3 / 7 |
| "双侧" words (clause head / 累及 / 还见于 / 跨 bracket) | 6 / 1 / 0 / 1 | 5 / 4 / 2 / 0 | 3 / 0 / 0 / 1 |

Specific properties, verified against the record fields:
- 跨多个结构: all 13 clauses have `host_rule == overlap` and max(host_fractions) < 0.4 (range 0.2863–0.3989) and list
  exactly the structures with share ≥ 0.10, largest first; the two with a bilateral structure (UCSF-PDGM-0483 thalamus,
  sub-strokecase0140 cerebellum) give every structure its own side and no common side word; the other 11 carry one
  common side word. No 跨 clause has a 累及 tail.
- 双侧: every "双侧" maps to `host_side == bilateral` or `host_sides[h] == bilateral` of a listed structure. Per-side
  voxel counts are not in the records, so the necessary condition volume × share ≥ 20 mm³ was checked: holds for all
  33 bilateral `host_sides` entries and all 14 bilateral `side` entries. Smallest supporting lesion: metastasis 100310C
  #38, 87.5 mm³ white matter straddling the midline.
- 还见于: never repeats a place named among the five (with or without 邻近), never repeats itself, every listed place
  belongs to a counted lesion, every counted lesion's place is either named among the five or listed.
- 距最近的: appears in exactly the two clauses with `host_rule == nearest` and `host_distance_mm > 10` (100126B #5
  29.39 mm → "约 29 mm"; 100197A #2 10.17 mm → "约 10 mm"). The third far lesion (100126B #21, 66.06 mm, 18.2 mm³) is a
  counted one, suppressed because "未能定位的区域" is already named among the five. "邻近" appears in 22 clauses, all
  nearest rule, ≤ 10 mm (max 7.83 mm), all with "（未与任何结构重叠）".
- Field sanity (all pass): lesions by score descending, all ≥ threshold; `impression` = disease's or "未检出相关异常";
  `host` = argmax of fractions; fractions sum to 1 ± 0.002; overlap ⇒ distance 0; nearest ⇒ fractions {} and
  distance > 0 and `host_sides` = {host}; `host_side == host_sides[host]`; brainstem always midline.

## 2. v1 → v2 diff

Same 501/461/250 file names. Record-level: only `sentence` changed, in 5/8/13 = 26 records; `threshold`,
`model_folds`, `study`, `disease_model`, `impression`, `anatomy_source` unchanged in all 1212. Lesion-level: counts,
`type`, `score`, `box`, `volume_mm3`, `host`, `host_rule`, `host_fractions`, `host_distance_mm` identical in all 5482
lesions. Side fields changed in 5 lesions:

| lesion | fields | v1 → v2 | why |
|---|---|---|---|
| metastasis 100196A #23, cerebellum 14.4 mm³ | side, host_side, host_sides | bilateral → left | smaller side ≈ 6.7 mm³ < 10 |
| metastasis 100196B #20, cerebellum 17.7 mm³ | side, host_side, host_sides | bilateral → left | smaller side < 10 mm³ |
| infarct sub-strokecase0244 #10, cerebellum 16.0 mm³ (2 voxels of 8 mm³) | side, host_side, host_sides | bilateral → right | 1 + 1 voxel, 8 mm³ < 10; tie → "right" |
| metastasis 100241A #1, brainstem 1452 mm³ | host_sides.white_matter only | bilateral → left | WM share 2.5 %, never written |
| infarct sub-strokecase0003 #4, cortex 28.6 mL | host_sides.cerebellum only | bilateral → right | cerebellum share 0.06 %, never written |

The 26 changed sentences (rule each applies; all verified against the fields; all intended): UCSF-PDGM-0392, -0402,
-0433, -0433_FU007d, -0483 (rule 1); 100126B, 100197A (rule 4); 100154D, 100180A, sub-strokecase0107, -0110, -0153,
-0193, -0208, -0248 (rule 3); 100196A, 100196B, sub-strokecase0244 (rule 2); 100263A, 100300D, sub-strokecase0039,
-0065, -0094, -0131, -0140, -0168 (rule 1).

Reports: `froc.csv`, `verdict.json`, `output.txt` byte-identical v1 ↔ v2 for all three diseases. `REPORT.md` differs
only in the command line and code stamp for glioma and infarct; metastasis additionally in the binding block:
side_agreement 0.99719 → 0.99684, host_side_agreement 0.99895 → 0.99860 (one matched pair each of 2846).

## 3. Reading the sentences

Read: all 26 changed sentences, the 7 boundary cases, and 60 unchanged records. All 60 unchanged ones are grammatical
and unambiguous. A scan of all 1212 sentences found no "，，", "、、", "（）", empty 还见于, zero volume, single-item 跨
bracket or nested bracket. Wording points, all Low:
- (a) 100197A, the 10 mm boundary: "未能定位的区域（距最近的左侧大脑皮层约 10 mm）" for 10.17 mm beside "邻近右侧大脑皮层
  （未与任何结构重叠）" for 7.83 mm.
- (b) A common side word before a bracket that contains the brainstem: "跨左侧多个结构（深部灰质、脑干、丘脑）" (100300D),
  "跨左侧多个结构（大脑白质、丘脑、脑干）" (sub-strokecase0131) can be read as "左侧脑干".
- (c) "跨多个结构" for tiny lesions: 100263A 93 mm³, 100300D 139 mm³ (pseudo-label junctions). Rule 1 has no volume floor.
- (d) The 0.4 threshold sits on real data (0.3989, 0.3986, 0.3924): the form can flip between follow-ups.
- (e) 深部灰质 without its gloss in 20 sentences (short form inside brackets).
- (f) sub-strokecase0168: the 还见于 key of a spanning lesion is its argmax place only, so a structure inside the 跨
  bracket can reappear in 还见于 (same convention as 累及).
- (g) Pre-existing (v1 = v2): sub-strokecase0201 two brackets back to back; 35 sentences with a bracket inside a 、-list.
- (h) UCSF-PDGM-0483 reads correctly and more clearly than v1. (i) sub-strokecase0094 reads well. (j) 100310C #38
  stays 双侧 at 87.5 mm³ (midline white matter; a labelling-convention question). (k) Ties under the floor become 右侧
  (`side_of` tie rule), visible in record fields only. (l) 100197A #1 "脑干…累及双侧大脑白质、双侧大脑皮层" is
  anatomically implausible (pseudo-label around the brainstem); NOT_EVIDENCE as marked.

## 4. Code review of c8b1a11

Small diff, matches spec §6 amendments. Edge cases checked: empty fractions (short-circuit), host None, ties in
shares (stable sort, argmax heads the list), `removeprefix` (py3.11), `_sided` ordering, `main_side` midline/bilateral
in the 跨 branch. Latent: the far-lesion bracket used the long name `HOST_ZH` (would nest a bracket for the deep grey
matter; no record triggers it) — fixed by the controller afterwards. `side_of` floor uses MIN_MM3; tie rule "right"
pre-existing but newly reachable. Tests at HEAD: 28 passed.

## 5. Mutations the new tests would not catch

| mutation | suite | v2 records whose sentence would change |
|---|---|---|
| `side_of` floor `>=` → `>` | pass | unknown without voxel data |
| floor dropped from `host_side`/`host_sides` but kept for `side` | pass | 100196A/B, 0244 host_side and two host_sides-only flips |
| `MAIN_SHARE_MIN` 0.4 → 0.39 / 0.36 | pass | 3 / 9 |
| tie → "left" | pass | fields of 0244 #10 and 0003 #4 |
| 跨 branch takes `side` instead of `host_side` | pass | 0 |
| named-place set keeps the 邻近 prefix for the five | pass | 0 |
Controls caught: `removeprefix` dropped, brainstem-first order dropped, 邻近 dropped from 还见于 words.

## 6. Other observations

Not done: no re-binding from the label maps (side flips checked by necessary conditions and voxel-size inference only),
no GPU, nothing written outside the scratch directory.

## 7. 待用户拍板的写法问题（都是小事，不挡交付）

1. 100197A：10.17 mm 写成"约 10 mm"，却归入"未能定位"。建议四舍五入等于 10 时写一位小数（"约 10.2 mm"），或写"超过 10 mm"。
2. "跨多个结构"的括号里有脑干时，前面的"左侧"会被读到脑干上（100300D、sub-strokecase0131）。可改为每个结构各带侧别、脑干在前。
3. 很小的病灶也写"跨多个结构"（100263A 93 mm³、100300D 139 mm³）。要不要给规则①加体积下限（比如 < 1 mL 仍写主结构），或改写成"位于…交界处"。
4. "跨多个结构"的病灶，"还见于"只按它占比最大的结构去重（sub-strokecase0168）。要不要把括号里的结构都算作已命名。
5. 左右各一个体素、正好骑在中线的病灶，现在记为"右侧"（side_of 平局取右）。要不要平局记 midline。
6. 顺手可改：距最近结构的括号里的长名（已改）；sub-strokecase0201 两个括号相连（v1 就有）。

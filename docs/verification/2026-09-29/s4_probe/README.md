# S4 (brain anatomy on fastMRI FLAIR): read-only probes during the design (2026-09-29)

S4 is in design; nothing here is a result of a model. Three probes read existing files and write only into this folder
(the montages go to `~/figs/foundation_model/s4probe/`). Every `.txt` holds the command and its unedited output.

| probe | question | answer |
|---|---|---|
| `sibbms_headers` | What does `/data2/congcong/data/FM_data/SibBMS_ms/sibbms.zip` hold (11 GB, not extracted)? | 100 healthy subjects (T1w, T2w, FLAIR) and 93 patients with multiple sclerosis in 272 sessions (plus T1w after contrast). Every volume is 197 x 233 x 189 at 1 mm, RAS, about three quarters of the voxels are zero: skull-stripped and in a template space. The members were read from the archive into memory; nothing was extracted. |
| `skullstrip_probe` | Is the outline of SynthSeg's label map (labels > 0) on the 447 annotated fastMRI FLAIR volumes a usable skull-strip? | In the middle slices yes. Volume inside the outline: median about 720 mL, 5th percentile about 437 mL. The line "share of the outline in the outer 6 voxels of the head" of this script is not meaningful (the erosion is three-dimensional and the volumes have 16 slices); read the montages and `skullstrip_slices` instead. |
| `skullstrip_slices` | The same per slice. | 14 volumes under 300 mL (SynthSeg failed or nearly so, one label map is empty). The filled outline covers a median of 0.47 of the head's cross-section in slice 0, 0.56–0.58 in slices 2–6, 0.28 in slice 11. In 167 of 433 volumes slice 0 or 1 covers under 80 % of what slices 2–6 cover; this is an indication only, because the head's cross-section is larger at the level of the face. Holes in the outline: up to 16 % of the filled outline in slice 0 (95th percentile), none from slice 5 up. |

What the montages show (`skullstrip_file_brain_AXFLAIR_203_6000900.png`, `…_200_6002487.png`): the outline follows the
brain's surface in the middle slices; in the lowest slice it can leave out the posterior quarter of the brain or label
the deep centre as background; the vertex is covered in part.

Consequence for the design: the outline cannot be used as it is to strip the skull; the proposal put to the user is a
brain-outline model of our own, trained on the middle slices, with the lowest two slices and the vertex left out of
the loss.

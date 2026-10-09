# SibBMS: why no row supervises U (spec §3.1, plan Task 13 Step 4)

Checked 2026-10-08 by the controller (read-only):

- The lesion annotations of SibBMS live in `SibBMS_ms/sibbms/Output/Annotation/sub-*/ses-001/` and exist for **10 subjects
  only** (`sub-037, 039, 043, 044, 045, 047, 050, …`; `ls Annotation | wc -l` = 10). Each folder holds its own
  `sub-XXX_Segmentation-label.nii.gz` with `sub-XXX_flair.nii.gz`, `sub-XXX_t1.nii.gz`, `sub-XXX_t1c.nii.gz`,
  `sub-XXX__t2.nii.gz`.
- Their grid is native (`sub-047_Segmentation-label.nii.gz`: 201 x 261 x 261 at 1 mm), not the 197 x 233 x 189 template
  grid of the FLAIR / T1w / T2w used for A (`MS/sub-037/ses-001/anat/sub-037_ses-001_FLAIR.nii.gz`: 197 x 233 x 189).
- The 358 SibBMS sessions in the sample table (265 MS, 93 healthy) therefore carry `u_supervised = False` and
  `lesion = None`: a plaque that is present but unlabelled must not be a negative. They supervise A and S only.
- The 10 annotated subjects stay available as a small external U check for Part 2 (registration or a run of SynthSeg on
  their native T1 would be needed first); they are not used in Part 1.

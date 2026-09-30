# S4 probe 2: geometry and brain coverage of the fastMRI AXFLAIR stacks (2026-09-30, read-only)

Input for the "simulation" section of the S4 design: how a fastMRI FLAIR stack sits on the brain, measured on the
SynthSeg pseudo-labels of the 447 annotated AXFLAIR volumes (`derived/synthseg/fastmri_brain/seg_native`). The per-slice
class table is an indication only (the lowest slices under-label the deep grey matter, see `../../2026-09-29/s4_probe`);
the brain-area profile does not depend on which label a voxel gets. Script and unedited output: `fastmri_flair_coverage.{py,txt}`.

- Geometry: 16 slices in 417 volumes, 14 in 27, 12 in 3; slice spacing 5.0 mm everywhere; in-plane 0.688 mm (329
  volumes; matrices 320 × 320 or 260 × 320) or 0.86 mm (81; 276 × 276 or 320 × 320).
- Coverage: the lowest slice holds a cross-section of 83 % of the stack's largest one (median; 5th–95th percentile
  59–97 %), i.e. the stack starts around the upper cerebellum / basal ganglia level (cerebellum present in 59 % of the
  lowest slices, basal ganglia in 81 %), reaches its widest section 15 mm higher, and ends above the vertex: brain is
  present over 65 mm (median; 51–70 mm) and 2–3 slices at the top are empty in 348 of 447 volumes (4 or more in 78, 0–1
  in 20). Brainstem and lower cerebellum are outside the field of view in most volumes.

# S4 inference smoke (plan Task 9 Step 9; spec §8) — NOT_EVIDENCE

Run 2026-10-08 09:47–09:48 on GPU 7, code f220389 (`scripts/infer_brain_anatomy.py`, chain `anatobind/infer/brain_anatomy.py`:
outline model Dataset908 2d → skull-stripped stack → student Dataset907 3d_fullres without mirroring → SynthSeg values).
Outputs under `/data2/congcong/data/FM_data/derived/brain_anatomy/infer_smoke/{svd,lesion}` (`record.json`, `anatomy.nii.gz`,
`brain_mask.nii.gz`, the two nnU-Net input/output folders).

## The registry lesion used for the box

```
$ PYTHONNOUSERSITE=1 PYTHONPATH=. nice -n 19 ~/anaconda3/envs/nvgen/bin/python - <<'EOF'
from anatobind.level_r.registry import load_registry
r = sorted(load_registry(), key=lambda r: (r["file"], r["lesion_id"]))[0]
print(r["file"], r["lesion_id"], "--box", r["x0"], r["y0"], r["z0"], r["x1"], r["y1"], r["z1"] + 1)
EOF
file_brain_AXFLAIR_200_6002425 0 --box 91 182 2 98 188 3
```

Registry row: label "Nonspecific white matter lesion", one slice (z 2), `host_lookup_all` = `host_lookup_parenchyma` = 41
(SynthSeg right cerebral white matter).

## S2's smoke volume

```
$ PYTHONNOUSERSITE=1 PYTHONPATH=. nice -n 19 ~/anaconda3/envs/nvgen/bin/python scripts/infer_brain_anatomy.py --h5 /data2/congcong/data/FM_data/fastMRI_lh_brain_knee/kspace/brain/multicoil_train/file_brain_AXFLAIR_201_6002917.h5 --out /data2/congcong/data/FM_data/derived/brain_anatomy/infer_smoke/svd --gpu 7
brain 728.1 mL; reliable slices [2, 12]; anatomy -> /data2/congcong/data/FM_data/derived/brain_anatomy/infer_smoke/svd/anatomy.nii.gz
```

`record.json`: shape [320, 320, 16], spacing [0.6875, 0.6875, 5.0] mm; class volumes (mL): white matter L 140.5 / R 127.1,
cortex L 141.3 / R 135.2, thalamus L 4.9 / R 5.0, basal ganglia L 7.6 / R 6.5, brainstem 0.05, cerebellum L 0.0 / R 0.009,
other deep grey L 0.31 / R 0.06, ventricles 50.3.

## The lesion volume with the registry box

```
$ PYTHONNOUSERSITE=1 PYTHONPATH=. nice -n 19 ~/anaconda3/envs/nvgen/bin/python scripts/infer_brain_anatomy.py --h5 /data2/congcong/data/FM_data/fastMRI_lh_brain_knee/kspace/brain/multicoil_train/file_brain_AXFLAIR_200_6002425.h5 --out /data2/congcong/data/FM_data/derived/brain_anatomy/infer_smoke/lesion --gpu 7 --box 91 182 2 98 188 3
brain 778.9 mL; reliable slices [2, 11]; anatomy -> /data2/congcong/data/FM_data/derived/brain_anatomy/infer_smoke/lesion/anatomy.nii.gz
box [91, 182, 2, 98, 188, 3]: host white_matter (overlap), side right, fractions {'white_matter': 0.7143, 'cortex': 0.2857}
```

`record.json`: class volumes (mL): white matter L 163.2 / R 161.3, cortex L 160.3 / R 167.9, thalamus L 5.2 / R 5.8,
basal ganglia L 4.6 / R 5.5, brainstem 0.12, cerebellum L 0.0 / R 0.06, other deep grey L 0.14 / R 0.23, ventricles 9.6.

Binding of the box, student anatomy against the SynthSeg map of the same stack (one Python call each, 2026-10-08 09:49):

| map | BrainBinder: host, rule, side | host fractions | SynthSeg values inside the box |
|---|---|---|---|
| S4 student (`anatomy.nii.gz`) | white_matter, overlap, right | white_matter 0.714, cortex 0.286 | {41: 30, 42: 12} |
| SynthSeg (`seg_native/file_brain_AXFLAIR_200_6002425_seg.nii.gz`) | white_matter, overlap, right | white_matter 0.667, cortex 0.333 | {41: 28, 42: 14} |

`BrainLookup` (the registry's rule) on the SynthSeg map: `BRAIN_ALL` (41, 0.667), `BRAIN_PARENCHYMA` (41, 0.667) — the
registry's `host_lookup_all` = 41. Same host, same side, same rule on both maps; the two maps differ in 2 of 42 voxels of
the box. All of this is agreement between two pseudo-label maps: NOT_EVIDENCE.

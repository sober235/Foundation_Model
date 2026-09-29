# Read-only checks on the real data (controller, 2026-09-29)

Five questions came up in the task reviews of the brain multi-disease plan that the unit tests cannot answer, because
tests never read `/data2`. Each was answered by a scan of the real files. The scans read headers or label maps only and
write nothing outside this folder. Every `.txt` holds the command and its unedited output.

Run from the repository root, one at a time (each uses one CPU process under `nice -n 19`):

```
PYTHONNOUSERSITE=1 PYTHONPATH=. nice -n 19 ~/anaconda3/envs/nvgen/bin/python docs/verification/2026-09-29/brain_multidisease/checks/<name>.py
```

| check | question (where it came from) | result | time |
|---|---|---|---|
| `floor_boundary` | Can a float32 header value tip the 10 mm3 floor by one voxel? (review of Task 4) | No. Glioma: 501 cases at exactly 1.0 mm3, floor 10 voxels. Metastasis: voxel volume 0.18–2.30 mm3, floors 5 to 55 voxels; the four cases where 10 / volume is an integer have spacings 0.5, 1.0 or 2.0, exact in float32. Infarct: floors 1 (30 cases), 2 (218), 7 (2); none near an integer. | 10 s |
| `anatomy_grid` | Does every case have its SynthSeg map, on the label's grid? (`case_scan` compares shapes only; review of Task 8) | Yes: 501 / 461 / 250 cases, none missing, none with another shape, largest affine deviation 0. | 2 s |
| `channel_grid` | Would the grid check of the inference entry (shape and affine within 1e-3) refuse any of our own data? (fix round of Task 7) | No: all 3887 channel files have exactly the affine of their case's SynthSeg map. | 9 s |
| `crossrun_inputs` | Do the channel files of the three cross runs exist, on the grid of the label the overlap is counted against? (review of Task 10) | Yes: 100, 100 and 105 cases, no file missing, none off the grid, largest affine deviation 8.5e-08. | 5 s |
| `host_voxels` | Can a lesion of these datasets get no host structure at all, so that two unbound rows would count as agreeing? (review of Task 8) | No: no SynthSeg map is without host voxels; the smallest counts are 907612 (glioma), 436980 (metastasis), 57647 (infarct). | 10 min |

What these checks do not show: that the SynthSeg maps are anatomically right (they are pseudo-labels, and lesions
displace tissue), or that the headers' left and right are the patient's (spec M10 states this as an assumption).

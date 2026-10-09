# ISLES: SynthSeg on the DWI grid is kept as A supervision (spec §12 P1; plan Task 13 Step 3)

Evidence (`isles_check.txt`, `isles_check/host_volumes_median_ml.csv`, median over the first 30 cases of each source):

| host | PDGM | BMSR | ISLES | ISLES / PDGM |
|---|---|---|---|---|
| white matter L / R | 226.8 / 223.2 | 240.0 / 243.2 | 206.8 / 205.0 | 0.91 / 0.92 |
| cortex L / R | 279.0 / 274.9 | 259.6 / 268.4 | 234.1 / 228.6 | 0.84 / 0.83 |
| thalamus L / R | 7.1 / 7.7 | 6.4 / 6.9 | 5.9 / 6.1 | 0.83 / 0.79 |
| basal ganglia L / R | 10.4 / 11.3 | 10.3 / 10.7 | 9.5 / 9.8 | 0.91 / 0.87 |
| brainstem | 24.6 | 24.1 | 20.5 | 0.83 |
| cerebellum L / R | 67.7 / 68.4 | 68.9 / 67.2 | 58.4 / 57.4 | 0.86 / 0.84 |
| other deep grey L / R | 10.1 / 9.7 | 9.7 / 9.7 | 7.7 / 8.0 | 0.76 / 0.82 |

Every ISLES median lies within 0.76–0.92 of PDGM's, inside the 30 % band the plan set (ISLES patients are older stroke
patients at 2 mm voxels; smaller volumes are expected). Montages (USER_REPORTED, the controller looked at
`isles_sub-strokecase0001.png` and `0004.png`): the host classes follow the DWI anatomy — cerebellum and brainstem on
the low slices, ventricles, thalamus and basal ganglia at the mid level, white matter / cortex with the right sides;
the maps are coarser than on 1 mm T1 (2 mm voxels) but not misplaced.

Ruling: ISLES rows keep `a_supervised` (A and S) in addition to U on DWI. NOT_EVIDENCE: a pseudo-label compared with
other pseudo-labels.

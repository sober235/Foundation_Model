# Level R pilot 150-lesion sample verification (2026-09-26)

## Sampler command and output

```bash
python scripts/level_r_pilot_sample.py --out data/level_r/pilot_150.json
```

```
cell                                 n alloc picked
0-2|inplane_0.62_slice_3             3     3      3
0-2|inplane_0.69_slice_5           187    14     14
0-2|inplane_0.86_slice_3             1     1      1
0-2|inplane_0.86_slice_5            19     8      8
0|inplane_0.62_slice_3              20     8      3
0|inplane_0.69_slice_5             493    36     50
0|inplane_0.86_slice_3               5     5      1
0|inplane_0.86_slice_5              27     8      8
2-4|inplane_0.62_slice_3             6     6      3
2-4|inplane_0.69_slice_5           196    15     15
2-4|inplane_0.86_slice_3             3     3      2
2-4|inplane_0.86_slice_5            12     8      8
>4|inplane_0.62_slice_3              3     3      2
>4|inplane_0.69_slice_5            299    22     22
>4|inplane_0.86_slice_3              2     2      2
>4|inplane_0.86_slice_5             21     8      8
total picked 150 (seed 0) -> data/level_r/pilot_150.json
```

## Patient-cap constraint verification

```bash
python - <<'EOF'
import json
from anatobind.level_r.registry import load_registry
from collections import Counter
p = json.load(open("data/level_r/pilot_150.json"))
reg = {r["lesion_id"]: r for r in load_registry()}
pp = Counter(reg[i]["patient_id"] for i in p["lesion_ids"])
print("lesions", len(p["lesion_ids"]), "patients", len(pp), "max per patient", max(pp.values()))
EOF
```

```
lesions 150 patients 82 max per patient 3
```

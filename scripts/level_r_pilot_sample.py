#!/usr/bin/env python
"""Draw the Level R pilot (spec R8 / §9) from the Gate 0.5 registry and write data/level_r/pilot_150.json.

  python scripts/level_r_pilot_sample.py --out data/level_r/pilot_150.json
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from anatobind.level_r.pilot import sample_pilot  # noqa: E402
from anatobind.level_r.registry import REGISTRY, load_registry  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--registry", type=Path, default=REGISTRY)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--min-per-cell", type=int, default=8)
    ap.add_argument("--max-per-patient", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    if a.out.exists():
        sys.exit(f"{a.out} exists; the pilot is drawn once")
    p = sample_pilot(load_registry(a.registry), a.n, a.min_per_cell, a.max_per_patient, a.seed)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(p, indent=1))
    print(f"{'cell':32s} {'n':>5s} {'alloc':>5s} {'picked':>6s}")
    for c, v in p["cells"].items():
        print(f"{c:32s} {v['n']:5d} {v['alloc']:5d} {v['picked']:6d}")
    print(f"total picked {len(p['lesion_ids'])} (seed {a.seed}) -> {a.out}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Level R export (spec §4): the Gate 0.5 registry's 165 FLAIR volumes as uint16 arrays + geometry, and every
lesion's per-slice boxes recomputed along the Gate 0.5 path and asserted equal to the registry.

  PYTHONNOUSERSITE=1 PYTHONPATH=. nice -n 19 ~/anaconda3/envs/nvgen/bin/python scripts/level_r_export.py \
      --out /data2/congcong/data/FM_data/derived/level_r
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from anatobind.data_engine.fastmri import read_fastmri_plus_rows  # noqa: E402
from anatobind.level_r.export import export_all  # noqa: E402
from anatobind.level_r.registry import REGISTRY, load_registry  # noqa: E402

FM = Path("/data2/congcong/data/FM_data")
CSV = FM / "fastMRI_lh_brain_knee/Annotations/brain.csv"
KROOT = FM / "fastMRI_lh_brain_knee/kspace/brain"


def h5_of(stem):
    for split in ("multicoil_train", "multicoil_val"):
        p = KROOT / split / f"{stem}.h5"
        if p.exists():
            return p
    raise FileNotFoundError(stem)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--registry", type=Path, default=REGISTRY)
    ap.add_argument("--limit", type=int, default=0, help="only the first N volumes (smoke test)")
    a = ap.parse_args()
    if (a.out / "lesions.json").exists():
        sys.exit(f"{a.out / 'lesions.json'} exists; this export never overwrites (choose another --out)")
    registry = load_registry(a.registry)
    if a.limit:
        keep = sorted({r["file"] for r in registry})[:a.limit]
        registry = [r for r in registry if r["file"] in keep]
    recs = export_all(registry, read_fastmri_plus_rows(CSV), a.out, h5_of)
    print(f"exported {len(recs)} lesions from {len({r['volume_code'] for r in recs})} volumes to {a.out}")


if __name__ == "__main__":
    main()

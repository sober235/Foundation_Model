#!/usr/bin/env python
"""Paired comparison of two evaluated arms (anatobind.aur.compare): e.g. C2 (Stage I backbone) against C0 (random).

    PYTHONNOUSERSITE=1 PYTHONPATH=. ~/anaconda3/envs/nvgen/bin/python scripts/aur_compare.py \
        --a <EVAL_DIR_C2> --b <EVAL_DIR_C0> --name-a C2 --name-b C0 --out <NEW_DIR>

Writes compare.json and COMPARE.md into a new directory. NOT_EVIDENCE (pseudo-labels)."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from anatobind.aur.compare import compare, markdown  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--a", type=Path, required=True)
    ap.add_argument("--b", type=Path, required=True)
    ap.add_argument("--name-a", default="A")
    ap.add_argument("--name-b", default="B")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n-boot", type=int, default=2000)
    a = ap.parse_args(argv)
    a.out.mkdir(parents=True, exist_ok=False)
    res = compare(a.a, a.b, n_boot=a.n_boot)
    res.update({"name_a": a.name_a, "name_b": a.name_b})
    (a.out / "compare.json").write_text(json.dumps(res, indent=1))
    text = markdown(res, a.name_a, a.name_b)
    (a.out / "COMPARE.md").write_text(text)
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())

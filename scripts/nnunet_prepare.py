#!/usr/bin/env python
"""Build the nnU-Net raw dataset (--stage raw) or write the fold splits (--stage splits).

  source scripts/nnunet_env.sh
  PYTHONPATH=. python scripts/nnunet_prepare.py --stage raw
  nice -n 19 nnUNetv2_plan_and_preprocess -d 901 -c 3d_fullres --verify_dataset_integrity -np 4
  PYTHONPATH=. python scripts/nnunet_prepare.py --stage splits
"""
import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from anatobind.nnunet.prepare import build_raw, write_splits  # noqa: E402
from anatobind.train.dataset import list_ready_scans  # noqa: E402

M1R = Path("/data2/congcong/data/FM_data/derived/skmtea/m1r")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=("raw", "splits"), required=True)
    a = ap.parse_args()
    if a.stage == "raw":
        print(f"{build_raw(M1R, os.environ['nnUNet_raw'], list_ready_scans(M1R))} training cases")
    else:
        write_splits(os.environ["nnUNet_preprocessed"], json.loads((M1R / "splits.json").read_text())["folds"])
        print("splits_final.json written")


if __name__ == "__main__":
    main()

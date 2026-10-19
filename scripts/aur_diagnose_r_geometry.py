#!/usr/bin/env python
"""CPU-only native replay or explicit versioned stable-distance derivation.

Use --derive-from with --source-snapshot to create a fresh derived cache. A
stable derived success never reclassifies the original native replay failure.
"""
import os
import sys
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from anatobind.aur.ar_diagnostics import main

if __name__ == "__main__":
    raise SystemExit(main(replay=True))

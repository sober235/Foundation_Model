#!/usr/bin/env python
"""Collect A32 and R together with one new frozen float32 TRAIN forward/crop."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from anatobind.aur.ar_diagnostics import main

if __name__ == "__main__":
    raise SystemExit(main())

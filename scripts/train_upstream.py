#!/usr/bin/env python
"""Thin CLI over anatobind.train.train_upstream.main (see its docstring for the command)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from anatobind.train.train_upstream import main  # noqa: E402

if __name__ == "__main__":
    main()

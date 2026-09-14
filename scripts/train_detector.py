#!/usr/bin/env python
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from anatobind.train.train_detector import main  # noqa: E402

if __name__ == "__main__":
    main()

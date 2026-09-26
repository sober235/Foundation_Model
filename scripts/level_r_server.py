#!/usr/bin/env python
"""Level R service (spec §10). Binds 127.0.0.1:8790 by default; pass --bind 0.0.0.0 explicitly for LAN access.

  D=/data2/congcong/data/FM_data/derived/level_r
  setsid nohup env PYTHONNOUSERSITE=1 PYTHONPATH=. ~/anaconda3/envs/nvgen/bin/python scripts/level_r_server.py \
      --db $D/level_r.sqlite --data-root $D --pid-file $D/server.pid > $D/server.log 2>&1 &
"""
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from anatobind.level_r.server import make_server  # noqa: E402
from anatobind.level_r.store import Store  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", type=Path, required=True)
    ap.add_argument("--data-root", type=Path, required=True, help="directory holding volumes/ from level_r_export.py")
    ap.add_argument("--bind", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8790)
    ap.add_argument("--pid-file", type=Path)
    a = ap.parse_args()
    if not (a.data_root / "volumes").is_dir():
        sys.exit(f"{a.data_root}/volumes is missing; run scripts/level_r_export.py first")
    srv = make_server(Store(a.db), a.data_root, a.bind, a.port)
    if a.pid_file:
        a.pid_file.write_text(str(os.getpid()))
    print(f"Level R serving {a.data_root} on http://{a.bind}:{srv.server_address[1]}/ (db {a.db})", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Fetch SKM-TEA segmentation_masks/raw-data-track through the Redivis API.

Needs a read-only API token (scope data.data) in ~/.redivis_token, created by the user at
https://redivis.com/workspace/settings/tokens. The token is read from that file only, exported to
the client through REDIVIS_API_TOKEN, and never printed. Runs with the isolated client venv:

  V=/tmp/claude-1002/-home-congcongliu--claude/ae80a0ab-ab7e-4342-8ff8-78d7232fe5e7/scratchpad/redivis_venv
  PYTHONPATH=. $V/bin/python scripts/fetch_skmtea_rawtrack_api.py --list        # tables and files, no download
  PYTHONPATH=. $V/bin/python scripts/fetch_skmtea_rawtrack_api.py --download    # raw-data-track NIfTIs -> DEST
"""
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from anatobind.data_engine.rawtrack_fetch import check_complete, extract_raw_track, select_api_files  # noqa: E402

TOKEN_FILE = Path.home() / ".redivis_token"
FM = Path("/data2/congcong/data/FM_data")
DEST = FM / "SKM-TEA_ltr/segmentation_masks/raw-data-track"
ARCHIVES = FM / "SKM-TEA_ltr/redivis_archives"
RAW = FM / "SKM-TEA/files_recon_calib-24"
ORG = "aimi"   # the dataset page lives under the Stanford institution domain, the owner is the AIMI organization
DATASET_NAMES = ("skm_tea", "SKM-TEA")     # the URL id is 5r8z-achpw6q4f
MAX_FILES = 100000


def load_token():
    if not TOKEN_FILE.exists():
        raise SystemExit(f"{TOKEN_FILE} not found: create a data.data token at "
                         "https://redivis.com/workspace/settings/tokens and store it there (umask 077).")
    token = TOKEN_FILE.read_text().strip()
    if len(token) < 20:
        raise SystemExit(f"{TOKEN_FILE} does not look like a token")
    os.environ["REDIVIS_API_TOKEN"] = token


def find_dataset(redivis):
    errors = []
    for name in DATASET_NAMES:
        try:
            ds = redivis.organization(ORG).dataset(name)
            ds.get()
            return ds
        except Exception as exc:  # the client raises on 404 and on 401 alike
            errors.append(f"{ORG}.{name}: {type(exc).__name__}: {str(exc)[:120]}")
    try:
        for ds in redivis.organization(ORG).list_datasets(max_results=1000):
            name = str(ds.properties.get("name", "")).lower()
            if "skm" in name and "tea" in name:
                ds.get()
                return ds
    except Exception as exc:
        errors.append(f"list_datasets: {type(exc).__name__}: {str(exc)[:120]}")
    raise SystemExit("dataset not found:\n  " + "\n  ".join(errors))


def all_files(ds):
    out = []
    for table in ds.list_tables():
        table.get()
        props = table.properties
        kind = props.get("kind") or props.get("tableType") or ""
        print(f"table {props.get('name')!r} kind={kind!r} rows={props.get('numRows')} files={props.get('numFiles')}")
        try:
            files = table.list_files(MAX_FILES)
        except Exception as exc:
            print(f"   (no file listing: {type(exc).__name__}: {str(exc)[:100]})")
            continue
        out += files
    return out


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--list", action="store_true")
    g.add_argument("--download", action="store_true")
    a = ap.parse_args()
    load_token()
    import redivis  # noqa: E402  (imported after the token is in the environment)

    ds = find_dataset(redivis)
    print(f"dataset {ds.properties.get('name')!r} reference {ds.qualified_reference!r} "
          f"version {ds.properties.get('version', {}).get('tag') if isinstance(ds.properties.get('version'), dict) else ds.properties.get('version')}")
    files = all_files(ds)
    print(f"{len(files)} files in total")
    niftis, archives = select_api_files(files)
    print(f"raw-data-track NIfTIs by path: {len(niftis)}; candidate archives: {[f.name for f in archives]}")
    if a.list:
        for f in sorted(files, key=lambda f: str(f.properties.get("path") or f.name))[:80]:
            print(f"  {f.properties.get('size', '?'):>12}  {f.properties.get('path') or f.name}")
        if len(files) > 80:
            print(f"  ... {len(files) - 80} more")
        return

    scans = sorted(p.name[:-3] for p in RAW.glob("MTR_*.h5"))
    DEST.mkdir(parents=True, exist_ok=True)
    written = []
    if niftis:
        for f in niftis:
            target = DEST / f.name
            if not target.exists():
                f.download(str(target), overwrite=False)
            written.append(f.name)
        print(f"downloaded {len(written)} NIfTIs")
    elif archives:
        ARCHIVES.mkdir(parents=True, exist_ok=True)
        for f in archives:
            target = ARCHIVES / f.name
            if not target.exists():
                print(f"downloading {f.name} ({f.properties.get('size', '?')} bytes)", flush=True)
                f.download(str(target), overwrite=False)
            written += extract_raw_track(target, DEST)
    else:
        raise SystemExit("nothing matched: run --list and inspect the file names")
    missing, extra = check_complete(sorted(set(written)), scans)
    print(f"raw-data-track files on disk: {len(set(written))} (expected {len(scans)}); missing {missing}; unexpected {extra}")
    sys.exit(1 if missing else 0)


if __name__ == "__main__":
    main()

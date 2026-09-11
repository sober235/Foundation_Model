#!/usr/bin/env python
"""Download and unpack SKM-TEA segmentation_masks/raw-data-track from a Redivis signed URL.

The user pastes the signed download URL into URL_FILE. It is read once and never printed; the
file and the archive are deleted after a complete extraction.

  PYTHONNOUSERSITE=1 PYTHONPATH=. ~/anaconda3/envs/nvgen/bin/python scripts/fetch_skmtea_rawtrack_seg.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from anatobind.data_engine.rawtrack_fetch import check_complete, download, extract_raw_track  # noqa: E402

FM = Path("/data2/congcong/data/FM_data")
URL_FILE = FM / "SKM-TEA_ltr/.signed_url_raw_track"
DEST = FM / "SKM-TEA_ltr/segmentation_masks/raw-data-track"
ARCHIVE = FM / "SKM-TEA_ltr/raw-data-track.download"
RAW = FM / "SKM-TEA/files_recon_calib-24"


def main():
    scans = sorted(p.name[:-3] for p in RAW.glob("MTR_*.h5"))
    if not ARCHIVE.exists():
        url = URL_FILE.read_text().strip()
        if not url.startswith("https://"):
            raise SystemExit(f"{URL_FILE} does not hold an https URL")
        n, md5 = download(url, ARCHIVE)
        print(f"downloaded {n} bytes, md5 {md5}", flush=True)
    written = extract_raw_track(ARCHIVE, DEST)
    missing, extra = check_complete(written, scans)
    print(f"raw-data-track files written: {len(written)} (expected {len(scans)}); missing {missing}; unexpected {extra}")
    if missing or extra:
        raise SystemExit(1)
    URL_FILE.unlink(missing_ok=True)
    ARCHIVE.unlink()
    print("complete; signed URL file and archive removed")


if __name__ == "__main__":
    main()

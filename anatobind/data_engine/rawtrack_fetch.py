"""Fetch SKM-TEA's gradient-warp-corrected segmentations (segmentation_masks/raw-data-track).

SKM-TEA ships two segmentation sets. The dicom-track masks were drawn on scanner DICOMs, which
carry the vendor's gradient-warp correction; the raw-data-track masks were registered by the
authors onto the SENSE reconstructions (paper appendix A.3). Anything paired with images
reconstructed from raw k-space must use the raw-data-track set (appendix A.6).
"""
import hashlib
import re
import shutil
import tarfile
import urllib.request
import zipfile
from pathlib import Path

MEMBER = re.compile(r"(?:^|/)raw-data-track/(MTR_\d{3}\.nii\.gz)$")


def select_members(names):
    """Archive member name -> output file name, for raw-data-track NIfTIs only."""
    out = {}
    for name in names:
        m = MEMBER.search(name)
        if m:
            out[name] = m.group(1)
    return out


def extract_raw_track(archive, dest):
    """Copy the raw-data-track NIfTIs out of a tar or zip archive; returns the file names written."""
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    written = []
    if tarfile.is_tarfile(archive):
        with tarfile.open(archive) as tar:
            members = [m for m in tar.getmembers() if m.isfile()]
            keep = select_members(m.name for m in members)
            for m in members:
                if m.name in keep:
                    with tar.extractfile(m) as src, open(dest / keep[m.name], "wb") as dst:
                        shutil.copyfileobj(src, dst)
                    written.append(keep[m.name])
    elif zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as zf:
            for name, out in select_members(zf.namelist()).items():
                with zf.open(name) as src, open(dest / out, "wb") as dst:
                    shutil.copyfileobj(src, dst)
                written.append(out)
    else:
        raise ValueError(f"{archive} is neither a tar nor a zip archive")
    return sorted(written)


ARCHIVE = re.compile(r"(segmentation|raw-data-track)[^/]*\.(tar\.gz|tgz|tar|zip)$", re.IGNORECASE)


def select_api_files(files):
    """Split Redivis File objects into (raw-data-track NIfTIs, candidate archives).

    A NIfTI is selected only when its path names the raw-data-track folder; a bare MTR_xxx.nii.gz
    cannot be told from its dicom-track twin and is left alone. Archives whose name mentions
    segmentation masks or the raw-data track are returned separately for extract_raw_track.
    """
    niftis, archives = [], []
    for f in files:
        path = f.properties.get("path") or f.name
        if MEMBER.search(str(path)):
            niftis.append(f)
        elif ARCHIVE.search(f.name):
            archives.append(f)
    return niftis, archives


def check_complete(written, scans):
    """(missing scan ids, unexpected scan ids) of a written file list against the expected scans."""
    got = {w[: -len(".nii.gz")] for w in written}
    return sorted(set(scans) - got), sorted(got - set(scans))


def download(url, out_path, chunk=1 << 20):
    """Stream url to out_path; returns (bytes, md5 hex). The URL itself is never logged."""
    md5, n = hashlib.md5(), 0
    with urllib.request.urlopen(url, timeout=120) as resp, open(out_path, "wb") as fh:
        while True:
            block = resp.read(chunk)
            if not block:
                break
            fh.write(block)
            md5.update(block)
            n += len(block)
    return n, md5.hexdigest()

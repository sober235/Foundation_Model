#!/usr/bin/env python
# scripts/audit_fastmri_plus_boxes.py
"""Gate 0 overlay / intensity audit (v2.6 §4.2 item 4) over every annotated volume: under the converted mapping a
fastMRI+ box must sit on the bright tissue it names, under the as-is mapping it must not.
Knee: joint effusion and subchondral oedema boxes on fat-saturated PD (CORPDFS). Brain: nonspecific white-matter
lesions and lacunar infarcts on FLAIR, reported per series. Contrast = 75th percentile inside the box / median of
a ring around it (the 2026-09-15 probe's statistic, probe0b_contrast.py).

  PYTHONNOUSERSITE=1 PYTHONPATH=. nice -n 19 ~/anaconda3/envs/nvgen/bin/python scripts/audit_fastmri_plus_boxes.py \
      --out docs/verification/2026-09-24/gate0 --figs ~/figs/anatobind_gate0
"""
import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from anatobind.data_engine.fastmri import convert_box_csv_to_rss, read_fastmri_plus_rows  # noqa: E402

FM = Path("/data2/congcong/data/FM_data")
KNEE_CSV = FM / "fastMRI_lh_brain_knee/Annotations/knee.csv"
BRAIN_CSV = FM / "fastMRI_lh_brain_knee/Annotations/brain.csv"
KNEE_ROOT = FM / "fastMRI_lh_brain_knee/kspace/knee"
BRAIN_ROOT = FM / "fastMRI_lh_brain_knee/kspace/brain"
KNEE_LABELS = {"Joint Effusion", "Bone- Subchondral edema"}
BRAIN_LABELS = {"Nonspecific white matter lesion", "Lacunar infarct"}
KNEE_MIN_WIN, BRAIN_MIN_WIN, SERIES_MIN_WIN, SERIES_MIN_N = 0.90, 0.90, 0.75, 3


def h5_path(root, stem):
    for split in ("multicoil_train", "multicoil_val"):
        p = root / split / f"{stem}.h5"
        if p.exists():
            return p
    return None


def series_of(file):
    m = re.search(r"AX[A-Z0-9]+_(\d+)_", file)
    return m.group(1) if m else "?"


def box_contrast(img, row0, row1, col0, col1):
    """75th percentile inside the box over the median of a ring 2..6 px around it; 1.0 on flat background."""
    inner = np.zeros(img.shape, bool)
    inner[row0:row1, col0:col1] = True
    ring = ndimage.binary_dilation(inner, iterations=6) & ~ndimage.binary_dilation(inner, iterations=2)
    return float(np.percentile(img[inner], 75) / (np.median(img[ring]) + 1e-9))


def volume_verdict(ratios_asis, ratios_converted):
    a, c = float(np.median(ratios_asis)), float(np.median(ratios_converted))
    return "converted" if c > a else "as_is" if a > c else "tie"


def audit_volume(rss, rows):
    """rss: (slice, row, col). Returns (as-is ratios, converted ratios) over the volume's boxes."""
    ns, nr, nc = rss.shape
    asis, conv = [], []
    for r in rows:
        if not 0 <= r["slice"] < ns:
            continue
        img = rss[r["slice"]]
        for kind, (row0, row1, col0, col1) in (("as_is", (r["y"], r["y"] + r["height"], r["x"], r["x"] + r["width"])),
                                                ("converted", convert_box_csv_to_rss(r["x"], r["y"], r["width"], r["height"], nr))):
            row0, row1, col0, col1 = max(0, row0), min(nr, row1), max(0, col0), min(nc, col1)
            if row1 <= row0 or col1 <= col0:
                continue
            (asis if kind == "as_is" else conv).append(box_contrast(img, row0, row1, col0, col1))
    return asis, conv


def audit_organ(csv_path, root, labels, acquisition=None, file_filter=None):
    """file_filter: predicate on the file stem; when given, rows whose file fails it are dropped before grouping."""
    rows = [r for r in read_fastmri_plus_rows(csv_path) if r["label"] in labels and r["width"] >= 3 and r["height"] >= 3]
    if file_filter is not None:
        rows = [r for r in rows if file_filter(r["file"])]
    by_file = defaultdict(list)
    for r in rows:
        by_file[r["file"]].append(r)
    per_volume = {}
    for f in sorted(by_file):
        p = h5_path(root, f)
        if p is None:
            continue
        with h5py.File(p, "r") as h:
            if acquisition is not None and str(h.attrs.get("acquisition", "")) != acquisition:
                continue
            rss = h["reconstruction_rss"][()]
        asis, conv = audit_volume(rss, by_file[f])
        if asis and conv:
            per_volume[f] = {"n_boxes": len(conv), "median_as_is": float(np.median(asis)),
                             "median_converted": float(np.median(conv)), "verdict": volume_verdict(asis, conv),
                             "series": series_of(f), "rss_rows": int(rss.shape[1])}
    return per_volume


def summarise(per_volume):
    out = {"n": len(per_volume), "converted": sum(v["verdict"] == "converted" for v in per_volume.values()),
           "as_is": sum(v["verdict"] == "as_is" for v in per_volume.values()),
           "median_ratio_as_is": float(np.median([v["median_as_is"] for v in per_volume.values()])) if per_volume else None,
           "median_ratio_converted": float(np.median([v["median_converted"] for v in per_volume.values()])) if per_volume else None,
           "per_series": {}}
    for s in sorted({v["series"] for v in per_volume.values()}):
        sel = [v for v in per_volume.values() if v["series"] == s]
        out["per_series"][s] = {"n": len(sel), "converted": sum(v["verdict"] == "converted" for v in sel),
                                "rss_rows": sorted({v["rss_rows"] for v in sel})}
    return out


def decide(summary):
    knee, brain = summary["knee"], summary["brain"]
    if knee["n"] == 0 or brain["n"] == 0:
        return False
    if knee["converted"] / knee["n"] < KNEE_MIN_WIN or brain["converted"] / brain["n"] < BRAIN_MIN_WIN:
        return False
    for s in brain["per_series"].values():
        if s["n"] >= SERIES_MIN_N and s["converted"] / s["n"] < SERIES_MIN_WIN:
            return False
    return True


def overlays(per_volume, csv_path, root, labels, figs, name, per_series=False):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = [r for r in read_fastmri_plus_rows(csv_path) if r["label"] in labels and r["file"] in per_volume]
    picks = []
    if per_series:
        for s in sorted({v["series"] for v in per_volume.values()}):
            picks.append(next(f for f, v in per_volume.items() if v["series"] == s))
    else:
        picks = sorted(per_volume)[:8]
    fig, axes = plt.subplots(2, len(picks), figsize=(2.6 * len(picks), 5.6), squeeze=False)
    for j, f in enumerate(picks):
        r = next(r for r in rows if r["file"] == f)
        with h5py.File(h5_path(root, f), "r") as h:
            img = h["reconstruction_rss"][r["slice"]]
        nr = img.shape[0]
        for i, (kind, (row0, row1, col0, col1)) in enumerate((("as-is", (r["y"], r["y"] + r["height"], r["x"], r["x"] + r["width"])),
                                                              ("converted", convert_box_csv_to_rss(r["x"], r["y"], r["width"], r["height"], nr)))):
            ax = axes[i, j]
            ax.imshow(img, cmap="gray", vmin=0, vmax=np.percentile(img, 99.5))
            ax.add_patch(plt.Rectangle((col0, row0), col1 - col0, row1 - row0, fill=False, edgecolor="lime", linewidth=1.2))
            ax.set_title(f"{kind} {per_volume[f]['series']} {f[-7:]} s{r['slice']}", fontsize=7)
            ax.axis("off")
    plt.tight_layout()
    figs.mkdir(parents=True, exist_ok=True)
    plt.savefig(figs / f"{name}.png", dpi=110)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--figs", type=Path, default=Path.home() / "figs/anatobind_gate0")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    knee = audit_organ(KNEE_CSV, KNEE_ROOT, KNEE_LABELS, acquisition="CORPDFS_FBK")
    brain = audit_organ(BRAIN_CSV, BRAIN_ROOT, BRAIN_LABELS, file_filter=lambda f: "AXFLAIR" in f)
    summary = {"knee": summarise(knee), "brain": summarise(brain)}
    summary["pass"] = decide(summary)
    (a.out / "audit.json").write_text(json.dumps({"summary": summary, "knee": knee, "brain": brain}, indent=1))
    overlays(knee, KNEE_CSV, KNEE_ROOT, KNEE_LABELS, a.figs, "knee_effusion_asis_vs_converted")
    overlays(brain, BRAIN_CSV, BRAIN_ROOT, BRAIN_LABELS, a.figs, "brain_small_lesions_per_series", per_series=True)
    lines = ["# Gate 0 box audit", "",
             "| organ | volumes | converted wins | as-is wins | median ratio as-is | median ratio converted |", "|---|---|---|---|---|---|"]
    for organ in ("knee", "brain"):
        s = summary[organ]
        lines.append(f"| {organ} | {s['n']} | {s['converted']} | {s['as_is']} | {s['median_ratio_as_is']:.3f} | {s['median_ratio_converted']:.3f} |")
    lines += ["", "brain per series (n, converted wins, RSS rows):", ""]
    for s, v in summary["brain"]["per_series"].items():
        lines.append(f"- {s}: n={v['n']}, converted={v['converted']}, rows={v['rss_rows']}")
    lines += ["", f"GATE0_AUDIT: {'PASS' if summary['pass'] else 'FAIL'} (knee >= {KNEE_MIN_WIN:.0%}, brain >= {BRAIN_MIN_WIN:.0%}, "
              f"every series with >= {SERIES_MIN_N} volumes >= {SERIES_MIN_WIN:.0%})"]
    (a.out / "audit.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()

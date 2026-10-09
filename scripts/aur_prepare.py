#!/usr/bin/env python
# scripts/aur_prepare.py
"""Data preparation and the pre-flight checks of the AnatoBind brain model (spec §3, §12 P1–P2).

  --stage samples --out <json>              the sample table with splits (refuses an existing file)
  --stage grids --samples <json>            every row: image, anatomy and lesion on one grid (prints the failures)
  --stage isles_check --samples <json> --out <dir>   host volumes of the SynthSeg maps per source and six ISLES
                                            montages (DWI with the SynthSeg map), for a human to look at

  PYTHONNOUSERSITE=1 PYTHONPATH=. nice -n 19 ~/anaconda3/envs/nvgen/bin/python scripts/aur_prepare.py --stage samples --out /data2/congcong/data/FM_data/derived/aur/samples.json
"""
import argparse
import json
import sys
from pathlib import Path

import nibabel as nib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from anatobind.anatomy.sources import FM  # noqa: E402
from anatobind.aur.labels import HOST_NAMES, N_HOSTS, host_map  # noqa: E402
from anatobind.aur.samples import all_samples, assign_splits, counts, write_samples  # noqa: E402
from anatobind.infer.brain_disease import check_grid  # noqa: E402

S4_CASES = FM / "derived/brain_anatomy/cases.json"
N_VOLUME_CASES = 30
N_MONTAGE = 6


def stage_samples(out, root=FM, s4_cases=S4_CASES):
    rows = assign_splits(all_samples(root), json.loads(Path(s4_cases).read_text()))
    write_samples(out, rows)
    c = counts(rows)
    for source, d in c.items():
        print(f"{source}: train {d['train'][0]} cases / {d['train'][1]} rows / {d['train'][2]} with U; "
              f"test {d['test'][0]} cases / {d['test'][1]} rows / {d['test'][2]} with U")
    print(f"wrote {out}: {len(rows)} rows")
    return rows


def stage_grids(rows):
    """Returns the rows whose files are not on one grid."""
    bad = []
    for r in rows:
        try:
            seg = nib.load(r["anatomy"])
            check_grid([r["image"]] + ([r["lesion"]] if r["lesion"] else []), seg)
        except (ValueError, FileNotFoundError) as e:
            bad.append((r["case"], r["sequence"], str(e)))
            print(f"BAD {r['case']} {r['sequence']}: {e}")
    print(f"{len(rows)} rows checked, {len(bad)} off their grid")
    return bad


def host_volumes_ml(seg, spacing):
    hm = host_map(seg)
    ml = float(np.prod(spacing)) / 1000.0
    return [float((hm == i + 1).sum()) * ml for i in range(N_HOSTS)]


def stage_isles_check(rows, out, n_cases=N_VOLUME_CASES, n_montage=N_MONTAGE):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out = Path(out)
    if out.exists():
        raise FileExistsError(f"{out} exists")
    out.mkdir(parents=True)
    table = {}
    for source in ("pdgm", "bmsr", "isles"):
        seen, vols = set(), []
        for r in rows:
            if r["source"] != source or r["case"] in seen:
                continue
            seen.add(r["case"])
            img = nib.load(r["anatomy"])
            vols.append(host_volumes_ml(np.asarray(img.dataobj), img.header.get_zooms()[:3]))
            if len(seen) >= n_cases:
                break
        table[source] = np.median(np.array(vols), 0).tolist() if vols else [float("nan")] * N_HOSTS
    lines = ["host," + ",".join(table)] + [f"{HOST_NAMES[i]}," + ",".join(f"{table[s][i]:.1f}" for s in table) for i in range(N_HOSTS)]
    (out / "host_volumes_median_ml.csv").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    done = 0
    for r in rows:
        if r["source"] != "isles" or r["source_sequence"] != "DWI":
            continue
        img, seg = nib.load(r["image"]), nib.load(r["anatomy"])
        data, lab = np.asarray(img.dataobj).astype(np.float32), np.asarray(seg.dataobj)
        ks = [int(data.shape[2] * f) for f in (0.3, 0.45, 0.6, 0.75)]
        fig, axes = plt.subplots(2, len(ks), figsize=(3.2 * len(ks), 6.4))
        for i, k in enumerate(ks):
            for row in range(2):
                axes[row, i].imshow(data[:, :, k].T, cmap="gray", vmin=0, vmax=np.percentile(data, 99.5), origin="lower")
                axes[row, i].axis("off")
            axes[0, i].set_title(f"slice {k}")
            axes[1, i].imshow(np.ma.masked_where(lab[:, :, k].T == 0, host_map(lab[:, :, k]).T), cmap="tab20", vmin=0, vmax=15, alpha=0.5, origin="lower")
        fig.suptitle(f"{r['case']}: DWI / SynthSeg host classes")
        fig.tight_layout()
        png = out / f"isles_{r['case']}.png"
        fig.savefig(png, dpi=72)
        plt.close(fig)
        print(f"wrote {png}")
        done += 1
        if done >= n_montage:
            break
    return table


def main(argv=None):
    ap = argparse.ArgumentParser(description="AnatoBind brain data preparation")
    ap.add_argument("--stage", choices=("samples", "grids", "isles_check"), required=True)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--samples", type=Path)
    a = ap.parse_args(argv)
    if a.stage == "samples":
        if a.out is None:
            ap.error("--out is needed")
        stage_samples(a.out)
        return 0
    if a.samples is None:
        ap.error("--samples is needed")
    rows = json.loads(a.samples.read_text())
    if a.stage == "grids":
        return 1 if stage_grids(rows) else 0
    if a.out is None:
        ap.error("--out is needed")
    stage_isles_check(rows, a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())

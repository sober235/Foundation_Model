"""Paired comparison of two evaluated arms on the same test rows (SSL-first plan §6: C2 = Stage I backbone, C0 = random
backbone, otherwise identical). Reads each arm's `cases.json` and `aggregate.json` (scripts/aur_eval.py) and reports,
per track, both arms and the difference A - B with a patient-level bootstrap interval (the rows of one patient move
together): A, the 13-host macro Dice of the A-supervised rows; U, the lesion sensitivity and false positives per scan
at each arm's own operating point (the threshold its evaluation chose on the validation split) per source and
sequence; R, the controlled-track host accuracy with the paired table (both right, A only, B only, both wrong) and the
B0 lookup of each arm. Everything rests on pseudo-labels: NOT_EVIDENCE."""
import json
from pathlib import Path

import numpy as np

from anatobind.aur.eval import scan, scan_matches


def _load(d):
    d = Path(d)
    cases = json.loads((d / "cases.json").read_text())
    agg = json.loads((d / "aggregate.json").read_text())
    return {(c["source"], c["case"], c["sequence"]): c for c in cases}, agg


def patient_bootstrap(values, patients, n_boot=1000, seed=0, stat=np.mean):
    """(2.5 %, 97.5 %) of `stat` over resamples of the patients (all rows of a drawn patient enter together)."""
    values = np.asarray(values, float)
    pts = sorted(set(patients))
    rows = {p: [i for i, q in enumerate(patients) if q == p] for p in pts}
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_boot):
        sel = [i for j in rng.integers(0, len(pts), len(pts)) for i in rows[pts[j]]]
        out.append(stat(values[sel]))
    out = np.asarray(out, float)
    out = out[np.isfinite(out)]
    return (float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))) if out.size else (float("nan"), float("nan"))


def _patient(row):
    return f"{row['source']}:{row.get('patient', row['case'])}"


def _u_counts(row, thr):
    """(hits, truths, false positives) of one row at a threshold; ignored truths excluded."""
    s = scan(row["case"], row["u"]["gt"], row["u"]["dets"])
    hits, fp, _ = scan_matches(s, thr)
    return len(hits), sum(1 for g in s["gt"] if not g.get("ignore")), fp


def compare(dir_a, dir_b, n_boot=1000, seed=0):
    a, agg_a = _load(dir_a)
    b, agg_b = _load(dir_b)
    if set(a) != set(b):
        raise ValueError(f"the two arms do not hold the same rows ({len(set(a) ^ set(b))} rows differ)")
    keys = sorted(a)
    res = {"evidence": "NOT_EVIDENCE: SynthSeg / lesion pseudo-labels", "n_rows": len(keys), "arm_a": str(dir_a), "arm_b": str(dir_b)}
    # A
    rows = [k for k in keys if a[k].get("a_supervised", True) and a[k].get("a") and b[k].get("a")]
    va = np.array([a[k]["a"]["host_macro"] for k in rows], float)
    vb = np.array([b[k]["a"]["host_macro"] for k in rows], float)
    ok = np.isfinite(va) & np.isfinite(vb)
    pts = [_patient(a[k]) for k, m in zip(rows, ok) if m]
    d = (va - vb)[ok]
    res["a"] = {"n": int(ok.sum()), "mean_a": float(va[ok].mean()) if ok.any() else float("nan"), "mean_b": float(vb[ok].mean()) if ok.any() else float("nan"),
                "diff": float(d.mean()) if d.size else float("nan"), "ci95": list(patient_bootstrap(d, pts, n_boot, seed)) if d.size else [float("nan")] * 2}
    # U per source/sequence at each arm's own threshold
    res["u"] = {}
    groups = sorted({(k[0], k[2]) for k in keys if a[k].get("u") and b[k].get("u")})
    for src, seq in groups:
        ks = [k for k in keys if k[0] == src and k[2] == seq and a[k].get("u") and b[k].get("u")]
        ta = agg_a.get("u", {}).get("per_source", {}).get(src, {}).get("thr")
        tb = agg_b.get("u", {}).get("per_source", {}).get(src, {}).get("thr")
        if ta is None or tb is None:
            continue
        ca = np.array([_u_counts(a[k], ta) for k in ks], float)
        cb = np.array([_u_counts(b[k], tb) for k in ks], float)
        n_gt = ca[:, 1].sum()
        pts = [_patient(a[k]) for k in ks]
        both = np.stack([ca[:, 0], cb[:, 0], ca[:, 1]], 1)
        idx = np.arange(len(ks))

        def diff_stat(sel, both=both):
            s = both[sel.astype(int)]
            return (s[:, 0].sum() - s[:, 1].sum()) / max(s[:, 2].sum(), 1.0)

        lo, hi = patient_bootstrap(idx, pts, n_boot, seed, stat=diff_stat)
        res["u"][f"{src}/{seq}"] = {"n_scans": len(ks), "n_gt": int(n_gt), "thr_a": ta, "thr_b": tb,
                                    "sens_a": float(ca[:, 0].sum() / max(n_gt, 1.0)), "sens_b": float(cb[:, 0].sum() / max(n_gt, 1.0)),
                                    "fp_per_scan_a": float(ca[:, 2].mean()), "fp_per_scan_b": float(cb[:, 2].mean()),
                                    "diff": float((ca[:, 0].sum() - cb[:, 0].sum()) / max(n_gt, 1.0)), "ci95": [lo, hi]}
    # R, controlled track, paired by instance
    ra, rb, rb0a, rb0b, pts = [], [], [], [], []
    for k in keys:
        if not (a[k].get("r") and b[k].get("r")):
            continue
        ib = {i["instance"]: i for i in b[k]["r"]["instances"]}
        for i in a[k]["r"]["instances"]:
            j = ib.get(i["instance"])
            if j is None or j["truth"] != i["truth"]:
                continue
            ra.append(i["r"] == i["truth"])
            rb.append(j["r"] == j["truth"])
            rb0a.append(i["b0"] == i["truth"])
            rb0b.append(j["b0"] == j["truth"])
            pts.append(_patient(a[k]))
    ra, rb = np.array(ra, bool), np.array(rb, bool)
    d = ra.astype(float) - rb.astype(float)
    res["r"] = {"n": int(ra.size), "aba_a": float(ra.mean()) if ra.size else float("nan"), "aba_b": float(rb.mean()) if rb.size else float("nan"),
                "aba_b0_a": float(np.mean(rb0a)) if rb0a else float("nan"), "aba_b0_b": float(np.mean(rb0b)) if rb0b else float("nan"),
                "diff": float(d.mean()) if d.size else float("nan"), "ci95": list(patient_bootstrap(d, pts, n_boot, seed)) if d.size else [float("nan")] * 2,
                "paired": {"both_right": int((ra & rb).sum()), "a_only": int((ra & ~rb).sum()), "b_only": int((~ra & rb).sum()), "both_wrong": int((~ra & ~rb).sum())}}
    return res


def _f(v, d=4):
    return "nan" if v is None or v != v else f"{v:.{d}f}"


def markdown(res, name_a="A", name_b="B"):
    a, r = res["a"], res["r"]
    lines = [f"# {name_a} vs {name_b} on the same {res['n_rows']} test rows ({res['evidence']})", "",
             f"Differences are {name_a} − {name_b}; intervals are 95 % patient-level bootstrap intervals.", "",
             "## A: 13-host macro Dice (A-supervised rows)", "", f"| rows | {name_a} | {name_b} | difference | 95 % CI |", "|---|---|---|---|---|",
             f"| {a['n']} | {_f(a['mean_a'])} | {_f(a['mean_b'])} | {_f(a['diff'])} | [{_f(a['ci95'][0])}, {_f(a['ci95'][1])}] |", "",
             "## U: lesion sensitivity at each arm's own operating point", "",
             f"| source/sequence | scans | lesions | thr {name_a} / {name_b} | sens {name_a} | sens {name_b} | FP/scan {name_a} | FP/scan {name_b} | difference | 95 % CI |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for key, u in res["u"].items():
        lines.append(f"| {key} | {u['n_scans']} | {u['n_gt']} | {_f(u['thr_a'], 2)} / {_f(u['thr_b'], 2)} | {_f(u['sens_a'])} | {_f(u['sens_b'])} | "
                     f"{_f(u['fp_per_scan_a'], 3)} | {_f(u['fp_per_scan_b'], 3)} | {_f(u['diff'])} | [{_f(u['ci95'][0])}, {_f(u['ci95'][1])}] |")
    p = r["paired"]
    lines += ["", "## R: controlled-track host accuracy", "", f"| instances | ABA {name_a} | ABA {name_b} | difference | 95 % CI | B0 {name_a} | B0 {name_b} |",
              "|---|---|---|---|---|---|---|",
              f"| {r['n']} | {_f(r['aba_a'])} | {_f(r['aba_b'])} | {_f(r['diff'])} | [{_f(r['ci95'][0])}, {_f(r['ci95'][1])}] | {_f(r['aba_b0_a'])} | {_f(r['aba_b0_b'])} |", "",
              f"Paired: both right {p['both_right']}, {name_a} only {p['a_only']}, {name_b} only {p['b_only']}, both wrong {p['both_wrong']}.", ""]
    return "\n".join(lines)

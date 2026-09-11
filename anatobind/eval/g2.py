"""Gate G2 (RESEARCH_PLAN v2.2 13.6), pre-registered on 2026-09-11.

Paired set for a view v: in-segmentation lesions whose outcome is WRONG_HOST or CORRECT in both the
clean view and v, i.e. detected with the right class twice. Delta_v is the wrong-host rate in v
minus the rate in clean on that set. G2 passes iff Delta >= 0.05 and the 95% CI of Delta from a
bootstrap over scans excludes 0, for noise_q3 or for us16.
"""
import numpy as np

from anatobind.eval.matching import CORRECT, WRONG_HOST

GATE_VIEWS = ("noise_q3", "us16")
MIN_DELTA = 0.05
REPS = 10_000


def paired_counts(outcomes, view, ref="clean"):
    per = {}
    for (scan, _ann), by_view in outcomes.items():
        a, b = by_view.get(ref), by_view.get(view)
        if a in (WRONG_HOST, CORRECT) and b in (WRONG_HOST, CORRECT):
            n, wr, wv = per.get(scan, (0, 0, 0))
            per[scan] = (n + 1, wr + (a == WRONG_HOST), wv + (b == WRONG_HOST))
    scans = sorted(per)
    return scans, np.array([per[s] for s in scans], dtype=float).reshape(-1, 3)


def delta(arr):
    n = arr[:, 0].sum()
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    return (arr[:, 2].sum() - arr[:, 1].sum()) / n, arr[:, 1].sum() / n, arr[:, 2].sum() / n


def bootstrap_ci(arr, reps=REPS, seed=0, alpha=0.05):
    if len(arr) == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    s = arr[rng.integers(0, len(arr), size=(reps, len(arr)))]
    n = np.maximum(s[..., 0].sum(1), 1.0)
    d = (s[..., 2].sum(1) - s[..., 1].sum(1)) / n
    return float(np.quantile(d, alpha / 2)), float(np.quantile(d, 1 - alpha / 2))


def g2(outcomes, gate_views=GATE_VIEWS, min_delta=MIN_DELTA, reps=REPS, seed=0):
    res = {}
    for v in gate_views:
        scans, arr = paired_counts(outcomes, v)
        d, rate_ref, rate_view = delta(arr)
        lo, hi = bootstrap_ci(arr, reps, seed)
        res[v] = {"n_lesions": int(arr[:, 0].sum()), "n_scans": len(scans), "rate_clean": rate_ref,
                  "rate_view": rate_view, "delta": d, "ci95": [lo, hi],
                  "pass": bool(d >= min_delta and lo > 0)}
    res["pass"] = any(res[v]["pass"] for v in gate_views)
    return res

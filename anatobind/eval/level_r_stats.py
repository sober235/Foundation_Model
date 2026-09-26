"""Reader agreement for Level R (spec §8; v2.6 §7.5, §7.7). Inputs are the latest answers of the two readers (Store
.latest_labels) and the registry rows; nothing here reads the database or the sealed folds directly. The reading-time
summary takes the full label history instead (Store.label_rows), so a revisit adds its time rather than replacing the
first read."""
import math
from collections import Counter, defaultdict

import numpy as np

from anatobind.level_r.registry import BANDS, is_3mm
from anatobind.level_r.schema import NOT_A_LESION, PRIMARY_HOSTS

N_FULL = 1297
NAN = float("nan")
AC1_CATEGORIES = list(PRIMARY_HOSTS) + [NOT_A_LESION]


def host_of(label):
    return NOT_A_LESION if label["not_a_lesion"] else label["primary_host"]


def set_of(label):
    return {NOT_A_LESION} if label["not_a_lesion"] else set(label["acceptable_hosts"])


def pairs(a, b, lesion_ids=None):
    A = {l["lesion_id"]: l for l in a}
    B = {l["lesion_id"]: l for l in b}
    ids = set(A) & set(B)
    if lesion_ids is not None:
        ids &= set(lesion_ids)
    return [{"lesion_id": i, "x": host_of(A[i]), "y": host_of(B[i]), "sx": set_of(A[i]), "sy": set_of(B[i])} for i in sorted(ids)]


def raw_agreement(x, y):
    return sum(p == q for p, q in zip(x, y)) / len(x) if x else NAN


def cohen_kappa(x, y):
    n = len(x)
    if not n:
        return NAN
    cx, cy = Counter(x), Counter(y)
    pe = sum(cx[c] * cy[c] for c in set(cx) | set(cy)) / n ** 2
    po = raw_agreement(x, y)
    return (po - pe) / (1 - pe) if pe < 1 else NAN


def gwet_ac1(x, y, categories=None):
    n = len(x)
    if not n:
        return NAN
    cats = list(categories) if categories else sorted(set(x) | set(y))
    cx, cy = Counter(x), Counter(y)
    pi = {c: (cx[c] + cy[c]) / (2 * n) for c in cats}
    pe = sum(p * (1 - p) for p in pi.values()) / (len(cats) - 1) if len(cats) > 1 else 0.0
    po = raw_agreement(x, y)
    return (po - pe) / (1 - pe) if pe < 1 else NAN


def positive_agreement(x, y):
    out = {}
    cx, cy = Counter(x), Counter(y)
    for c in sorted(set(x) | set(y)):
        agree = sum(p == q == c for p, q in zip(x, y))
        denom = cx[c] + cy[c]
        out[c] = {"n_x": cx[c], "n_y": cy[c], "positive_agreement": 2 * agree / denom if denom else NAN}
    return out


def confusion(x, y):
    cats = sorted(set(x) | set(y))
    return {"categories": cats, "counts": [[sum(p == a and q == b for p, q in zip(x, y)) for b in cats] for a in cats]}


def set_agreement(P):
    return sum(bool(p["sx"] & p["sy"]) for p in P) / len(P) if P else NAN


def field_agreement(a, b, key, lesion_ids=None):
    """Agreement on one categorical answer field (lesion_type or side) over the same pairs as the host statistics: n, raw,
    positive agreement per class and the confusion table (rows reader a, columns reader b). A not_a_lesion answer
    carries no value and takes the class not_a_lesion."""
    A = {l["lesion_id"]: l for l in a}
    B = {l["lesion_id"]: l for l in b}
    ids = [p["lesion_id"] for p in pairs(a, b, lesion_ids)]
    x = [A[i].get(key) or NOT_A_LESION for i in ids]
    y = [B[i].get(key) or NOT_A_LESION for i in ids]
    return {"n": len(ids), "raw": raw_agreement(x, y), "positive_agreement": positive_agreement(x, y), "confusion": confusion(x, y)}


def bootstrap_ci(outcomes_by_patient, n_boot=2000, seed=0, alpha=0.05):
    """Percentile CI of the pooled mean, resampling patients with replacement (v2.6 §12.3: patients are the units)."""
    pats = sorted(outcomes_by_patient)
    arrs = [np.asarray(outcomes_by_patient[p], float) for p in pats]
    rng = np.random.default_rng(seed)
    stats = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(pats), len(pats))
        stats.append(float(np.concatenate([arrs[i] for i in idx]).mean()))
    return float(np.percentile(stats, 100 * alpha / 2)), float(np.percentile(stats, 100 * (1 - alpha / 2)))


def _layer(P):
    return {"n": len(P), "raw": raw_agreement([p["x"] for p in P], [p["y"] for p in P]), "set_agreement": set_agreement(P)}


def _outcomes_by_patient(P, reg_by_id):
    out = defaultdict(list)
    for p in P:
        out[reg_by_id[p["lesion_id"]]["patient_id"]].append(float(p["x"] == p["y"]))
    return out


def strata_report(P, reg_by_id, n_boot=2000, seed=0):
    out = {"band": {}, "stratum_geometry": {}}
    for key in out:
        for value in sorted({reg_by_id[p["lesion_id"]][key] for p in P}, key=lambda v: (BANDS.index(v) if v in BANDS else 99, v)):
            out[key][value] = _layer([p for p in P if reg_by_id[p["lesion_id"]][key] == value])
    if "0" in out["band"]:                  # the 0 mm band is R7's second layer: its raw gets a patient-bootstrap interval too
        band0 = [p for p in P if reg_by_id[p["lesion_id"]]["band"] == "0"]
        out["band"]["0"]["raw_ci95"] = list(bootstrap_ci(_outcomes_by_patient(band0, reg_by_id), n_boot, seed))
    out["slice_3mm"] = _layer([p for p in P if is_3mm(reg_by_id[p["lesion_id"]]["stratum_geometry"])])
    return out


def time_summary(labels, lesion_ids=None):
    """Reading time per lesion = time_seconds summed over every submission for it (pass the full history, label_rows,
    so a revisit adds its time instead of replacing the first read), restricted to lesion_ids when given. n counts the
    lesions with any recorded time."""
    keep = None if lesion_ids is None else set(lesion_ids)
    per = defaultdict(float)
    for l in labels:
        if l.get("time_seconds") is not None and (keep is None or l["lesion_id"] in keep):
            per[(l.get("reader_id"), l["lesion_id"])] += l["time_seconds"]
    t = np.array(list(per.values()), float)
    if not t.size:
        return {"n": 0, "median_s": None, "q1_s": None, "q3_s": None, "hours_for_1297": None}
    med = float(np.median(t))
    return {"n": int(t.size), "median_s": med, "q1_s": float(np.percentile(t, 25)), "q3_s": float(np.percentile(t, 75)),
            "hours_for_1297": med * N_FULL / 3600}


def gate_r7(raw_ci_low, raw_band0, band0_ci_low):
    """R7 as the spec states it; band0_ci_low is reported next to the 0 mm point estimate and does not enter the rule."""
    pass_all = bool(raw_ci_low >= 0.80)
    pass_band0 = bool(raw_band0 >= 0.70)
    return {"all_ci_low": raw_ci_low, "pass_all": pass_all, "band0_raw": raw_band0, "band0_ci_low": band0_ci_low,
            "pass_band0": pass_band0, "single_host_endpoint_allowed": pass_all and pass_band0}


def summarise(a, b, registry, lesion_ids=None, n_boot=2000, seed=0, a_rows=None, b_rows=None):
    """a, b: each reader's latest answers. a_rows, b_rows: their full label history for the reading-time block (falls
    back to a, b). AC1 uses the fixed K = 8 classes of the form (7 hosts + not_a_lesion), not the classes observed.
    lesion_type_agreement and side_agreement cover the same pairs; they do not enter the R7 gate."""
    reg = {r["lesion_id"]: r for r in registry}
    P = pairs(a, b, lesion_ids)
    x, y = [p["x"] for p in P], [p["y"] for p in P]
    outcomes = _outcomes_by_patient(P, reg)
    lo, hi = bootstrap_ci(outcomes, n_boot, seed) if P else (NAN, NAN)
    strata = strata_report(P, reg, n_boot, seed)
    band0 = strata["band"].get("0")
    raw0, ci0 = (band0["raw"], band0["raw_ci95"][0]) if band0 else (NAN, NAN)
    return {"n_pairs": len(P), "n_patients": len(outcomes), "raw": raw_agreement(x, y), "raw_ci95": [lo, hi],
            "kappa": cohen_kappa(x, y), "ac1": gwet_ac1(x, y, categories=AC1_CATEGORIES), "ac1_categories": len(AC1_CATEGORIES),
            "positive_agreement": positive_agreement(x, y), "confusion": confusion(x, y), "set_agreement": set_agreement(P),
            "lesion_type_agreement": field_agreement(a, b, "lesion_type", lesion_ids),
            "side_agreement": field_agreement(a, b, "side", lesion_ids),
            "strata": strata, "gate_r7": gate_r7(lo, raw0, ci0),
            "time": {"reader_a": time_summary(a if a_rows is None else a_rows, lesion_ids),
                     "reader_b": time_summary(b if b_rows is None else b_rows, lesion_ids)}}

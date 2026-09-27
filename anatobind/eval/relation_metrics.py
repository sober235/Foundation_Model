"""Gate R1 statistics (v2.6 §12; spec 2026-09-27 §6): set-valued correctness, the singleton-subset metrics, rescue /
harm / net rescue against a comparator, patient-level cluster bootstrap, exact McNemar, strata and the Δd curve, and the
§12.4 GO rule as a function. Nothing here knows where predictions or labels came from."""
import numpy as np
from scipy.stats import binomtest

from anatobind.relation.table import N_OUT, N_SLOTS, NONE_SLOT

N_BOOT = 10_000
SEED = 0
DELTA_D_EDGES = (0.0, 1.0, 2.0, 4.0, 8.0, float("inf"))


def predicted_slot(probs):
    return np.asarray(probs).argmax(1)


def is_correct(probs, acceptable):
    pred = predicted_slot(probs)
    acc = np.asarray(acceptable, bool)
    ok = pred < NONE_SLOT
    out = np.zeros(len(pred), bool)
    out[ok] = acc[np.nonzero(ok)[0], pred[ok]]
    return out


def topk_correct(probs, acceptable, k=2):
    p = np.asarray(probs)[:, :N_SLOTS]
    top = np.argsort(-p, 1)[:, :k]
    acc = np.asarray(acceptable, bool)
    return np.array([acc[i, top[i]].any() for i in range(len(p))])


def _per_class(truth, pred):
    """Recall, precision and F1 per slot over the classes present in the truth; balanced accuracy = mean recall."""
    recalls, f1s = [], []
    for c in range(N_SLOTS):
        t, p = truth == c, pred == c
        if not t.any():
            continue
        tp = float((t & p).sum())
        recall = tp / t.sum()
        precision = tp / p.sum() if p.any() else 0.0
        recalls.append(recall)
        f1s.append(0.0 if tp == 0 else 2 * precision * recall / (precision + recall))
    return float(np.mean(recalls)), float(np.mean(f1s))


def summary(probs, acceptable):
    acc = np.asarray(acceptable, bool)
    correct = is_correct(probs, acc)
    single = acc.sum(1) == 1
    out = {"n": int(len(correct)), "accuracy": float(correct.mean()) if len(correct) else float("nan"),
           "singleton_rate": float(single.mean()) if len(correct) else float("nan"),
           "top2_accuracy": float(topk_correct(probs, acc, 2).mean()) if len(correct) else float("nan")}
    truth, pred = acc[single].argmax(1), predicted_slot(probs)[single]
    conf = np.zeros((N_SLOTS, N_OUT), int)
    for t, p in zip(truth, pred):
        conf[t, p] += 1
    bal, f1 = _per_class(truth, pred) if single.any() else (float("nan"), float("nan"))
    out["singleton"] = {"n": int(single.sum()), "accuracy": float(correct[single].mean()) if single.any() else float("nan"),
                        "macro_f1": f1, "balanced_accuracy": bal, "confusion": conf.tolist()}
    return out


def rescue_harm(correct_model, correct_comp):
    m, c = np.asarray(correct_model, bool), np.asarray(correct_comp, bool)
    rescue, harm = int((m & ~c).sum()), int((~m & c).sum())
    return {"rescue": rescue, "harm": harm, "net": rescue - harm, "net_rate": (rescue - harm) / len(m) if len(m) else 0.0}


def patient_bootstrap(values, patients, n_boot=N_BOOT, seed=SEED, alpha=0.05):
    """Mean of ``values`` with a percentile CI from resampling patients with replacement (clusters of lesions)."""
    v, p = np.asarray(values, float), np.asarray(patients)
    ids = np.unique(p)
    groups = [v[p == i] for i in ids]
    sums = np.array([g.sum() for g in groups])
    counts = np.array([len(g) for g in groups], float)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(ids), size=(n_boot, len(ids)))
    stat = sums[draws].sum(1) / counts[draws].sum(1)
    return float(v.mean()), float(np.quantile(stat, alpha / 2)), float(np.quantile(stat, 1 - alpha / 2))


def mcnemar(b, c):
    """Exact two-sided p-value for the discordant counts (rescue b, harm c)."""
    if b + c == 0:
        return 1.0
    return float(binomtest(min(b, c), b + c, 0.5, alternative="two-sided").pvalue)


def by_stratum(values, keys):
    v, k = np.asarray(values, float), np.asarray(keys)
    return {str(s): {"n": int((k == s).sum()), "mean": float(v[k == s].mean())} for s in np.unique(k)}


def delta_d_curve(values, delta_d, patients, edges=DELTA_D_EDGES, n_boot=N_BOOT, seed=SEED):
    v, d, p = np.asarray(values, float), np.asarray(delta_d, float), np.asarray(patients)
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (d >= lo) & (d < hi)
        if sel.any():
            mean, ci_lo, ci_hi = patient_bootstrap(v[sel], p[sel], n_boot, seed)
        else:
            mean = ci_lo = ci_hi = None
        out.append({"bin": f"[{lo}, {hi})", "n": int(sel.sum()), "mean": mean, "lo": ci_lo, "hi": ci_hi})
    return out


def gate_r1(model_probs, comparators, acceptable, patients, n_boot=N_BOOT, seed=SEED):
    """v2.6 §12.4: net rescue vs Bgeo+ has a CI lower bound > 0, and the model beats Bprior, Bgeo+ and B2 in accuracy."""
    acc = np.asarray(acceptable, bool)
    cm = is_correct(model_probs, acc)
    correct = {name: is_correct(p, acc) for name, p in comparators.items()}
    diff = cm.astype(float) - correct["bgeo"].astype(float)
    mean, lo, hi = patient_bootstrap(diff, patients, n_boot, seed)
    accuracy = {"model": float(cm.mean()), **{k: float(v.mean()) for k, v in correct.items()}}
    out = {"accuracy": accuracy, "net_rescue_vs_bgeo": {"mean": mean, "lo": lo, "hi": hi, **rescue_harm(cm, correct["bgeo"])},
           "net_rescue_ci_low_gt_0": bool(lo > 0), "gt_bprior": bool(accuracy["model"] > accuracy["bprior"]),
           "gt_bgeo": bool(accuracy["model"] > accuracy["bgeo"]), "gt_b2": bool(accuracy["model"] > accuracy["b2"])}
    out["go"] = bool(out["net_rescue_ci_low_gt_0"] and out["gt_bprior"] and out["gt_bgeo"] and out["gt_b2"])
    return out

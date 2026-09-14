"""风险–覆盖、AURC、阈值选择与 H2/H3（leg 2 spec 3.3、3.4）。

labels 用 1 = 正确、0 = 错误；risk 是被接受样本中错误的比例。
"""
import numpy as np

RISK_TARGET = 0.05
REPS = 10_000


def risk_coverage(scores, labels):
    scores, labels = np.asarray(scores, float), np.asarray(labels, float)
    order = np.argsort(-scores)
    lab = labels[order]
    k = np.arange(1, len(lab) + 1)
    coverage = k / len(lab)
    risk = 1.0 - np.cumsum(lab) / k
    return coverage, risk


def aurc(scores, labels):
    coverage, risk = risk_coverage(scores, labels)
    return float(np.trapezoid(risk, coverage) / (coverage[-1] - coverage[0])) if len(coverage) > 1 else float("nan")


def threshold_at_risk(scores, labels, r_max=RISK_TARGET):
    """满足风险上限的最小分数阈值；无解返回 inf。"""
    scores, labels = np.asarray(scores, float), np.asarray(labels, float)
    order = np.argsort(-scores)
    lab, sc = labels[order], scores[order]
    k = np.arange(1, len(lab) + 1)
    risk = 1.0 - np.cumsum(lab) / k
    ok = np.nonzero(risk <= r_max)[0]
    return float(sc[ok[-1]]) if len(ok) else float("inf")


def bootstrap_delta(per_patient, fn, reps=REPS, seed=0, alpha=0.05):
    """per_patient: list[list[row]]，按患者重采样；fn(rows) -> 标量。返回 (delta, lo, hi)。"""
    rng = np.random.default_rng(seed)
    base = fn([r for rows in per_patient for r in rows])
    out = np.empty(reps)
    n = len(per_patient)
    for b in range(reps):
        pick = rng.integers(0, n, n)
        out[b] = fn([r for i in pick for r in per_patient[i]])
    return float(base), float(np.quantile(out, alpha / 2)), float(np.quantile(out, 1 - alpha / 2))


VIEW_ORDER = ("clean", "noise_q1", "noise_q2", "noise_q3", "us4", "us8", "us16")
GATE_BINS = VIEW_ORDER[1:]          # H2 只在退化档位内部判定；clean 一并报告但不参与判定


def _by_patient(rows):
    groups = {}
    for r in rows:
        groups.setdefault(r["patient"], []).append(r)
    return list(groups.values())


def _delta_aurc(rows):
    if not rows or len(set(r["correct"] for r in rows)) < 2:
        return float("nan")
    y = [r["correct"] for r in rows]
    return aurc([r["head_score"] for r in rows], y) - aurc([r["peak_score"] for r in rows], y)


def h2(rows, bins=GATE_BINS, reps=REPS, seed=0):
    """每个退化档位内部：可靠性头的 AURC 减去峰值得分基线的 AURC。越负越好。"""
    out = {}
    for v in VIEW_ORDER:
        sel = [r for r in rows if r["view"] == v]
        d, lo, hi = bootstrap_delta(_by_patient(sel), _delta_aurc, reps, seed) if sel else (float("nan"),) * 3
        out[v] = {"n": len(sel), "delta_aurc": d, "ci95": [lo, hi],
                  "pass": bool(np.isfinite(d) and d < 0 and hi < 0)}
    out["pass"] = all(out[v]["pass"] for v in bins)
    return out


def _head_coverage(rows, key, r_max=RISK_TARGET):
    if not rows:
        return 0.0
    s = np.array([r[key] for r in rows], float)
    y = np.array([r["correct"] for r in rows], float)
    tau = threshold_at_risk(s, y, r_max)
    return float((s >= tau).mean()) if np.isfinite(tau) else 0.0


def _rule_coverage(rows, r_max=RISK_TARGET):
    """按加速倍数一刀切：取风险仍 <= r_max 的最宽松截断，返回它保留的覆盖率。"""
    if not rows:
        return 0.0
    for k in range(len(VIEW_ORDER), 0, -1):
        keep = set(VIEW_ORDER[:k])
        sel = [r for r in rows if r["view"] in keep]
        if sel and 1.0 - float(np.mean([r["correct"] for r in sel])) <= r_max:
            return len(sel) / len(rows)
    return 0.0


def h3(rows, reps=REPS, seed=0, r_max=RISK_TARGET):
    """固定风险下，学习闸门的覆盖率是否高于一刀切规则；同时报告 spec 3.2 的最小值基线。"""
    groups = _by_patient(rows)
    res = {"risk_target": r_max,
           "coverage_head": _head_coverage(rows, "head_score", r_max),
           "coverage_rule": _rule_coverage(rows, r_max),
           "coverage_min_baseline": _head_coverage(rows, "min_lesion", r_max)}
    for name, fn in (("vs_rule", lambda rs: _head_coverage(rs, "head_score", r_max) - _rule_coverage(rs, r_max)),
                     ("vs_min", lambda rs: _head_coverage(rs, "head_score", r_max)
                      - _head_coverage(rs, "min_lesion", r_max))):
        d, lo, hi = bootstrap_delta(groups, fn, reps, seed)
        res[name] = {"delta_coverage": d, "ci95": [lo, hi], "pass": bool(d > 0 and lo > 0)}
    res["pass"] = res["vs_rule"]["pass"]
    return res

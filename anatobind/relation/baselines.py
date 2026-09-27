"""The non-torch arms of spec 2026-09-27 §5: B0 (the C1 rule on the table), Bprior (smoothed counts) and Bgeo+
(sklearn learners on the flat feature vector). Every arm returns (N, 8) probabilities restricted to the candidate set."""
import numpy as np

from anatobind.relation.table import N_OUT, N_SLOTS, features_flat, mask_to_candidates

PRIOR_VARIANTS = ("majority", "type", "type_side", "type_side_location")


def target_weights(acceptable):
    acc = np.asarray(acceptable, float)
    n = acc.sum(1, keepdims=True)
    return np.where(n > 0, acc / np.where(n > 0, n, 1.0), 0.0)


def b0_probs(table):
    p = np.zeros((len(table), N_OUT))
    p[np.arange(len(table)), table.c1_slot()] = 1.0
    return mask_to_candidates(p, table.candidates())


class PriorModel:
    def __init__(self, variant, alpha=1.0):
        if variant not in PRIOR_VARIANTS:
            raise ValueError(variant)
        self.variant, self.alpha = variant, float(alpha)
        self.counts, self.total = {}, np.zeros(N_SLOTS)

    def key(self, row):
        if self.variant == "majority":
            return ()
        if self.variant == "type":
            return (row["lesion_type"],)
        if self.variant == "type_side":
            return (row["lesion_type"], row["side"])
        return (row["lesion_type"], row["coarse_location"])          # coarse_location already carries the side (P9)

    def fit(self, table, acceptable):
        w = target_weights(acceptable)
        self.counts, self.total = {}, np.zeros(N_SLOTS)
        for row, wi in zip(table.rows, w):
            self.counts[self.key(row)] = self.counts.get(self.key(row), np.zeros(N_SLOTS)) + wi
            self.total += wi
        return self

    def predict(self, table):
        p = np.zeros((len(table), N_OUT))
        for i, row in enumerate(table.rows):
            c = self.counts.get(self.key(row), self.total)
            p[i, :N_SLOTS] = (c + self.alpha) / (c.sum() + N_SLOTS * self.alpha)
        return mask_to_candidates(p, table.candidates())


# --- Bgeo+ (spec §5.3) ---------------------------------------------------------------------------------------------
GEO_KINDS = ("lr", "hgb", "mlp")
GEO_GRIDS = {"lr": [{"C": c} for c in (0.01, 0.1, 1.0, 10.0)],
             "hgb": [{"learning_rate": lr, "max_depth": d} for lr in (0.03, 0.1) for d in (3, 6)],
             "mlp": [{"alpha": a} for a in (1e-3, 1e-4)]}


def expand_sets(X, acceptable):
    """One training row per (lesion, acceptable slot) with weight 1 / |Y|; unlabeled lesions are dropped."""
    w = target_weights(acceptable)
    rows, ys, ws = [], [], []
    for i in range(len(w)):
        for s in np.nonzero(w[i] > 0)[0]:
            rows.append(i)
            ys.append(int(s))
            ws.append(float(w[i, s]))
    return np.asarray(X)[rows], np.array(ys, int), np.array(ws, float)


class GeoLearner:
    def __init__(self, kind, params, seed=0):
        from sklearn.ensemble import HistGradientBoostingClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.neural_network import MLPClassifier
        from sklearn.preprocessing import StandardScaler
        if kind not in GEO_KINDS:
            raise ValueError(kind)
        self.kind, self.params, self.scaler = kind, dict(params), StandardScaler()
        if kind == "lr":
            self.model = LogisticRegression(C=params["C"], max_iter=2000)
        elif kind == "hgb":
            self.model = HistGradientBoostingClassifier(learning_rate=params["learning_rate"], max_depth=params["max_depth"], random_state=seed)
        else:
            self.model = MLPClassifier(hidden_layer_sizes=(64, 64), alpha=params["alpha"], early_stopping=True, max_iter=500, random_state=seed)

    def fit(self, X, acceptable):
        Xr, y, w = expand_sets(X, acceptable)
        Xs = self.scaler.fit_transform(Xr)
        self.model.fit(Xs, y, sample_weight=w)       # all three accept sample_weight in scikit-learn 1.9
        return self

    def predict_probs(self, X, candidates):
        p = np.zeros((len(X), N_OUT))
        probs = self.model.predict_proba(self.scaler.transform(np.asarray(X)))
        for j, c in enumerate(self.model.classes_):
            p[:, int(c)] = probs[:, j]
        return mask_to_candidates(p, candidates)

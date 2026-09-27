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

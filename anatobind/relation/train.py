"""Training of the torch arms (spec 2026-09-27 §5.7–5.9): AdamW, the set-valued loss, flip-consistent augmentation,
early stopping on the inner validation fold, refit on the outer training fold for the median best epoch, and the
stage-2 hook that starts from stage-1 weights."""
import copy
import time

import numpy as np
import torch

from anatobind.relation.cv import (
    empty_preds, inner_folds, labels_for_fold, outer_folds, select_config, set_accuracy, trainable_rows,
)
from anatobind.relation.encoder import TORCH_CONFIGS, augment, config_key, crop
from anatobind.relation.labels import acceptable_matrix
from anatobind.relation.models import B1Model, B2Model, set_nll
from anatobind.relation.table import N_OUT, N_SLOTS, features_per_slot, mask_to_candidates


def acceptable8(acc7):
    a = np.asarray(acc7, bool)
    return np.concatenate([a, np.zeros((len(a), 1), bool)], 1)


def _model(arm, in_ch):
    if arm == "b1":
        return B1Model(in_ch)
    if arm == "b2":
        return B2Model(in_ch)
    raise ValueError(arm)


def _forward(model, arm, xb, gb, pb):
    return model(xb, gb, pb) if arm == "b1" else model(xb)


def _tensors(device, *arrays):
    return [torch.from_numpy(np.ascontiguousarray(a)).to(device) for a in arrays]


def predict_torch_arm(arm, state, x, geo, present, idx, device, batch=256):
    model = _model(arm, x.shape[1]).to(device)
    model.load_state_dict(state)
    model.eval()
    out = []
    with torch.no_grad():
        for s in range(0, len(idx), batch):
            b = idx[s:s + batch]
            xb, gb, pb = _tensors(device, x[b].astype(np.float32), geo[b].astype(np.float32), present[b])
            out.append(_forward(model, arm, xb, gb, pb).softmax(-1).cpu().numpy())
    probs = np.concatenate(out) if out else np.zeros((0, N_OUT))
    return mask_to_candidates(probs, present[idx])


def fit_torch_arm(arm, x, geo, present, acc8, train_idx, val_idx, seed, device, epochs=40, patience=8, init_state=None,
                  lr=1e-3, wd=1e-4, batch=64):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = _model(arm, x.shape[1]).to(device)
    if init_state is not None:
        model.load_state_dict(init_state)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    best_state, best_acc, best_epoch, bad = copy.deepcopy(model.state_dict()), -1.0, 0, 0
    for epoch in range(1, epochs + 1):
        model.train()
        order = rng.permutation(train_idx)
        for s in range(0, len(order), batch):
            b = order[s:s + batch]
            xa, ga = augment(x[b], geo[b], rng)
            xb, gb, pb, yb = _tensors(device, xa, ga, present[b], acc8[b])
            loss = set_nll(_forward(model, arm, xb, gb, pb), yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
        if val_idx is None:
            continue
        val_acc = set_accuracy(predict_torch_arm(arm, model.state_dict(), x, geo, present, val_idx, device), acc8[val_idx][:, :N_SLOTS])
        if val_acc > best_acc:
            best_state, best_acc, best_epoch, bad = copy.deepcopy(model.state_dict()), val_acc, epoch, 0
        else:
            bad += 1
            if bad >= patience:
                break
    if val_idx is None:
        return copy.deepcopy(model.state_dict()), epochs, float("nan")
    return best_state, best_epoch, best_acc


def run_torch_arm(arm, table, patches, labels, configs=TORCH_CONFIGS, seed=0, device="cpu", epochs=40, patience=8, inner_k=5,
                  init=None, log=None):
    geo, present, patients = features_per_slot(table), table.candidates(), table.patients()
    by_key = {config_key(c): c for c in configs}
    xs = {key: crop(patches["image"], patches["mask"], c["px"], c["slices"]) for key, c in by_key.items()}
    preds = empty_preds(table)
    record = {"arm": arm, "init_from_states": init is not None, "folds": {}}
    states = {}
    for k, tr, te in outer_folds(table):
        t0 = time.time()
        has, acc = acceptable_matrix(labels_for_fold(labels, k), table.lesion_id)
        acc8 = acceptable8(acc)
        ok = trainable_rows(has, acc, present)
        n_untrainable = int((has[tr] & ~ok[tr]).sum())
        tr = tr[ok[tr]]
        init_state = None
        keys = list(by_key)
        if init is not None:                                     # stage 2: fine-tune the stage-1 model of this fold
            init_key, init_state = init[k]
            keys = [init_key]
        scores, best_epochs, inner_epochs = [], {}, {}
        for key in keys:
            accs, eps = [], []
            for itr, ival in inner_folds(patients[tr], inner_k, seed):
                _, ep, va = fit_torch_arm(arm, xs[key], geo, present, acc8, tr[itr], tr[ival], seed, device, epochs, patience, init_state)
                accs.append(va)
                eps.append(ep)
            scores.append((key, float(np.mean(accs))))
            inner_epochs[key] = eps
            best_epochs[key] = max(1, int(round(float(np.median(eps)))))
        chosen, rec = select_config(scores)
        state, _, _ = fit_torch_arm(arm, xs[chosen], geo, present, acc8, tr, None, seed, device, best_epochs[chosen], patience, init_state)
        preds["probs"][te] = predict_torch_arm(arm, state, xs[chosen], geo, present, te, device)
        preds["config"][te] = chosen
        record["folds"][k] = {**rec, "epochs": best_epochs[chosen], "inner_epochs": inner_epochs, "n_untrainable": n_untrainable}
        states[k] = {n: v.detach().cpu() for n, v in state.items()}
        if log:
            log(f"{arm} fold {k}: chose {chosen} (tie {rec['tie']}), refit {best_epochs[chosen]} epochs, "
                f"untrainable {n_untrainable}, {time.time() - t0:.0f} s")
    return preds, record, states

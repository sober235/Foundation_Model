"""Diagnostic (2026-10-10): do C0's step-3000 features hold the tissue information the A head fails to read? A
softmax linear classifier (33 classes: none + 32 entities) on frozen per-voxel features, trained on 6 of the 8
deterministic crops and tested on the other 2 (one per source held out): the full-resolution mask features (32 dims),
the coarse F1-grid mask features (64 dims, upsampled by nearest neighbour) and the image intensity alone, with and
without the local position. Reports test accuracy and the white matter / cortex Dice (left and right merged). CPU."""
import json, sys
import numpy as np, torch
import torch.nn.functional as F
sys.path.insert(0, "/data0/congcong/code/Project_Doing/foundation_model")
from anatobind.aur.dataset import load_volume, make_crop
from anatobind.aur import infer as I
from anatobind.aur.labels import ENTITY_LABELS
torch.set_num_threads(12)
torch.manual_seed(0)
RUN = "/data2/congcong/data/FM_data/derived/aur/ssl_runs/c0_stage2_240k_20261010_0903"
rows = json.load(open("/data2/congcong/data/FM_data/derived/aur/samples_1mm_v1.json"))
train = [r for r in rows if r["split"] == "train" and r.get("a_supervised", True)]
rng0 = np.random.default_rng(7)
by = {}
for r in train:
    by.setdefault(r["source"], []).append(r)
pick = [v[int(i)] for s, v in sorted(by.items()) for i in rng0.choice(len(v), 2, replace=False)]
m, _ = I.model_from_export(f"{RUN}/aur_stage2_step2000.pt", torch.device("cpu"))
m.load_state_dict(torch.load(f"{RUN}/resume_step3000.pt", map_location="cpu", weights_only=False)["model"], strict=True)
m.eval()
E = lambda l: ENTITY_LABELS.index(l) + 1
feats = {"full32": [], "coarse64": [], "intensity": [], "intensity+local": []}
ys, src = [], []
rs = np.random.default_rng(0)
for k, r in enumerate(pick):
    vol = load_volume(r)
    c = make_crop(vol, np.random.default_rng([3, k]), do_augment=False, lesion_centred=False)
    with torch.no_grad():
        out = m(c["image"][None], c["valid"][None], c["coords"][None], c["local"][None])
        full = out["pix"]["full"][0].float()                                          # (32, D, H, W)
        coarse = F.interpolate(out["pix"]["coarse"].float(), size=full.shape[1:], mode="nearest")[0]
    ent = c["entity"].numpy(); keep = (c["valid"].numpy() > 0.5) & (~c["a_ignore"].numpy())
    idx = np.flatnonzero(keep.ravel()); idx = rs.choice(idx, size=min(40000, idx.size), replace=False)
    feats["full32"].append(full.flatten(1)[:, idx].T.numpy())
    feats["coarse64"].append(coarse.flatten(1)[:, idx].T.numpy())
    img = c["image"][0].flatten()[idx].numpy()[:, None]
    loc = c["local"].flatten(1)[:, idx].T.numpy()
    feats["intensity"].append(img)
    feats["intensity+local"].append(np.concatenate([img, loc], 1))
    ys.append(ent.ravel()[idx]); src.append(r["source"])
test = [1, 3, 5, 7]                                                                    # the second crop of every source
tr = [i for i in range(len(pick)) if i not in test]
def dice(t, p, ls):
    a, b = np.isin(t, ls), np.isin(p, ls)
    return 2 * float((a & b).sum()) / max(float(a.sum() + b.sum()), 1)
for name, fs in feats.items():
    X = torch.tensor(np.concatenate([fs[i] for i in tr]), dtype=torch.float32); Y = torch.tensor(np.concatenate([ys[i] for i in tr]), dtype=torch.long)
    Xt = torch.tensor(np.concatenate([fs[i] for i in test]), dtype=torch.float32); Yt = np.concatenate([ys[i] for i in test])
    mu, sd = X.mean(0, keepdim=True), X.std(0, keepdim=True).clamp(min=1e-3)
    X, Xt = (X - mu) / sd, (Xt - mu) / sd
    lin = torch.nn.Linear(X.shape[1], 33); opt = torch.optim.Adam(lin.parameters(), lr=1e-2)
    for _ in range(300):
        opt.zero_grad(); loss = F.cross_entropy(lin(X), Y); loss.backward(); opt.step()
    with torch.no_grad():
        p = lin(Xt).argmax(1).numpy()
    print(f"{name:16} dims {X.shape[1]:3d}  test acc {float((p == Yt).mean()):.3f}  WM Dice {dice(Yt, p, [E(2), E(41)]):.3f}  cortex Dice {dice(Yt, p, [E(3), E(42)]):.3f}  "
          f"L-WM {dice(Yt, p, [E(2)]):.3f} R-WM {dice(Yt, p, [E(41)]):.3f}", flush=True)

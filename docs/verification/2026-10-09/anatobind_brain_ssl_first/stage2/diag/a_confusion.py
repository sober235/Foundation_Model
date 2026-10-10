"""Diagnostic (2026-10-10): where does C0's A head go wrong at step 3000? On the same 8 deterministic training crops:
the confusion between the true entity and the argmax entity inside the true foreground, and the Dice after merging the
left and right entity of each pair (tissue right, side wrong vs tissue wrong). CPU only."""
import json, sys
import numpy as np, torch
sys.path.insert(0, "/data0/congcong/code/Project_Doing/foundation_model")
from anatobind.aur.dataset import load_volume, make_crop
from anatobind.aur import infer as I
from anatobind.aur.labels import ENTITY_LABELS
torch.set_num_threads(12)
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
NAME = {2: "L-WM", 41: "R-WM", 3: "L-ctx", 42: "R-ctx", 4: "L-LV", 43: "R-LV", 24: "CSF", 8: "L-cbctx", 47: "R-cbctx", 16: "stem"}
idx = {ENTITY_LABELS.index(l) + 1: n for l, n in NAME.items()}
order = list(idx)
conf = np.zeros((len(order) + 1, len(order) + 1), np.int64)
pairs = {"WM": (2, 41), "ctx": (3, 42), "LV": (4, 43), "cbctx": (8, 47), "thal": (10, 49), "caud": (11, 50), "put": (12, 51)}
merged = {k: [0, 0] for k in pairs}
side_lr = []
for k, r in enumerate(pick):
    vol = load_volume(r)
    c = make_crop(vol, np.random.default_rng([3, k]), do_augment=False, lesion_centred=False)
    with torch.no_grad():
        out = m(c["image"][None], c["valid"][None], c["coords"][None], c["local"][None])
        arg = m.entity_masks(out)[0].float().argmax(0) + 1
    ent = c["entity"].long()
    fg = (~c["a_ignore"].bool()) & c["valid"].bool() & (ent > 0)
    t, p = ent[fg].numpy(), arg[fg].numpy()
    ti = np.array([order.index(x) if x in idx else len(order) for x in t]); pi = np.array([order.index(x) if x in idx else len(order) for x in p])
    np.add.at(conf, (ti, pi), 1)
    for name, (a, b) in pairs.items():
        ea, eb = ENTITY_LABELS.index(a) + 1, ENTITY_LABELS.index(b) + 1
        tt = np.isin(t, [ea, eb]); pp = np.isin(p, [ea, eb])
        merged[name][0] += int((tt & pp).sum()); merged[name][1] += int(tt.sum() + pp.sum())
    lx = c["local"][2][fg].numpy()                                         # local x of the foreground voxels in [-1, 1]
    side_lr.append((r["source"], round(float(lx.min()), 2), round(float(lx.max()), 2)))
labels = [idx[x] for x in order] + ["other"]
print("rows = truth, columns = argmax (share of the truth row):")
print("        " + " ".join(f"{l:>8}" for l in labels))
for i, l in enumerate(labels):
    s = conf[i].sum()
    print(f"{l:>8} " + " ".join(f"{(conf[i, j] / s if s else 0):8.2f}" for j in range(len(labels))) + f"   n={s}")
print("Dice with left and right merged:", {k: round(2 * v[0] / max(v[1], 1), 3) for k, v in merged.items()})
print("local x range of the foreground per crop:", side_lr)

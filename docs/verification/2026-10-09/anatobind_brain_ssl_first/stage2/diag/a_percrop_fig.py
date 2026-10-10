"""Diagnostic (2026-10-10): C0 step 3000 A head per crop (8 deterministic training crops, no augmentation): Dice of
white matter and cortex with left and right merged, and a figure (middle axial slice: image, SynthSeg-derived truth,
argmax prediction inside the valid region) per crop. CPU only."""
import json, sys
import numpy as np, torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
E = lambda l: ENTITY_LABELS.index(l) + 1
cmap = plt.get_cmap("tab20", 33)
fig, ax = plt.subplots(len(pick), 3, figsize=(9, 3 * len(pick)))
for k, r in enumerate(pick):
    vol = load_volume(r)
    c = make_crop(vol, np.random.default_rng([3, k]), do_augment=False, lesion_centred=False)
    with torch.no_grad():
        out = m(c["image"][None], c["valid"][None], c["coords"][None], c["local"][None])
        arg = (m.entity_masks(out)[0].float().argmax(0) + 1).numpy()
    ent = c["entity"].numpy(); valid = c["valid"].numpy() > 0.5
    fg = valid & (ent > 0) & (~c["a_ignore"].numpy())
    d = {}
    for name, ls in (("WM", (2, 41)), ("ctx", (3, 42))):
        t = np.isin(ent, [E(l) for l in ls]) & fg; p = np.isin(arg, [E(l) for l in ls]) & fg
        d[name] = round(2 * float((t & p).sum()) / max(float(t.sum() + p.sum()), 1), 3)
    print(r["source"], r["sequence"], r["case"], "WM", d["WM"], "ctx", d["ctx"], flush=True)
    z = ent.shape[0] // 2
    ax[k, 0].imshow(c["image"][0, z].numpy(), cmap="gray", vmin=-1, vmax=1); ax[k, 0].set_title(f"{r['source']} {r['sequence']}", fontsize=8)
    ax[k, 1].imshow(np.where(valid[z], ent[z], 0), cmap=cmap, vmin=0, vmax=32, interpolation="nearest"); ax[k, 1].set_title("truth (SynthSeg)", fontsize=8)
    ax[k, 2].imshow(np.where(fg[z], arg[z], 0), cmap=cmap, vmin=0, vmax=32, interpolation="nearest"); ax[k, 2].set_title(f"argmax in fg  WM {d['WM']} ctx {d['ctx']}", fontsize=8)
    for a in ax[k]:
        a.axis("off")
plt.tight_layout(); plt.savefig("/home/congcongliu/figs/anatobind/c0_step3000_A_percrop.png", dpi=90)
print("saved")

"""Diagnostic (2026-10-10): is C0's low A Dice a calibration effect of the 0.5 threshold (logits negative everywhere)
or a failure to tell entities apart? On 8 deterministic training crops (no augmentation), for the step-2000 export and
the step-3000 resume: per-entity Dice of the training rule (logit > 0), Dice of the argmax over the 32 entities inside
the true foreground, and the share of foreground voxels whose largest logit is > 0. CPU only."""
import json, sys
import numpy as np, torch
sys.path.insert(0, "/data0/congcong/code/Project_Doing/foundation_model")
from anatobind.aur.dataset import load_volume, make_crop
from anatobind.aur.model import AnatoBindBrain
from anatobind.aur import infer as I
torch.set_num_threads(12)
RUN = "/data2/congcong/data/FM_data/derived/aur/ssl_runs/c0_stage2_240k_20261010_0903"
rows = json.load(open("/data2/congcong/data/FM_data/derived/aur/samples_1mm_v1.json"))
train = [r for r in rows if r["split"] == "train" and r.get("a_supervised", True)]
rng0 = np.random.default_rng(7)
by = {}
for r in train:
    by.setdefault(r["source"], []).append(r)
pick = [v[int(i)] for s, v in sorted(by.items()) for i in rng0.choice(len(v), 2, replace=False)]
crops = []
for k, r in enumerate(pick):
    vol = load_volume(r)
    c = make_crop(vol, np.random.default_rng([3, k]), do_augment=False, lesion_centred=False)
    crops.append((r["source"], r["sequence"], c))
print("crops:", [(s, q) for s, q, _ in crops], flush=True)

def model_at(which):
    m, meta = I.model_from_export(f"{RUN}/aur_stage2_step2000.pt", torch.device("cpu"))
    if which == "step3000":
        ck = torch.load(f"{RUN}/resume_step3000.pt", map_location="cpu", weights_only=False)
        m.load_state_dict(ck["model"], strict=True)
    return m.eval()

for which in ("step2000", "step3000"):
    model = model_at(which)
    tot = {"thr_inter": np.zeros(32), "thr_sum": np.zeros(32), "arg_inter": np.zeros(32), "arg_sum": np.zeros(32), "tgt": np.zeros(32)}
    fg_pos, fg_n, maxlog = 0, 0, []
    for s, q, c in crops:
        with torch.no_grad():
            out = model(c["image"][None], c["valid"][None], c["coords"][None], c["local"][None])
            logit = model.entity_masks(out)[0].float()                      # (32, D, H, W)
        ent = c["entity"].long()                                            # 0 none, 1..32
        keep = (~c["a_ignore"].bool()) & c["valid"].bool()
        fg = keep & (ent > 0)
        mx, arg = logit.max(0)
        fg_pos += int((mx[fg] > 0).sum()); fg_n += int(fg.sum()); maxlog.append(mx[fg].numpy())
        for k in range(32):
            t = (ent == k + 1) & keep
            p_thr = (logit[k] > 0) & keep
            p_arg = (arg == k) & fg
            tot["thr_inter"][k] += float((p_thr & t).sum()); tot["thr_sum"][k] += float(p_thr.sum() + t.sum())
            tot["arg_inter"][k] += float((p_arg & t).sum()); tot["arg_sum"][k] += float(p_arg.sum() + t.sum())
            tot["tgt"][k] += float(t.sum())
    present = tot["tgt"] > 0
    thr = 2 * tot["thr_inter"] / np.maximum(tot["thr_sum"], 1)
    arg = 2 * tot["arg_inter"] / np.maximum(tot["arg_sum"], 1)
    big = np.argsort(-tot["tgt"])[:6]
    ml = np.concatenate(maxlog)
    print(f"== {which}: entities present {int(present.sum())}; macro Dice logit>0 {thr[present].mean():.3f}; macro Dice argmax-in-foreground {arg[present].mean():.3f}; "
          f"foreground voxels with max logit > 0: {fg_pos / max(fg_n, 1):.3f}; max-logit quantiles (5/50/95%) {np.percentile(ml, [5, 50, 95]).round(2).tolist()}")
    print("   six largest entities (index, voxels, Dice logit>0, Dice argmax-in-fg):", [(int(k + 1), int(tot['tgt'][k]), round(float(thr[k]), 3), round(float(arg[k]), 3)) for k in big], flush=True)

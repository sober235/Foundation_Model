"""Whole-volume inference of AnatoBind-Brain (spec §8; SSL-first plan T12).

The volume (z, y, x, normalised like a training crop) is covered by crops of the training size that overlap by half
and are weighted with a Gaussian; windows that leave the volume carry the constant -1 and valid = 0 there, like the
training crops at the image border. Every window gives the entity masks (32 sigmoids), the event masks of the present
queries (presence > 0.5) and the sequence logits; the entity probabilities and the lesion probability (the largest
present event mask) are accumulated with the weights, the entity map is the argmax where the largest entity
probability reaches 0.5 (none elsewhere), the lesion instances are the 26-connected components of the lesion map above
a threshold that reach the volume floor, each scored by its mean probability. A lesion's host comes from one crop
centred on it: the event query whose thresholded mask overlaps the instance most (ties and zero overlap broken by the
soft overlap, and flagged) is bound by the relation head; the crop is what the model sees, so an instance larger than
the crop is bound by its part inside it (`in_crop_share`). The heads run under the same bf16 autocast as the backbone
(on CUDA), as in training. Everything here reads pseudo-label-trained heads: NOT_EVIDENCE."""
import itertools
import json
from pathlib import Path

import nibabel as nib
import numpy as np
import torch

from anatobind.aur.crops import coordinates_mm, crop_window, extract, local_coordinates, normalise, to_zyx
from anatobind.aur.labels import HOST_NAMES, HOST_SIDE, HOST_TISSUE, NO_HOST, N_ENTITIES, SEQ_TYPES
from anatobind.aur.model import AnatoBindBrain
from anatobind.eval.lesion_components import component_rows, components
from anatobind.infer.brain_disease import HOST_SHORT_ZH, SIDE_ZH, volume_text

PRESENCE_THRESHOLD = 0.5
ENTITY_THRESHOLD = 0.5
LESION_THRESHOLD = 0.3                 # the map threshold of the evaluated operating point (eval.INSTANCE_THRESHOLD)
ANATOMY_SOURCE = "AnatoBind A head (NOT_EVIDENCE)"
NOWHERE_ZH = "未能定位的区域"
NOTHING_ZH = "未见异常"
MAX_SENTENCE_LESIONS = 5
FORCE_AUTOCAST = False                 # tests: run the bf16 autocast path on the CPU


def windows(shape, crop, overlap=0.5):
    """Windows [(start, stop)] * 3 covering the (z, y, x) volume: stride = crop x (1 - overlap), the last window of
    an axis ends at the volume's edge; an axis no longer than the crop gets one window centred on the volume."""
    if not 0.0 <= overlap < 1.0:
        raise ValueError(f"overlap must be in [0, 1), got {overlap}")
    axes = []
    for n, c in zip(shape, crop):
        if n <= c:
            axes.append([-((c - n) // 2)])
            continue
        stride = max(1, int(round(c * (1.0 - overlap))))
        starts = list(range(0, n - c + 1, stride))
        if starts[-1] != n - c:
            starts.append(n - c)
        axes.append(starts)
    return [[(a, a + c) for a, c in zip(starts, crop)] for starts in itertools.product(*axes)]


def gaussian_weight(crop, sigma_scale=0.125):
    """(D, H, W) float32 weights, 1 at the centre, Gaussian with sigma = axis length x sigma_scale, never 0."""
    axes = []
    for c in crop:
        i = np.arange(c, dtype=np.float32) - c / 2.0
        axes.append(np.exp(-0.5 * (i / (c * sigma_scale)) ** 2).astype(np.float32))
    return (axes[0][:, None, None] * axes[1][None, :, None] * axes[2][None, None, :]).astype(np.float32)


def prepare(image, affine, window):
    """One window of a volume as the model's four arrays (image with -1 outside, validity, mm coordinates, local)."""
    img, valid = extract(image, window, fill=-1.0)
    coords = coordinates_mm(window, affine=affine)
    local = local_coordinates(window, image.shape)
    return (img.astype(np.float32), valid.astype(np.float32), np.ascontiguousarray(coords, dtype=np.float32),
            np.ascontiguousarray(local, dtype=np.float32))


def _tensors(arrays, device):
    img, valid, coords, local = (np.stack(a) for a in zip(*arrays))
    return (torch.from_numpy(img)[:, None].to(device), torch.from_numpy(valid).to(device), torch.from_numpy(coords).to(device),
            torch.from_numpy(local).to(device))


def _autocast(device):
    enabled = device.type == "cuda" or FORCE_AUTOCAST
    return torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=enabled)


def _inside(window, shape):
    """(destination slices inside the volume, source slices inside the crop) of a window that may leave the volume."""
    dst, src = [], []
    for (a, b), n in zip(window, shape):
        lo, hi = max(a, 0), min(b, n)
        dst.append(slice(lo, hi))
        src.append(slice(lo - a, hi - a))
    return tuple(dst), tuple(src)


@torch.no_grad()
def predict_volume(model, image, affine, crop, device, batch_size=1, overlap=0.5, presence_threshold=PRESENCE_THRESHOLD,
                   entity_threshold=ENTITY_THRESHOLD):
    """image (z, y, x) float32 normalised; affine the RAS affine of the (x, y, z) NIfTI. Returns the entity map
    (uint8, 0 none, i + 1 for entity i), the largest entity probability per voxel (float16), the lesion probability
    (float32), the mean sequence probabilities, the mean entity presence over the windows and the number of windows.
    The accumulators live on the device (32 x volume float32: 1.5 GB for 256 x 256 x 176) so that no full-resolution
    mask crosses to the host; only the three result maps do."""
    shape = tuple(image.shape)
    crop = tuple(crop)
    ws = windows(shape, crop, overlap)
    weight = torch.from_numpy(gaussian_weight(crop)).to(device)
    ent_acc = torch.zeros((N_ENTITIES,) + shape, device=device)
    les_acc = torch.zeros(shape, device=device)
    wsum = torch.zeros(shape, device=device)
    seq = torch.zeros(len(SEQ_TYPES), device=device, dtype=torch.float64)
    presence = torch.zeros(N_ENTITIES, device=device, dtype=torch.float64)
    model.eval()
    for i in range(0, len(ws), max(1, batch_size)):
        group = ws[i:i + max(1, batch_size)]
        arrays = [prepare(image, affine, w) for w in group]
        img, valid, coords, local = _tensors(arrays, device)
        with _autocast(device):
            out = model(img, valid, coords, local)
            ent = model.entity_masks(out).float().sigmoid()
            ev_pres = out["event_presence"].float().sigmoid()
            ev = model.event_masks(out).float().sigmoid()
            seq += torch.softmax(out["seq_logits"].float(), -1).sum(0)
            presence += out["entity_presence"].float().sigmoid().sum(0)
        for b, w in enumerate(group):
            dst, src = _inside(w, shape)
            wb = weight[src]
            ent_acc[(slice(None),) + dst] += ent[b][(slice(None),) + src] * wb
            keep = ev_pres[b] > presence_threshold
            if bool(keep.any()):
                les_acc[dst] += ev[b][keep].amax(0)[src] * wb
            wsum[dst] += wb
    wsum.clamp_(min=1e-12)
    ent_acc /= wsum
    les_acc /= wsum
    best, arg = ent_acc.max(0)
    entity = torch.where(best >= entity_threshold, arg + 1, torch.zeros_like(arg)).to(torch.uint8)
    return {"entity": entity.cpu().numpy(), "entity_max_prob": best.to(torch.float16).cpu().numpy(), "lesion_prob": les_acc.float().cpu().numpy(),
            "seq_probs": (seq / len(ws)).float().cpu().numpy(), "entity_presence": (presence / len(ws)).float().cpu().numpy(),
            "n_windows": len(ws)}


def instances_from_probability(prob, voxel_mm3, threshold=LESION_THRESHOLD, score_threshold=None):
    """(instance map int32 with ids 1..N, rows) from a lesion probability map: the 26-connected components of
    prob >= threshold that reach the volume floor, in component order, each with its half-open box in array order,
    voxel count, volume and mean probability (score); with score_threshold the components below it are dropped."""
    comp, n = components(np.asarray(prob) >= threshold)
    inst = np.zeros(comp.shape, np.int32)
    rows = []
    for r in component_rows(comp, n, voxel_mm3, "lesion"):
        if r["ignore"]:
            continue
        sel = comp == r["component"]
        score = float(np.asarray(prob)[sel].mean())
        if score_threshold is not None and score < score_threshold:
            continue
        k = len(rows) + 1
        inst[sel] = k
        rows.append({"instance": k, "box": [int(v) for v in r["box"]], "n_voxels": r["n_voxels"], "volume_mm3": float(r["mm3"]), "score": score})
    return inst, rows


@torch.no_grad()
def bind_instances(model, image, affine, inst, crop, device):
    """One crop centred on each instance of `inst` (int ids, 0 background): the event query whose thresholded mask
    overlaps the instance most inside the crop is bound (zero hard overlap: the query with the largest soft overlap,
    flagged); returns per instance the host index, the 14 host probabilities, the query, its IoU with the instance,
    the window and the share of the instance inside it."""
    ids = [int(k) for k in np.unique(inst) if k > 0]
    if not ids:
        return []
    shape, crop = tuple(image.shape), tuple(crop)
    model.eval()
    rows = []
    for k in ids:
        vox = np.argwhere(inst == k)
        window = crop_window(shape, crop, None, centre=vox.mean(0))
        arrays = [prepare(image, affine, window)]
        img, valid, coords, local = _tensors(arrays, device)
        target, _ = extract(inst == k, window, fill=False)
        target = torch.from_numpy(target).to(device)
        with _autocast(device):
            out = model(img, valid, coords, local)
            prob = model.event_masks(out).float().sigmoid()[0]                                     # (M, D, H, W)
            hard = prob > 0.5
            inter = (hard & target).flatten(1).sum(1).float()
            union = (hard | target).flatten(1).sum(1).float().clamp(min=1.0)
            iou = inter / union
            soft = (prob * target).flatten(1).sum(1)
            zero = bool(iou.max() <= 0)
            q = int(torch.argmax(soft if zero else iou))
            logits = model.bind(out, 0, torch.tensor([q], device=device)).float()
        probs = torch.softmax(logits, -1)[0].cpu().numpy()
        rows.append({"instance": k, "host": int(np.argmax(probs)), "host_probs": probs.astype(np.float32), "query": q,
                     "query_iou": float(iou[q]), "zero_overlap": zero, "window": [(int(a), int(b)) for a, b in window],
                     "in_crop_share": float(target.sum().item() / max(len(vox), 1))})
    return rows


# ---- entry-point helpers (spec §8): the model from an export, the input and its orientation, the record ----

def model_from_export(path, device):
    """The AnatoBindBrain of a Stage II / III export (its configuration is in the export's metadata), in eval mode."""
    ck = torch.load(str(path), map_location="cpu", weights_only=False)
    if "model_state_dict" not in ck:
        raise ValueError(f"{path} is not a Stage II / III export (no model_state_dict; a Stage I file holds only the backbone)")
    meta = ck["meta"]
    cfg = dict(meta.get("config", {}).get("model", {}))
    cfg.pop("use_checkpoint", None)
    model = AnatoBindBrain(**cfg, use_checkpoint=False)
    model.load_state_dict(ck["model_state_dict"], strict=True)
    return model.to(device).eval(), meta


def to_original(canonical_xyz, orig_affine):
    """An (x, y, z) array in the closest-canonical (RAS) layout of an image -> the image's own array layout."""
    ornt = nib.io_orientation(np.asarray(orig_affine, float))
    inv = np.empty_like(ornt)
    for i, (j, flip) in enumerate(ornt):
        inv[int(j)] = (i, flip)
    return nib.orientations.apply_orientation(np.asarray(canonical_xyz), inv)


def load_input(path):
    """(normalised (z, y, x) image of the closest-canonical volume, its RAS affine, the original image). Trailing
    singleton dimensions (a 4D file with one frame) are dropped; anything else than three spatial axes is refused."""
    img = nib.load(str(path))
    canon = nib.as_closest_canonical(img)
    data = np.asarray(canon.dataobj)
    while data.ndim > 3 and data.shape[-1] == 1:
        data = data[..., 0]
    if data.ndim != 3:
        raise ValueError(f"{path}: expected a 3D volume, got shape {tuple(data.shape)}")
    image = normalise(to_zyx(data.astype(np.float32)))
    return image, np.asarray(canon.affine, dtype=np.float64), img


def lesion_records(rows, bound):
    """The instance rows joined with their binding: host name, side, tissue, 14 probabilities."""
    by_id = {b["instance"]: b for b in bound}
    out = []
    for r in rows:
        b = by_id.get(r["instance"])
        host = int(b["host"]) if b else NO_HOST
        out.append({"instance": int(r["instance"]), "host": HOST_NAMES[host], "host_index": host,
                    "host_side": HOST_SIDE[host] if host < NO_HOST else None, "host_tissue": HOST_TISSUE[host] if host < NO_HOST else None,
                    "host_probs": [float(p) for p in b["host_probs"]] if b else None, "volume_mm3": float(r["volume_mm3"]),
                    "n_voxels": int(r["n_voxels"]), "box": [int(v) for v in r["box"]], "score": float(r["score"]),
                    "query_iou": float(b["query_iou"]) if b else None, "zero_overlap": (bool(b["zero_overlap"]) if b.get("zero_overlap") is not None else None) if b else None,
                    "in_crop_share": float(b["in_crop_share"]) if b else None})
    return out


def sentences(lesions, max_lesions=MAX_SENTENCE_LESIONS):
    """"<side><host>存在异常，体积约 …" for the largest lesions, the rest counted (spec §8)."""
    if not lesions:
        return [NOTHING_ZH]
    ordered = sorted(lesions, key=lambda l: -l["volume_mm3"])
    out = []
    for l in ordered[:max_lesions]:
        if l["host_tissue"] is None:
            place = NOWHERE_ZH
        else:
            place = (SIDE_ZH[l["host_side"]] if l["host_side"] else "") + HOST_SHORT_ZH[l["host_tissue"]]
        out.append(f"{place}存在异常，体积{volume_text(l['volume_mm3'])}")
    rest = len(ordered) - max_lesions
    if rest > 0:
        out.append(f"另有 {rest} 处异常未逐一列出")
    return out


def input_boxes(lesions_input):
    """{instance id: half-open box in the input array's own axis order} of a written instance map."""
    from scipy import ndimage
    out = {}
    for k, sl in enumerate(ndimage.find_objects(np.asarray(lesions_input)), start=1):
        if sl is not None:
            out[k] = [sl[0].start, sl[1].start, sl[2].start, sl[0].stop, sl[1].stop, sl[2].stop]
    return out


def write_outputs(out_dir, orig_img, anatomy_zyx, lesions_zyx, record):
    """anatomy.nii.gz (SynthSeg values, int16), lesions.nii.gz (instance ids, int32) in the input's own orientation and
    affine, and record.json; the lesion boxes of the record get `box_input` in the written file's axis order. The
    directory must not exist."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=False)
    written = {}
    for name, arr, dtype in (("anatomy.nii.gz", anatomy_zyx, np.int16), ("lesions.nii.gz", lesions_zyx, np.int32)):
        back = to_original(np.asarray(arr).transpose(2, 1, 0), orig_img.affine).astype(dtype)
        nib.save(nib.Nifti1Image(np.ascontiguousarray(back), orig_img.affine), str(out_dir / name))
        written[name] = back
    boxes = input_boxes(written["lesions.nii.gz"])
    for l in record.get("lesions", []):
        l["box_input"] = boxes.get(l["instance"])
    record["box_frame"] = {"box": "closest-canonical RAS volume, (z, y, x) half-open voxel indices",
                           "box_input": "lesions.nii.gz array axes (the input file's own order), half-open voxel indices"}
    (out_dir / "record.json").write_text(json.dumps(record, indent=1, ensure_ascii=False, default=str))
    return out_dir

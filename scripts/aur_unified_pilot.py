"""Bounded, private, single-process TRAIN-only compatibility pilot; never a formal training launcher."""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import sys
import subprocess
import time
import traceback

import numpy as np
import torch

from anatobind.aur.crops import augment, local_coordinates
from anatobind.aur.labels import SEQ_INDEX
from anatobind.aur.unified_data import collate_unified, load_unified_volume, make_unified_crop

DATA_ROOT = Path("/data2/congcong/data/FM_data")
PRIVATE_ROOT = Path.home()
TINY = dict(embed=32, depths=(1, 1, 1, 1), heads=(1, 1, 1, 1), window=(2, 4, 4),
            d_model=16, n_events=8, mask_dim=4, pixel_dim=8, dec_layers=1,
            dec_heads=2, rel_layers=1, rel_heads=2, use_checkpoint=False, dual_input=True)
CROPS = {"2d": (1, 32, 32), "3d": (8, 32, 32)}
PREFERRED = {"bmsr": "T1c", "isles": "DWI", "pdgm": "FLAIR"}


class PrivateParser(argparse.ArgumentParser):
    def error(self, message):
        allowed = {"output_exists", "invalid_pilot_budget", "invalid_pilot_deadline"}
        super().error(message if message in allowed else "invalid_arguments")


def validate_private_output(path, synthetic=False):
    """Refuse medical output outside home or inside a Git repository before opening input files."""
    path = Path(path).expanduser().resolve()
    if not synthetic and not path.is_relative_to(PRIVATE_ROOT.resolve()):
        raise ValueError("private_output_unsupported")
    repo = Path(__file__).resolve().parents[1]
    ancestor = path.parent
    while not ancestor.exists() and ancestor != ancestor.parent:
        ancestor = ancestor.parent
    check = subprocess.run(["git", "-C", str(ancestor), "rev-parse", "--show-toplevel"],
                           capture_output=True, text=True)
    if path.is_relative_to(repo) or check.returncode == 0:
        raise ValueError("private_output_unsupported")
    return path


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, payload):
    with Path(path).open("x") as stream:
        os.chmod(path, 0o600)
        json.dump(payload, stream, indent=2, allow_nan=False)


def select_training_rows(rows, validation, limit=3):
    """Select at most one sequence per TRAIN patient, after checking the whole patient split."""
    if not isinstance(limit, int) or not 1 <= limit <= 6:
        raise ValueError("invalid_patient_limit")
    historical = {(r["source"], str(r["patient"])) for r in rows if r["split"] == "test"}
    validation_ids = {(source, str(patient)) for source, patients in validation.items() for patient in patients}
    train = [r for r in rows if r["split"] == "train" and (r["source"], str(r["patient"])) not in validation_ids]
    if historical & validation_ids or any((r["source"], str(r["patient"])) in historical for r in train):
        raise ValueError("patient_split_overlap")
    candidates = [r for r in train if r["source"] in PREFERRED and r["sequence"] == PREFERRED[r["source"]]
                  and r["u_supervised"] and r.get("a_supervised", True) and r.get("r_supervised", True)]
    # Round-robin sources prevents an alphabetical prefix from selecting only a single disease source.
    pools = {source: sorted((r for r in candidates if r["source"] == source),
                            key=lambda r: (str(r["patient"]), r["sequence"])) for source in PREFERRED}
    selected, used = [], set()
    while len(selected) < limit:
        added = False
        for source, pool in pools.items():
            while pool and (source, str(pool[0]["patient"])) in used:
                pool.pop(0)
            if pool and len(selected) < limit:
                selected.append(pool.pop(0))
                used.add((source, str(selected[-1]["patient"])))
                added = True
        if not added:
            break
    if not selected:
        raise ValueError("no_eligible_training_rows")
    return selected


class SyntheticViews:
    """A small asymmetric observation with explicit, trustworthy synthetic supervision."""
    def __init__(self, seed=0):
        self.seed = int(seed)

    def __call__(self, mode, draw):
        shape = CROPS[mode]
        generator = torch.Generator().manual_seed(self.seed + draw * 2 + (mode == "3d"))
        image = torch.rand((1, 1, *shape), generator=generator) * 1.8 - .9
        valid = torch.ones((1, *shape))
        axes = [torch.arange(n, dtype=torch.float32) for n in shape]
        coords = torch.stack(torch.meshgrid(*axes, indexing="ij"))[None]
        window = tuple((0, n) for n in shape)
        local = torch.from_numpy(local_coordinates(window, shape))[None]
        entity = torch.ones((1, *shape), dtype=torch.long)
        entity[..., 16:] = 2
        instance = torch.zeros_like(entity)
        instance[:, :1 if mode == "2d" else 2, 8:12, 8:12] = 1
        present = torch.zeros((1, 32), dtype=torch.bool)
        present[:, :2] = True
        host_probs = torch.zeros((1, 14))
        host_probs[0, 0] = 1
        return dict(image=image, view1=image.clone(), view2=image.clone(), valid=valid, coords=coords, local=local,
                    entity=entity, entity_present=present, instance=instance, point_weight=valid.clone(),
                    a_ignore=instance > 0, seq=torch.tensor([SEQ_INDEX["FLAIR"]]), a_supervised=torch.tensor([True]),
                    u_supervised=torch.tensor([True]), r_supervised=torch.tensor([True]), n_instances=[1],
                    host=[torch.tensor([0])], host_probs=[host_probs], negatives=[torch.tensor([[1, -1]])],
                    r_eligible=[torch.tensor([True])], case=["synthetic"], input_mode=mode,
                    spacing=torch.ones((1, 3)), patient=torch.tensor([0]))


class TrainingViews:
    """Volume-balanced derived planes; deterministic draws permit exact runner resume."""
    def __init__(self, rows, seed=0, fixed_crops=False):
        self.rows, self.seed, self.fixed = list(rows), int(seed), bool(fixed_crops)
        self.cache = {}

    def __call__(self, mode, draw):
        index = draw % len(self.rows)
        row = self.rows[index]
        if index not in self.cache:
            self.cache[index] = load_unified_volume(row)
        ordinal = index if self.fixed else draw
        rng = np.random.default_rng([self.seed, int(mode == "3d"), ordinal])
        crop = make_unified_crop(self.cache[index], rng, mode, CROPS[mode],
                                 do_augment=not self.fixed, lesion_centred=True, n_events=TINY["n_events"])
        batch = collate_unified([[crop]])
        batch["view1"] = batch["image"].clone()
        second = augment(batch["image"][0, 0].numpy(), rng) if not self.fixed else batch["image"][0, 0].numpy()
        batch["view2"] = torch.from_numpy(np.ascontiguousarray(second))[None, None]
        identity = f"{row['source']}:{row['patient']}".encode()
        key = int.from_bytes(hashlib.sha256(identity).digest()[:8], "big") & ((1 << 63) - 1)
        batch["patient"] = torch.tensor([key])
        return batch


def runtime_digest():
    root = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    files = sorted({Path(module.__file__).resolve() for name, module in tuple(sys.modules.items())
                    if name.startswith("anatobind.aur") and getattr(module, "__file__", "").endswith(".py")})
    files.append(Path(__file__).resolve())
    for path in files:
        digest.update(str(path.relative_to(root)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def run_pipeline(args):
    from anatobind.aur.unified_train import UnifiedPilot

    out = validate_private_output(args.out, synthetic=args.synthetic)
    args.out = out
    out.mkdir(mode=0o700, parents=True, exist_ok=False)
    args._created_private_output = True
    os.chmod(out, 0o700)
    if args.synthetic:
        views, selected = SyntheticViews(args.seed), []
        contract = {"data_type": "synthetic", "rules_version": 1, "runtime_sha256": runtime_digest()}
    else:
        for path in (args.samples, args.val_patients):
            if not path.resolve().is_relative_to(DATA_ROOT.resolve()):
                raise ValueError("data_root_unsupported")
        rows, validation = json.loads(args.samples.read_text()), json.loads(args.val_patients.read_text())
        selected = select_training_rows(rows, validation, args.max_patients)
        # Check all referenced medical files before loading any arrays.
        medical_inputs = []
        for row_index, row in enumerate(selected):
            for field in ("image", "anatomy", "lesion"):
                if row.get(field) and not Path(row[field]).resolve().is_relative_to(DATA_ROOT.resolve()):
                    raise ValueError("data_root_unsupported")
                if row.get(field):
                    medical_inputs.append({"selection_index": row_index, "field": field,
                                           "sha256": file_hash(row[field])})
        write_json(out / "selection.json", selected)
        write_json(out / "medical_input_hashes.json", medical_inputs)
        selection_hash = file_hash(out / "selection.json")
        contract = {"data_type": "derived_train_planes", "rules_version": 1,
                    "samples_sha256": file_hash(args.samples), "split_sha256": file_hash(args.val_patients),
                    "selection_sha256": selection_hash, "fixed_crops": args.fixed_crops,
                    "medical_inputs_sha256": file_hash(out / "medical_input_hashes.json"),
                    "runtime_sha256": runtime_digest(), "crops": CROPS}
        views = TrainingViews(selected, args.seed, args.fixed_crops)
    config = dict(model=TINY, seed=args.seed, steps=args.steps_per_stage, lr=5e-4, weight_decay=.01,
                  points=256, u_mask_profile="unmatched_empty_v1", mode_cycle=["2d", "3d"], pilot=True,
                  ssl={"contrast_weight": 0., "mask_ratio": .4, "block_mm": (4., 8.)})
    write_json(out / "run_config.json", {"config": config, "data_contract": contract,
                                         "max_seconds": args.max_seconds, "device_type": torch.device(args.device).type})
    summary = dict(pilot_only=True, evidence="NOT_EVIDENCE", formal_training_ready=False,
                   historical_test_used=False, native_dicom_validated=False, model_size="tiny",
                   data_contract=contract, selected_rows=len(selected),
                   selected_source_counts=dict(Counter(row["source"] for row in selected)), stages={})
    start, initializer = time.monotonic(), None
    if torch.device(args.device).type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    for stage in ("I", "II", "III"):
        runner = UnifiedPilot(stage, config, contract, device=args.device, initializer=initializer)
        steps = []
        for _ in range(args.steps_per_stage):
            if time.monotonic() - start > args.max_seconds:
                runner.save(out / f"stage_{stage}_interrupted.pt")
                raise TimeoutError("pilot_deadline")
            before = time.monotonic()
            logged = runner.step(views)
            steps.append({**logged, "observed_wall_seconds": time.monotonic() - before})
        runner.save(out / f"stage_{stage}.pt")
        initializer = runner.export()
        with (out / f"stage_{stage}_export.pt").open("xb") as stream:
            os.chmod(stream.name, 0o600)
            torch.save(initializer, stream)
        summary["stages"][stage] = {"steps": steps, "exposures": steps[-1]["exposures"],
                                     "checkpoint_sha256": file_hash(out / f"stage_{stage}.pt")}
        del runner
    summary["wall_seconds"] = time.monotonic() - start
    if not args.synthetic:
        unchanged = all(file_hash(selected[item["selection_index"]][item["field"]]) == item["sha256"]
                        for item in medical_inputs)
        if not unchanged:
            raise ValueError("medical_inputs_changed_during_pilot")
        if (file_hash(args.samples) != contract["samples_sha256"]
                or file_hash(args.val_patients) != contract["split_sha256"]):
            raise ValueError("manifests_changed_during_pilot")
        summary["source_inputs_unchanged"] = True
    if torch.device(args.device).type == "cuda":
        summary["peak_cuda_allocated_bytes"] = torch.cuda.max_memory_allocated()
        summary["peak_cuda_reserved_bytes"] = torch.cuda.max_memory_reserved()
    write_json(out / "summary.json", summary)
    print("pilot_complete NOT_EVIDENCE")
    return summary


def main():
    parser = PrivateParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--samples", type=Path, default=DATA_ROOT / "derived/aur/samples_1mm_v1.json")
    parser.add_argument("--val-patients", type=Path, default=DATA_ROOT / "derived/aur/ssl_manifest_v1/val_patients.json")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--steps-per-stage", type=int, default=8)
    parser.add_argument("--max-patients", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-seconds", type=float, default=300.)
    parser.add_argument("--fixed-crops", action="store_true")
    args = parser.parse_args()
    args._created_private_output = False
    if not 2 <= args.steps_per_stage <= 64 or args.steps_per_stage % 2 or not 1 <= args.max_patients <= 6:
        parser.error("invalid_pilot_budget")
    if not np.isfinite(args.max_seconds) or not 0 < args.max_seconds <= 1800:
        parser.error("invalid_pilot_deadline")
    torch.set_num_threads(2)
    os.umask(0o077)
    try:
        args.out = args.out.expanduser().resolve()
        if args.out.exists():
            parser.error("output_exists")
        run_pipeline(args)
    except Exception:
        failure = traceback.format_exc()
        if args._created_private_output:
            try:
                with (args.out / "failure.txt").open("x") as stream:
                    os.chmod(stream.name, 0o600)
                    stream.write(failure)
            except Exception:
                pass  # Failure reporting must not expose an untrusted path through a second traceback.
        print("pilot_failed", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

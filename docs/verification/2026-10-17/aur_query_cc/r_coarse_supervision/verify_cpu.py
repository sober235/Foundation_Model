#!/usr/bin/env python3
"""Verify saved coarse-supervision assertions without image I/O or model execution.

Run with the nvgen interpreter, CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE=1,
PYTHONNOUSERSITE=1, OMP_NUM_THREADS=1 and MKL_NUM_THREADS=1. Only checkpoint
weights, aggregate run configuration and repository source are read. Output
contains no patient/sample identifiers or machine hostname.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


HISTORICAL = "77fd6ebc81ec945f77f7c9316f05119294b1aca7"
SOURCES = [
    "anatobind/aur/model.py", "anatobind/aur/heads.py",
    "anatobind/aur/train.py", "anatobind/aur/losses.py",
    "anatobind/aur/relation.py", "anatobind/aur/ssl/model.py",
    "anatobind/aur/ssl/checkpoint.py", "scripts/stage23_chain.sh",
]
EXPECTED_SHA = {
    "II": "c78413dcaac18de3d5b09e30c868ced346b3207c95097620dce35d76affd733d",
    "III": "38f6376333041a75be1cce83e927bc221468761f80ae952e2a3020805f7dcbde",
}
RUN_NAMES = {"II": "c2_stage2_240k_20261014_1344", "III": "c2_stage3_80k_20261014_1344"}
EXPORT_NAMES = {"II": "aur_stage2_best.pt", "III": "aur_stage3_best.pt"}
EXPECTED_GROUPS = {
    "II": {"backbone": 12929792, "heads": 7660040},
    "III": {"backbone": 12929856, "heads": 7742280, "relation": 2047489},
}


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git(repo, *args):
    return subprocess.run(["git", "-C", str(repo), *args], check=True,
                          capture_output=True, text=True).stdout.strip()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path("/home/congcongliu/src/Foundation_Model-aur-diagnostic"))
    parser.add_argument("--runs", type=Path, default=Path("/data2/congcong/data/FM_data/derived/aur/ssl_runs"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "Explicitly hide all GPUs")
    require(os.environ.get("PYTHONDONTWRITEBYTECODE") == "1", "Disable bytecode writes")
    require(os.environ.get("PYTHONNOUSERSITE") == "1", "Disable user-site packages")
    require(os.environ.get("OMP_NUM_THREADS") == "1", "Bound OMP threads to one")
    require(os.environ.get("MKL_NUM_THREADS") == "1", "Bound MKL threads to one")
    sys.path.insert(0, str(args.repo))
    import torch
    from anatobind.aur.model import AnatoBindBrain
    from anatobind.aur.train import param_groups
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)

    # Compare committed source with the training revision and also reject any
    # uncommitted change to those files. Neither check modifies the repository.
    git(args.repo, "diff", "--exit-code", HISTORICAL, "HEAD", "--", *SOURCES)
    git(args.repo, "diff", "--exit-code", "HEAD", "--", *SOURCES)
    report = {
        "verification": "CPU weight/config/source inspection only; no forward/backward or volume I/O",
        "all_assertions_passed": False,
        "environment": {"cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
                        "torch_version": torch.__version__, "torch_threads": torch.get_num_threads(),
                        "torch_interop_threads": torch.get_num_interop_threads()},
        "source": {"historical_code_sha": HISTORICAL,
                   "current_code_sha": git(args.repo, "rev-parse", "HEAD"),
                   "historical_source_equal": True, "relevant_worktree_clean": True,
                   "files": SOURCES},
        "stages": {}, "tensor_comparisons": {},
    }
    checkpoints = {}
    for stage in ("II", "III"):
        folder = args.runs / RUN_NAMES[stage]
        path = folder / EXPORT_NAMES[stage]
        digest = sha256(path)
        require(digest == EXPECTED_SHA[stage], f"{stage} checkpoint checksum differs")
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        require(all(value.device.type == "cpu" for value in checkpoint["model_state_dict"].values()),
                f"{stage} tensors must be on CPU")
        meta = checkpoint["meta"]
        run = json.loads((folder / "run_config.json").read_text())
        summary = json.loads((folder / "summary.json").read_text())
        cfg = meta["config"]
        # JSON sidecars serialize the checkpoint's crop tuple as a list.
        # Compare configuration values after that lossless JSON normalization.
        require(json.loads(json.dumps(cfg)) == run["config"],
                f"{stage} embedded and run configurations differ after JSON normalization")
        require(meta["stage"] == stage and meta["code_sha"] == HISTORICAL,
                f"{stage} unexpected checkpoint lineage")
        require(cfg["seed"] == 0 and cfg["u_mask_profile"] == "unmatched_empty_v1",
                f"{stage} unexpected seed/profile")
        expected_selection = (10000, 240000) if stage == "II" else (3000, 72000)
        require((meta["step"], meta["seen_crops"]) == expected_selection,
                f"{stage} selected checkpoint step differs")
        expected_completion = (10000, 240000) if stage == "II" else (3334, 80016)
        require((summary["steps"], summary["seen_crops"]) == expected_completion,
                f"{stage} completed run exposure differs")
        expected_lrs = ({"backbone": 6.124e-5, "heads": 6.124e-4} if stage == "II"
                        else {"backbone": 6.124e-5, "heads": 3.062e-4, "relation": 6.124e-4})
        require(cfg["lrs"] == expected_lrs, f"{stage} saved learning rates differ")
        require((cfg["warmup_steps"], cfg["backbone_warm_steps"], cfg["backbone_warm_scale"])
                == ((667, 667, 0.1) if stage == "II" else (133, 0, 1.0)),
                f"{stage} saved warmup differs")
        require(run["world"] == 2 and run["global_batch"] == 24 and run["params"] == 22719625,
                f"{stage} saved batch or model size differs")
        report["stages"][stage] = {
            "checkpoint_sha256": digest, "checkpoint_sha256_matches_frozen": True,
            "embedded_config_equals_run_config_after_json_normalization": True,
            "code_sha": meta["code_sha"],
            "stage": meta["stage"], "selected_step": meta["step"],
            "selected_seen_crops": meta["seen_crops"],
            "completed_steps": summary["steps"], "completed_seen_crops": summary["seen_crops"],
            "seed": cfg["seed"], "u_mask_profile": cfg["u_mask_profile"],
            "global_batch": run["global_batch"], "world": run["world"],
            "learning_rates": cfg["lrs"], "warmup_steps": cfg["warmup_steps"],
            "backbone_warm_steps": cfg["backbone_warm_steps"],
            "backbone_warm_scale": cfg["backbone_warm_scale"],
            "lambda_r": cfg["lambda_r"], "lambda_h": cfg["lambda_h"],
            "weight_decay": cfg["weight_decay"], "reported_params": run["params"],
            "reported_trainable_params": run["trainable_params"],
        }
        checkpoints[stage] = checkpoint

    # AnatoBindBrain is constructed on CPU before backbone checkpoint loading in
    # the trainer. Reproducing that seed and construction tests the random head
    # initialization exactly. Instantiation is not a forward pass.
    torch.manual_seed(0)
    model = AnatoBindBrain(use_checkpoint=False)
    initial = model.state_dict()
    second = checkpoints["II"]["model_state_dict"]
    third = checkpoints["III"]["model_state_dict"]
    for prefix in ("masks.embed_coarse.", "relation.", "masks.embed_full.",
                   "masks.lateral.", "masks.smooth."):
        keys = sorted(key for key in second if key.startswith(prefix))
        require(bool(keys), f"Missing tensor prefix {prefix}")
        equal_init = all(torch.equal(second[key], initial[key]) for key in keys)
        equal_later = all(torch.equal(third[key], second[key]) for key in keys)
        delta = max(float((third[key] - second[key]).abs().max()) for key in keys)
        require(equal_init == (prefix in ("masks.embed_coarse.", "relation.")),
                f"{prefix} Stage II initialization comparison differs")
        require(not equal_later and delta > 0, f"{prefix} did not change in Stage III")
        report["tensor_comparisons"][prefix] = {
            "tensor_count": len(keys), "stage2_bitwise_equal_seed0_init": equal_init,
            "stage3_bitwise_equal_stage2": equal_later, "stage3_vs_stage2_max_abs_delta": delta,
        }
    for stage in ("II", "III"):
        cfg = checkpoints[stage]["meta"]["config"]
        groups = param_groups(model, stage, cfg["lrs"])
        counts = {group["name"]: sum(p.numel() for p in group["params"]) for group in groups}
        require(counts == EXPECTED_GROUPS[stage], f"{stage} parameter groups differ")
        require(sum(counts.values()) == report["stages"][stage]["reported_trainable_params"],
                f"{stage} trainable count differs from run configuration")
        report["stages"][stage]["regenerated_parameter_group_counts"] = counts
    first_cfg = json.loads((args.runs / "stage1_r2_320k_20261010_1303/run_config.json").read_text())["config"]
    require(first_cfg["contrast_weight"] == 0.0, "Actual Stage I contrast weight differs")
    report["stage1_actual_contrast_weight"] = first_cfg["contrast_weight"]
    report["all_assertions_passed"] = True
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({"all_assertions_passed": True,
                      "stage3_checkpoint_sha256": report["stages"]["III"]["checkpoint_sha256"],
                      "stage2_coarse_bitwise_seed0_equal": report["tensor_comparisons"]["masks.embed_coarse."]["stage2_bitwise_equal_seed0_init"],
                      "stage3_coarse_max_abs_delta": report["tensor_comparisons"]["masks.embed_coarse."]["stage3_vs_stage2_max_abs_delta"]}, sort_keys=True))


if __name__ == "__main__":
    main()

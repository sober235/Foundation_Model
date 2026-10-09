# tests/test_aur_ssl_train.py
import importlib.util
import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest
import torch

import anatobind.aur.ssl.checkpoint as CK
import anatobind.aur.ssl.train as T
from anatobind.aur.swin import SwinBackbone

SMALL = {"embed": 32, "depths": (1, 1, 1, 1), "heads": (1, 2, 4, 8)}


def _load(name):
    path = Path(__file__).resolve().parents[1] / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _manifest(tmp_path, n_train=3, n_val=1):
    rows = []
    rng = np.random.default_rng(0)
    for i in range(n_train + n_val):
        img = np.zeros((24, 20, 12), np.float32)
        img[4:20, 4:16, 2:10] = rng.uniform(100, 300, (16, 12, 8))
        p = tmp_path / f"v{i}.nii.gz"
        nib.save(nib.Nifti1Image(img, np.eye(4)), str(p))
        rows.append({"case": f"c{i}", "source": "pdgm" if i % 2 else "hcp", "patient": f"p{i}", "sequence": "T1", "source_sequence": "T1",
                     "image": str(p), "anatomy": None, "lesion": None, "split": "train", "ssl_split": "train" if i < n_train else "val"})
    m = tmp_path / "samples_ssl.json"
    m.write_text(json.dumps(rows))
    return m


def _cfg(**over):
    cfg = {**SMALL, "seen_crops": 8, "microbatch": 2, "grad_accum": 1, "crops_per_volume": 2, "crop": (8, 16, 16), "workers": 0,
           "warmup_steps": 1, "val_every": 2, "val_volumes": 1, "save_every": 2, "log_every": 1, "block_mm": (8.0, 16.0)}
    cfg.update(over)
    return cfg


def test_lr_schedule_warms_up_then_decays():
    assert T.lr_lambda(0, 10, 100) == pytest.approx(0.1) and T.lr_lambda(9, 10, 100) == pytest.approx(1.0)
    assert T.lr_lambda(55, 10, 100) == pytest.approx(0.5, abs=0.01) and T.lr_lambda(100, 10, 100) == pytest.approx(0.0, abs=1e-6)


def test_a_tiny_run_writes_logs_validation_exports_and_resume_files(tmp_path):
    m = _manifest(tmp_path)
    out = tmp_path / "run"
    summary = T.run(_cfg(), m, out)
    assert summary["steps"] == 4 and summary["seen_crops"] == 8 and summary["global_batch"] == 2
    cfg = json.loads((out / "run_config.json").read_text())
    assert cfg["train_rows"] == 3 and cfg["val_rows"] == 1 and cfg["total_steps"] == 4 and cfg["manifest_sha256"]
    log = [json.loads(l) for l in (out / "log_rank0.jsonl").read_text().splitlines()]
    assert [r["step"] for r in log] == [1, 2, 3, 4] and all(np.isfinite(r["loss"]) for r in log)
    val = [json.loads(l) for l in (out / "val.jsonl").read_text().splitlines()]
    assert [v["step"] for v in val] == [2, 4] and all(np.isfinite(v["mim"]) and v["val_crops"] == 2 for v in val)
    best = json.loads((out / "best.json").read_text())
    assert best["step"] in (2, 4) and Path(best["path"]).is_file() and (out / "ssl_stage1_best.pt").is_file()
    assert (out / "resume_step2.pt").is_file() and (out / "resume_step4.pt").is_file()
    bb = SwinBackbone(**SMALL)
    meta = CK.load_backbone(out / "ssl_stage1_best.pt", bb)
    assert meta["stage"] == "I" and meta["seen_crops"] == best["step"] * 2
    with pytest.raises(FileExistsError):
        T.run(_cfg(), m, out)


def test_a_run_resumes_from_its_checkpoint(tmp_path):
    m = _manifest(tmp_path)
    out = tmp_path / "run"
    first = T.run(_cfg(seen_crops=8, max_steps=2), m, out)
    assert first["steps"] == 2 and (out / "resume_step2.pt").is_file()
    second = T.run(_cfg(seen_crops=8), m, out, resume=out / "resume_step2.pt")
    assert second["steps"] == 4 and second["seen_crops"] == 8
    log = [json.loads(l) for l in (out / "log_rank0.jsonl").read_text().splitlines()]
    assert [r["step"] for r in log] == [1, 2, 3, 4]


def test_probe_mode_times_steps_without_validation(tmp_path):
    m = _manifest(tmp_path)
    out = tmp_path / "probe"
    summary = T.run(_cfg(probe=True, max_steps=3, seen_crops=10 ** 6), m, out)
    assert summary["steps"] == 3 and summary["step_s_mean"] > 0 and (out / "probe.json").is_file() and (out / "probe_config.json").is_file()
    assert not (out / "val.jsonl").exists() and not any(out.glob("ssl_stage1_*.pt"))
    assert _load("aur_ssl_probe").main.__name__ == "main" and _load("aur_ssl_train").parser().parse_args(["--samples", "a", "--out", "b"]).seen_crops == 320000

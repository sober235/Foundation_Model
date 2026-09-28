# tests/test_nndet_train.py
import importlib.util
import json
import pickle
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]


def _load():
    spec = importlib.util.spec_from_file_location("nndet_train", REPO / "scripts/nndet_train.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_launcher_paths_match_the_env_file():
    mod = _load()
    text = (REPO / "scripts/nndet_env.sh").read_text()
    assert f"det_data={mod.DET_DATA}\n" in text and f"det_models={mod.DET_MODELS}\n" in text
    assert mod.train_dir(Path("/m"), 0) == Path("/m/Task903_FastMRIBrainSmallLesion/RetinaUNetV001_D3V001_3d/fold0")


def _prep(root, splits):
    prep = root / "Task903_FastMRIBrainSmallLesion" / "preprocessed"
    (prep / "D3V001_3d" / "imagesTr").mkdir(parents=True)
    (prep / "splits_final.pkl").write_bytes(pickle.dumps(splits, protocol=4))


def test_preflight_refusals(tmp_path):
    mod = _load()
    models, logs = tmp_path / "models", tmp_path / "logs"
    sj = [{"train": ["b"], "val": ["a"]}, {"train": ["a"], "val": ["b"]}]
    (tmp_path / "splits.json").write_text(json.dumps(sj))
    with pytest.raises(FileNotFoundError, match="nndet_prep"):
        mod.preflight(tmp_path / "empty", models, 0, tmp_path / "splits.json", logs)
    _prep(tmp_path / "auto", [{"train": ["a"], "val": ["b"]}])         # e.g. nnDetection's own KFold split
    with pytest.raises(ValueError, match="differs"):
        mod.preflight(tmp_path / "auto", models, 0, tmp_path / "splits.json", logs)
    data = tmp_path / "data"
    _prep(data, sj)
    assert mod.preflight(data, models, 0, tmp_path / "splits.json", logs) == logs / "fold0.log"
    mod.train_dir(models, 0).mkdir(parents=True)
    with pytest.raises(FileExistsError, match="overwrite"):
        mod.preflight(data, models, 0, tmp_path / "splits.json", logs)
    logs.mkdir()
    (logs / "fold1.log").write_text("")
    with pytest.raises(FileExistsError, match="fold1.log"):
        mod.preflight(data, models, 1, tmp_path / "splits.json", logs)


def test_command_pins_the_gpu_and_quotes_paths():
    cmd = _load().command(0, 3, Path("/tmp/a b/fold0.log"), Path("/r/scripts/nndet_env.sh"))
    assert cmd[:3] == ["setsid", "bash", "-c"]
    assert "source /r/scripts/nndet_env.sh && CUDA_VISIBLE_DEVICES=3 nice -n 19 nndet_train 903 -o exp.fold=0 --sweep" in cmd[3]
    assert cmd[3].endswith("> '/tmp/a b/fold0.log' 2>&1")

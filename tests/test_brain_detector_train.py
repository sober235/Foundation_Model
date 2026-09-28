import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location("bdt", Path(__file__).resolve().parents[1] / "scripts/brain_detector_train.py")
bdt = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bdt)


def test_idle_gpus_parse_nvidia_smi_and_skip_busy_cards():
    csv = "0, 14\n1, 30000\n2, 14\n3, 14\n4, 14\n"
    assert bdt.idle_gpus(csv, [0, 1, 2, 3], busy_pids={}) == [0, 2, 3]
    assert bdt.idle_gpus(csv, [0, 2], busy_pids={2: [123]}) == [0]


def test_assign_folds_round_robin_and_command():
    assert bdt.assign([0, 1, 2, 3, 4], [0, 1, 2, 3]) == [(0, 0), (1, 1), (2, 2), (3, 3), (4, 0)]
    assert bdt.train_command("2d", 3, "nnUNetTrainer_250epochs") == ["nnUNetv2_train", "903", "2d", "3", "-tr", "nnUNetTrainer_250epochs", "--npz"]

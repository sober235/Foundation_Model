import os
import subprocess
from pathlib import Path

ENV_SH = Path(__file__).resolve().parents[1] / "scripts/nndet_env.sh"


def _env_after_source(extra):
    env = {"PATH": "/usr/bin:/bin", "HOME": os.environ["HOME"], **extra}
    out = subprocess.run(["bash", "-c", f"source {ENV_SH} && env"], capture_output=True, text=True, check=True,
                         env=env).stdout
    return dict(line.split("=", 1) for line in out.splitlines() if "=" in line)


def test_nndet_env_sets_paths_threads_and_hides_nnunet():
    e = _env_after_source({"nnUNet_preprocessed": "/x", "nnUNet_raw": "/y", "nnUNet_results": "/z"})
    assert e["det_data"] == "/data2/congcong/data/FM_data/derived/nndet"
    assert e["det_models"] == "/data2/congcong/data/FM_data/derived/nndet_models"
    assert "." not in e["det_data"] and "." not in e["det_models"]
    assert e["OMP_NUM_THREADS"] == "1" and e["det_num_threads"] == "8" and e["PYTHONNOUSERSITE"] == "1"
    assert e["PATH"].split(":")[0] == os.path.expanduser("~/anaconda3/envs/nndet/bin")
    assert not any(k.startswith("nnUNet_") for k in e)

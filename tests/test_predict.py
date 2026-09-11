import numpy as np
import pytest

torch = pytest.importorskip("torch")

from anatobind.eval.predict import load_prediction, predict_volume, save_prediction
from anatobind.model.upstream import Upstream


def test_a_prediction_is_cropped_to_the_valid_depth_and_typed(tmp_path):
    torch.manual_seed(0)
    model = Upstream(K=6, M=4, num_classes=4, d_model=32, embed_dim=16, layers=2, heads=4, mask_dim=8)
    pred = predict_volume(model, torch.randn(1, 1, 32, 64, 64), valid_depth=30, device=torch.device("cpu"))
    lm = pred["label_map"]
    assert lm.shape == (30, 64, 64) and lm.dtype == np.uint8 and lm.max() <= 6
    assert pred["cls_prob"].shape == (4, 5) and np.allclose(pred["cls_prob"].sum(-1), 1.0, atol=1e-5)
    assert pred["boxes_vox"].shape == (4, 6) and (pred["boxes_vox"][:, 3:] >= pred["boxes_vox"][:, :3]).all()
    assert pred["a_embed"].dtype == np.float16 and pred["u_embed"].shape == (4, 32)
    save_prediction(tmp_path / "p.npz", pred)
    back = load_prediction(tmp_path / "p.npz")
    assert set(back) == set(pred) and np.array_equal(back["label_map"], lm)


class _Fixed(torch.nn.Module):
    """Every structure is off except structure 3 at one voxel; one lesion peak of class 1."""

    M = 3

    def forward(self, image):
        masks = torch.full((1, 6, 32, 8, 8), -10.0)
        masks[0, 2, 5, 1, 1] = 10.0
        heat = torch.full((1, 4, 16, 2, 2), -10.0)
        heat[0, 1, 4, 1, 0] = 10.0
        return {"masks": masks, "presence": torch.zeros(1, 6), "a_embed": torch.zeros(1, 6, 4), "heat": heat,
                "offset": torch.zeros(1, 3, 16, 2, 2), "size": torch.zeros(1, 3, 16, 2, 2),
                "feat": torch.zeros(1, 4, 16, 2, 2)}


def test_only_confident_voxels_get_a_label_and_the_peak_becomes_a_box():
    pred = predict_volume(_Fixed(), torch.zeros(1, 1, 32, 8, 8), valid_depth=32, device=torch.device("cpu"))
    assert pred["label_map"].sum() == 3 and pred["label_map"][5, 1, 1] == 3
    assert pred["cls_prob"].shape == (3, 5) and int(pred["cls_prob"][0].argmax()) == 1
    # peak cell (4, 1, 0), size one cell: centre (9, 6, 2) voxels, extent (2, 4, 4)
    assert np.allclose(pred["boxes_vox"][0], [8, 4, 0, 10, 8, 4])

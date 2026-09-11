import pytest

torch = pytest.importorskip("torch")

from anatobind.model.dense_head import CentreHead, centre_loss, centre_targets, decode_centres

GRID = (8, 8, 8)          # the F1 grid of a (16, 32, 32) input
BOX = torch.tensor([[4.0, 8, 12, 10, 20, 24]])   # voxels (z0, y0, x0, z1, y1, x1): centre cell (3, 3, 4), 3 cells a side


def test_the_head_predicts_on_the_f1_grid_with_a_low_prior():
    head = CentreHead(c1=16, num_classes=4, hidden=16, embed_dim=32)
    out = head(torch.randn(2, 16, *GRID))
    assert out["heat"].shape == (2, 4, *GRID) and out["offset"].shape == (2, 3, *GRID)
    assert out["size"].shape == (2, 3, *GRID) and out["feat"].shape == (2, 32, *GRID)
    assert abs(float(out["heat"].sigmoid().mean()) - 0.1) < 0.05


def test_targets_put_a_unit_peak_on_the_centre_cell_of_the_right_class():
    heat, cells, offsets, sizes = centre_targets(BOX, [2], GRID, 4, "cpu")
    assert heat.shape == (4, *GRID) and float(heat[2, 3, 3, 4]) == 1.0 and float(heat[2].max()) == 1.0
    assert float(heat[0].max()) == 0.0
    assert cells[0].tolist() == [3, 3, 4]
    torch.testing.assert_close(offsets[0], torch.zeros(3))
    torch.testing.assert_close(sizes[0], torch.log(torch.full((3,), 3.0)))


def _peaked(cell, cls, offset=(0.0, 0.0, 0.0), size_cells=3.0, C=4):
    out = {"heat": torch.full((1, C, *GRID), -10.0), "offset": torch.zeros(1, 3, *GRID),
           "size": torch.zeros(1, 3, *GRID), "feat": torch.arange(5 * 512, dtype=torch.float32).reshape(1, 5, *GRID)}
    z, y, x = cell
    out["heat"][0, cls, z, y, x] = 10.0
    out["offset"][0, :, z, y, x] = torch.tensor(offset)
    out["size"][0, :, z, y, x] = torch.log(torch.tensor(size_cells))
    return out


def test_decoding_recovers_the_box_class_and_embedding():
    out = _peaked((3, 3, 4), 1, offset=(0.25, 0.0, -0.25))
    boxes, cls_prob, embed = decode_centres(out, M=2, valid_depth=16)
    # centre = (cell + offset + 0.5) * stride = (7.5, 14, 17); size = 3 cells * stride = (6, 12, 12)
    torch.testing.assert_close(boxes[0], torch.tensor([4.5, 8.0, 11.0, 10.5, 20.0, 23.0]))
    assert int(cls_prob[0].argmax()) == 1 and int(cls_prob[1].argmax()) == 4
    torch.testing.assert_close(cls_prob.sum(-1), torch.ones(2))
    torch.testing.assert_close(embed[0], out["feat"][0, :, 3, 3, 4])


def test_peaks_in_the_depth_padding_are_ignored():
    out = _peaked((7, 0, 0), 0)                      # cell z = 7 covers voxels 14-15
    _, cls_prob, _ = decode_centres(out, M=1, valid_depth=12)
    assert int(cls_prob[0].argmax()) == 4            # the only peak sat in the padding: no detection


def test_the_loss_prefers_the_right_heatmap():
    heat_t, _, _, _ = centre_targets(BOX, [2], GRID, 4, "cpu")
    good = {"heat": torch.where(heat_t == 1, 8.0, -8.0)[None], "offset": torch.zeros(1, 3, *GRID),
            "size": torch.log(torch.full((1, 3, *GRID), 3.0))}
    bad = {**good, "heat": torch.zeros(1, 4, *GRID)}
    batch = {"boxes": [BOX], "box_classes": [torch.tensor([2])], "valid_depth": torch.tensor([16])}
    lg, lb = centre_loss(good, batch), centre_loss(bad, batch)
    assert float(lg["heat"]) < 0.1 * float(lb["heat"])
    assert float(lg["offset"]) == 0.0 and float(lg["size"]) == pytest.approx(0.0, abs=1e-6)

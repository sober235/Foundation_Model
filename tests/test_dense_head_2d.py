import pytest

torch = pytest.importorskip("torch")

from anatobind.model.dense_head_2d import CentreHead2D, centre_loss_2d, centre_targets_2d, decode_centres_2d

GRID = (16, 16)         # the F1 grid of a (32, 32) input, stride 2
BOX = torch.tensor([[8.0, 12, 20, 24]])   # pixels (y0, x0, y1, x1): centre cell (6, 8), 6 cells a side


def test_the_head_predicts_on_the_f1_grid_with_a_low_prior():
    head = CentreHead2D(c1=16, num_classes=5, hidden=16, embed_dim=32)
    out = head(torch.randn(2, 16, *GRID))
    assert out["heat"].shape == (2, 5, *GRID) and out["offset"].shape == (2, 2, *GRID)
    assert out["size"].shape == (2, 2, *GRID) and out["feat"].shape == (2, 32, *GRID)
    assert abs(float(out["heat"].sigmoid().mean()) - 0.1) < 0.05


def test_targets_put_a_unit_peak_on_the_centre_cell_of_the_right_class():
    heat, cells, offsets, sizes = centre_targets_2d(BOX, [2], GRID, 5, "cpu")
    assert heat.shape == (5, *GRID) and float(heat[2, 6, 8]) == 1.0 and float(heat[2].max()) == 1.0
    assert float(heat[0].max()) == 0.0
    assert cells[0].tolist() == [6, 8]
    torch.testing.assert_close(offsets[0], torch.full((2,), 0.5))
    torch.testing.assert_close(sizes[0], torch.log(torch.full((2,), 6.0)))


def _peaked(cell, cls, offset=(0.0, 0.0), size_cells=6.0, C=5):
    out = {"heat": torch.full((1, C, *GRID), -10.0), "offset": torch.zeros(1, 2, *GRID),
           "size": torch.zeros(1, 2, *GRID), "feat": torch.arange(5 * 256, dtype=torch.float32).reshape(1, 5, *GRID)}
    y, x = cell
    out["heat"][0, cls, y, x] = 10.0
    out["offset"][0, :, y, x] = torch.tensor(offset)
    out["size"][0, :, y, x] = torch.log(torch.tensor(size_cells))
    return out


def test_decoding_recovers_the_box_class_and_embedding():
    out = _peaked((6, 8), 1, offset=(0.25, -0.25))
    boxes, cls_prob, embed = decode_centres_2d(out, M=2)
    # centre = (cell + offset + 0.5) * stride = (13.5, 16.5); size = 6 cells * stride = (12, 12)
    torch.testing.assert_close(boxes[0], torch.tensor([7.5, 10.5, 19.5, 22.5]))
    assert int(cls_prob[0].argmax()) == 1 and int(cls_prob[1].argmax()) == 5
    torch.testing.assert_close(cls_prob.sum(-1), torch.ones(2))
    torch.testing.assert_close(embed[0], out["feat"][0, :, 6, 8])


def test_the_loss_prefers_the_right_heatmap():
    heat_t, _, _, _ = centre_targets_2d(BOX, [2], GRID, 5, "cpu")
    good = {"heat": torch.where(heat_t == 1, 8.0, -8.0)[None], "offset": torch.full((1, 2, *GRID), 0.5),
            "size": torch.log(torch.full((1, 2, *GRID), 6.0))}
    bad = {**good, "heat": torch.zeros(1, 5, *GRID)}
    batch = {"boxes": [BOX], "box_classes": [torch.tensor([2])]}
    lg, lb = centre_loss_2d(good, batch), centre_loss_2d(bad, batch)
    assert float(lg["heat"]) < 0.1 * float(lb["heat"])
    assert float(lg["offset"]) == 0.0 and float(lg["size"]) == pytest.approx(0.0, abs=1e-6)

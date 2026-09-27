import numpy as np
import pytest

torch = pytest.importorskip("torch")

from anatobind.relation.encoder import DX_INDEX, SIDE_LEFT, SIDE_RIGHT, TORCH_CONFIGS, LesionEncoder, augment, config_key, crop, flip_batch
from anatobind.relation.table import features_per_slot
from synth_relation import synthetic_table


def test_crop_sizes_and_channel_layout():
    t, p = synthetic_table(2, 2)
    for cfg in TORCH_CONFIGS:
        x = crop(p["image"], p["mask"], cfg["px"], cfg["slices"])
        assert x.shape == (4, 2 * cfg["slices"], cfg["px"], cfg["px"]) and x.dtype == np.float32
    x = crop(p["image"], p["mask"], 32, 1)
    assert np.array_equal(x[:, 0], p["image"][:, 1, 8:40, 8:40].astype(np.float32))       # the centre slice, centred crop
    assert x[:, 1].max() == 1.0 and set(np.unique(x[:, 1])) <= {0.0, 1.0}                  # mask channel
    assert config_key(TORCH_CONFIGS[0]) == "px32_s3"


def test_flip_is_an_involution_and_keeps_geometry_consistent():
    t, p = synthetic_table(2, 2)
    x, geo = crop(p["image"], p["mask"], 32, 3), features_per_slot(t)
    xf, gf = flip_batch(x, geo)
    assert np.array_equal(xf, x[:, :, ::-1, :]) and np.array_equal(gf[:, :, DX_INDEX], -geo[:, :, DX_INDEX])
    assert np.array_equal(gf[:, :, SIDE_LEFT], geo[:, :, SIDE_RIGHT]) and np.array_equal(gf[:, :, SIDE_RIGHT], geo[:, :, SIDE_LEFT])
    xff, gff = flip_batch(xf, gf)
    assert np.array_equal(xff, x) and np.array_equal(gff, geo)
    others = [i for i in range(geo.shape[2]) if i not in (DX_INDEX, SIDE_LEFT, SIDE_RIGHT)]
    assert np.array_equal(gf[:, :, others], geo[:, :, others])


def test_augment_scales_only_image_channels_and_flips_per_sample():
    t, p = synthetic_table(4, 4)
    x, geo = crop(p["image"], p["mask"], 32, 3), features_per_slot(t)
    xa, ga = augment(x, geo, np.random.default_rng(0), flip_p=1.0, intensity=0.1)
    assert np.array_equal(xa[:, 1::2], x[:, 1::2, ::-1, :])                                  # masks flipped, never scaled
    ratio = xa[:, 0::2] / np.where(x[:, 0::2, ::-1, :] == 0, 1, x[:, 0::2, ::-1, :])
    assert (np.abs(ratio - 1) <= 0.1 + 1e-6).all() and np.array_equal(ga[:, :, DX_INDEX], -geo[:, :, DX_INDEX])
    xb, gb = augment(x, geo, np.random.default_rng(0), flip_p=0.0, intensity=0.0)
    assert np.array_equal(xb, x) and np.array_equal(gb, geo)


def test_encoder_output_is_128_dimensional_for_every_config():
    for cfg in TORCH_CONFIGS:
        enc = LesionEncoder(2 * cfg["slices"])
        out = enc(torch.zeros(3, 2 * cfg["slices"], cfg["px"], cfg["px"]))
        assert out.shape == (3, 128)

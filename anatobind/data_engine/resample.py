"""Exact 2x in-plane reductions used to bring SKM-TEA from 0.3125 mm to 0.625 mm."""
import math

import numpy as np


def _blocks(vol):
    x, y, z = vol.shape
    return vol[: x - x % 2, : y - y % 2].reshape(x // 2, 2, y // 2, 2, z)


def downsample2_inplane_image(vol):
    return _blocks(np.asarray(vol, dtype=np.float32)).mean(axis=(1, 3)).astype(np.float32)


def downsample2_inplane_labels(lab):
    lab = np.asarray(lab)
    b = _blocks(lab).transpose(0, 2, 4, 1, 3).reshape(lab.shape[0] // 2, lab.shape[1] // 2, lab.shape[2], 4)
    n_labels = int(lab.max()) + 1
    counts = np.stack([(b == l).sum(-1) for l in range(n_labels)], axis=-1)  # (X/2, Y/2, Z, L)
    return np.argmax(counts, axis=-1).astype(np.uint8)  # argmax returns the smallest index on ties


def scale_box_inplane(box, factor=0.5):
    x0, y0, z0, x1, y1, z1 = box
    return (math.floor(x0 * factor), math.floor(y0 * factor), z0, math.ceil(x1 * factor), math.ceil(y1 * factor), z1)

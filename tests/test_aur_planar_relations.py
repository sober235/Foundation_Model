"""Geometry pseudo-targets are observation-local weak training labels, NOT_EVIDENCE."""
import importlib

import numpy as np
import pytest

from anatobind.aur.labels import ENTITY_LABELS, ENTITY_HOST
from anatobind.aur.targets import host_targets


def qualifier():
    try:
        module = importlib.import_module("anatobind.aur.planar_relations")
    except ModuleNotFoundError:
        pytest.fail("observation-level planar weak relation qualification is missing")
    return module.qualify_planar_weak_relations


def inputs():
    source = np.zeros((3, 8, 8), np.int32)
    source[:, 2:5, 2:5] = 1
    hosts = np.zeros(source.shape, np.uint8)
    hosts[:, 1:6, 1:6] = 1  # Map 1 means R host_index 0, a valid host.
    plane = {"instance": source[1:2].copy(), "host_map": hosts[1:2].copy()}
    return source, hosts, plane, np.ones((1, 8, 8), bool), source[1:2] > 0, [1], 1.0


def qualify(args=None, **kwargs):
    return qualifier()(*(inputs() if args is None else args), voxel_volume_mm3=1.0, **kwargs)


def test_host_zero_is_positive_and_overlap_is_independent_of_a_ignore():
    out = qualify()
    assert out["r_eligible"].tolist() == [True]
    assert out["host"].tolist() == [0]
    assert out["probs"].tolist() == [[1.0] + [0.0] * 13]
    assert out["provenance"] == ["geometry_weak2d_v1"]
    assert out["weight"].tolist() == [1.0]
    assert out["evidence"] == "NOT_EVIDENCE"


def test_same_legacy_one_hot_does_not_prove_source_overlap():
    args = list(inputs())
    positive = args[1].copy()
    nearest = positive.copy()
    nearest[args[0] > 0] = 0
    # Both maps yield the same old target because the nearby host is within NEAR_MM.
    label = next(label for label, host in zip(ENTITY_LABELS, ENTITY_HOST) if host == 0)
    a = host_targets(args[0], np.where(positive, label, 0), (1, 1, 1))
    b = host_targets(args[0], np.where(nearest, label, 0), (1, 1, 1))
    assert np.array_equal(a["probs"], b["probs"])
    args[1] = nearest
    out = qualify(args)
    assert out["r_eligible"].tolist() == [False]
    assert out["reasons"] == ["source_no_overlap"]
    assert out["host"].tolist() == [-1]  # Unknown eligibility does not invent NO_HOST.


@pytest.mark.parametrize("change, reason", [
    ("source_tie", "source_tied_host"),
    ("plane_tie", "plane_tied_host"),
    ("plane_no_overlap", "plane_no_overlap"),
    ("different_host", "plane_source_host_disagree"),
    ("invisible_host", "host_not_visible"),
    ("invalid_fov", "component_outside_valid"),
    ("thin_sliver", "plane_below_area"),
    ("small_source", "source_below_volume"),
    ("split_source", "source_not_single_component"),
])
def test_unreliable_source_or_transformed_observation_is_rejected(change, reason):
    args = list(inputs())
    source, hosts, plane, valid, ignore = args[:5]
    if change == "source_tie":
        hosts[0, 2:5, 2:5] = 2
        hosts[1, 2:5, 2:5] = 0  # Nine voxels per host.
    elif change == "plane_tie":
        plane["host_map"][0, 2:5, 2:5] = [[1, 1, 1], [1, 0, 2], [2, 2, 2]]
    elif change == "plane_no_overlap":
        plane["host_map"][plane["instance"] > 0] = 0
    elif change == "different_host":
        plane["host_map"][:] = 2
    elif change == "invisible_host":
        ignore[:] = True
    elif change == "invalid_fov":
        valid[0, 2, 2] = False
    elif change == "thin_sliver":
        args[6] = 0.1
    elif change == "small_source":
        source[:] = 0
        source[1, 2:5, 2:5] = 1
    elif change == "split_source":
        source[0, 7, 7] = 1
    out = qualify(args)
    assert out["r_eligible"].tolist() == [False]
    assert out["reasons"] == [reason]


@pytest.mark.parametrize("gate,reason", [("a_supervised", "anatomy_unqualified"),
                                         ("u_supervised", "lesion_unqualified"),
                                         ("r_supervised", "relation_unqualified")])
def test_unknown_or_disabled_supervision_cannot_enable_weak_relations(gate, reason):
    out = qualify(**{gate: False})
    assert out["r_eligible"].tolist() == [False] and out["reasons"] == [reason]


def test_only_train_can_receive_geometry_weak_targets():
    out = qualify(split="val")
    assert out["r_eligible"].tolist() == [False]
    assert out["reasons"] == ["not_train"]


def test_plane_negatives_do_not_inherit_a_second_host_outside_the_plane():
    args = list(inputs())
    args[1][0, 2, 2] = 2  # Secondary source overlap, absent from this observation.
    out = qualify(args)
    assert out["r_eligible"].tolist() == [True]
    assert out["negatives"][0, 1] == -1
    args[2]["host_map"][0, 2, 2] = 3
    assert qualify(args)["negatives"][0, 1] == 2


def test_multiple_observation_components_keep_one_source_identity():
    args = list(inputs())
    args[2]["instance"][:] = 0
    args[2]["instance"][0, 2, 2:4] = 1
    args[2]["instance"][0, 4, 2:4] = 2
    args[5] = [1, 1]
    out = qualify(args)
    assert out["r_eligible"].tolist() == [True, True]
    assert out["host"].tolist() == [0, 0]


@pytest.mark.parametrize("change", ["wrong_grid", "probability_host_map", "bad_mapping", "disconnected_plane"])
def test_malformed_geometry_is_refused(change):
    args = list(inputs())
    if change == "wrong_grid":
        args[1] = args[1][:2]
    elif change == "probability_host_map":
        args[1] = args[1].astype(float) + 0.2
    elif change == "bad_mapping":
        args[5] = [2]
    else:
        args[2]["instance"][0, 7, 7] = 1
    with pytest.raises(ValueError):
        qualify(args)

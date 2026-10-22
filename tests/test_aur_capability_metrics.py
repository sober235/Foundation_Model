"""Hand-derived TRAIN capability metrics and frozen-readout regression cases."""
import copy
import importlib
import math

import pytest
import torch


@pytest.fixture
def metrics():
    try:
        return importlib.import_module("scripts.aur_capability_metrics")
    except ModuleNotFoundError as exc:
        if exc.name != "scripts.aur_capability_metrics":
            raise
        class MissingMetrics:
            def __getattr__(self, name):
                pytest.fail("capability metrics are not implemented")
        return MissingMetrics()


def batch(shape=(1, 2, 4), mode="2d"):
    size = math.prod(shape)
    entity = torch.zeros(1, *shape, dtype=torch.long)
    instance = torch.zeros_like(entity)
    entity.flatten()[:8] = torch.tensor([1, 1, 2, 2, 0, 3, 32, 1])
    instance.flatten()[:4] = torch.tensor([1, 1, 2, 2])
    valid = torch.ones(1, *shape)
    valid.flatten()[7] = 0
    ignored = torch.zeros_like(entity, dtype=torch.bool)
    ignored.flatten()[5] = True
    return {"image": torch.zeros(1, 1, *shape), "coords": torch.zeros(1, 3, *shape),
            "local": torch.zeros(1, 3, *shape), "valid": valid, "entity": entity,
            "instance": instance, "a_ignore": ignored, "point_weight": valid.clone(),
            "a_supervised": torch.tensor([True]), "u_supervised": torch.tensor([True]),
            "r_supervised": torch.tensor([True]), "n_instances": [2],
            "host": [torch.tensor([0, 13])], "r_eligible": [torch.tensor([True, True])],
            "source_labels": ["synthetic"], "input_mode": mode}


def support(a=(), hosts=(), source="s0", u=0, background=1):
    counts = [0] * 32
    for k in a:
        counts[k] += 1
    r = [0] * 14
    for k in hosts:
        r[k] += 1
    return {"a_truth": counts, "u_instances": u, "u_voxels": u,
            "background_voxels": background, "r_hosts": r, "sources": [source]}


def test_support_uses_nonignored_anatomy_and_preserves_host_zero_and_no_host(metrics):
    b = batch()
    got = metrics.batch_support(b)
    assert got["a_truth"] == [2, 2] + [0] * 29 + [1]
    assert got["u_instances"] == 2 and got["u_voxels"] == 4
    assert got["background_voxels"] == 3
    assert got["r_hosts"] == [1] + [0] * 12 + [1]
    b["host"] = [torch.tensor([0, -1])]
    assert metrics.batch_support(b)["r_hosts"] == [1] + [0] * 13
    b["r_eligible"][0][0] = False
    assert metrics.batch_support(b)["r_hosts"] == [0] * 14


def test_unsupervised_targets_and_tiny_components_cannot_supply_panel_support(metrics):
    b = batch()
    b["a_supervised"].fill_(False)
    b["u_supervised"].fill_(False)
    b["point_weight"].flatten()[4] = 0
    got = metrics.batch_support(b)
    assert got["a_truth"] == [0] * 32
    assert got["u_instances"] == 0 and got["u_voxels"] == 0
    assert got["r_hosts"] == [0] * 14
    assert got["background_voxels"] == 2


@pytest.mark.parametrize("damage", ["shape", "mode", "host_length", "labels", "valid_nan"])
def test_invalid_target_contract_stops_before_measurement(metrics, damage):
    b = batch()
    if damage == "shape":
        b["instance"] = b["instance"][..., :-1]
    elif damage == "mode":
        b["input_mode"] = "4d"
    elif damage == "host_length":
        b["host"] = [torch.tensor([0])]
    elif damage == "labels":
        b["entity"].flatten()[0] = 33
    else:
        b["valid"].flatten()[0] = float("nan")
    with pytest.raises(ValueError, match="capability"):
        metrics.batch_support(b)


def test_panel_coverage_refuses_missing_classes_sources_or_background(metrics):
    supports = [support(range(32), (0,), "s0", 2), support((), (13,), "s1", 2), support(source="s2")]
    got = metrics.panel_support(supports)
    assert got["status"] == "PASS"
    assert got["a_truth"] == [1] * 32 and got["supported_r_hosts"] == 2
    assert got["source_count"] == 3 and got["observations"] == 3
    supports[0]["a_truth"][31] = 0
    assert metrics.panel_support(supports)["status"] == "INSUFFICIENT"
    supports[0]["a_truth"][31] = 1
    supports[2]["background_voxels"] = 0
    assert metrics.panel_support(supports)["status"] == "INSUFFICIENT"
    assert metrics.panel_support([])["status"] == "INSUFFICIENT"


def test_greedy_panel_prefers_new_anatomy_then_host_source_instances_and_lower_draw(metrics):
    pool = [support((0, 1), (0,), "s0", 1), support((0, 1), (0,), "s0", 1),
            support((2,), (0,), "s0", 4), support((), (13,), "s1", 1),
            support((), (), "s2", 3), support((), (), "s2", 4)]
    assert metrics.select_panel(pool, max_observations=5) == [0, 2, 3, 5, 4]
    assert metrics.select_panel(pool, max_observations=1) == [0]
    assert metrics.select_panel([], max_observations=16) == []
    with pytest.raises(ValueError):
        metrics.select_panel(pool, max_observations=17)


class ControlledModel(torch.nn.Module):
    """Only prediction generation is substituted; matching and metrics remain real."""
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))
        self.forward_count = 0
        self.bound = []

    def forward(self, image, valid, coords, local, *, input_mode):
        self.forward_count += 1
        shape = image.shape[2:]
        self.shape = shape
        levels = [{"feat": torch.ones(1, 2, *shape), "valid": valid} for _ in range(4)]
        return {"entity_presence": torch.full((1, 32), 20.),
                "event_presence": torch.tensor([[20., -20., 20., -20.]]), "levels": levels}

    def entity_masks(self, out, points=None):
        x = torch.full((1, 32, *self.shape), -20.)
        x.flatten(2)[0, 0, [0, 2]] = 20
        x.flatten(2)[0, 1, [1, 3]] = 20
        x.flatten(2)[0, 2, 5] = 20
        x.flatten(2)[0, 31, 6] = 20
        return x if points is None else x.flatten(2).gather(2, points[:, None].expand(-1, 32, -1))

    def event_masks(self, out, points=None):
        x = torch.full((1, 4, *self.shape), -20.)
        x.flatten(2)[0, 0, [0, 1]] = 20
        x.flatten(2)[0, 1, [2, 3]] = 20
        x.flatten(2)[0, 2, 4] = 20
        return x if points is None else x.flatten(2).gather(2, points[:, None].expand(-1, 4, -1))

    def bind(self, out, b, indices):
        # Geometry remains the model's own. The metrics may supply query indices only.
        self.bound.append(indices.clone())
        logits = torch.zeros(len(indices), 14)
        logits[indices == 0, 0] = math.log(13)
        return logits


def test_controlled_metrics_separate_mask_quality_presence_and_dense_union(metrics):
    model = ControlledModel().train()
    rng_before = torch.random.get_rng_state().clone()
    record = metrics.measure_batch(model, batch())
    got = metrics.aggregate_measurements([record])
    assert model.training and model.forward_count == 1
    assert torch.equal(rng_before, torch.random.get_rng_state())
    assert got["a"]["truth"] == [2, 2] + [0] * 29 + [1]
    assert got["a"]["predicted"] == [2, 2] + [0] * 29 + [1]
    assert got["a"]["intersection"] == [1, 1] + [0] * 29 + [1]
    assert got["a"]["macro"] == pytest.approx(2 / 3)
    assert got["u"]["matched_mask_dice"] == pytest.approx(1.)
    assert got["u"]["present_recall"] == pytest.approx(.5)
    assert got["u"]["unmatched_active_per_observation"] == 1
    assert got["u"]["union_dice"] == pytest.approx(4 / 7)
    assert got["r"]["truth"] == [1] + [0] * 12 + [1]
    assert got["r"]["correct"] == [1] + [0] * 13
    assert got["r"]["macro_accuracy"] == .5
    assert got["r"]["cross_entropy"] == pytest.approx((math.log(2) + math.log(14)) / 2)
    assert got["r"]["majority_accuracy"] == .5
    assert got["r"]["majority_macro_accuracy"] == .5
    assert model.bound[0].tolist() == [0, 1]
    assert record["cells"]["features"].device.type == "cpu"


def test_only_eligible_actual_matches_reach_relation_binding(metrics):
    b = batch()
    b["r_eligible"][0][1] = False
    model = ControlledModel().eval()
    got = metrics.aggregate_measurements([metrics.measure_batch(model, b)])
    assert model.bound[0].tolist() == [0]
    assert not model.training and got["r"]["eligible"] == 1
    assert got["r"]["macro_accuracy"] == 1
    b["host"][0][0] = -1
    got = metrics.aggregate_measurements([metrics.measure_batch(model, b)])
    assert got["r"]["eligible"] == 0 and got["r"]["macro_accuracy"] is None


def test_failure_restores_training_state_and_nonfinite_predictions_are_refused(metrics):
    model = ControlledModel().train()
    original = model.entity_masks
    def damaged(out, points=None):
        result = original(out, points)
        result.fill_(float("nan"))
        return result
    model.entity_masks = damaged
    with pytest.raises(ValueError, match="finite"):
        metrics.measure_batch(model, batch())
    assert model.training


def test_aggregation_keeps_empty_denominators_missing_and_refuses_nonfinite_counts(metrics):
    got = metrics.aggregate_measurements([])
    assert got["a"]["macro"] is None
    assert got["u"]["matched_mask_dice"] is None
    assert got["r"]["cross_entropy"] is None
    record = metrics.measure_batch(ControlledModel(), batch())
    record["u"]["dice_sum"] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        metrics.aggregate_measurements([record])


@pytest.mark.parametrize("mode,shape", [("2d", (1, 32, 32)), ("3d", (8, 32, 32))])
def test_real_tiny_model_measures_both_modes_without_mutating_parameters(metrics, mode, shape):
    from anatobind.aur.model import AnatoBindBrain
    cfg = dict(embed=32, depths=(1, 1, 1, 1), heads=(1, 2, 4, 8), window=(2, 4, 4),
               d_model=16, n_events=4, mask_dim=4, pixel_dim=8, dec_layers=1,
               dec_heads=2, rel_layers=1, rel_heads=2, use_checkpoint=False, dual_input=True)
    model = AnatoBindBrain(**cfg).train()
    before = {k: t.clone() for k, t in model.state_dict().items()}
    record = metrics.measure_batch(model, batch(shape, mode))
    got = metrics.aggregate_measurements([record])
    assert got["u"]["instances"] == 2 and got["r"]["eligible"] == 2
    assert len(record["cells"]["targets"]) <= 512
    assert record["cells"]["features"].shape[1] == 448
    assert model.training
    assert all(torch.equal(t, before[k]) for k, t in model.state_dict().items())


def cell_record(repeats=4):
    # Four cells/host: two even fit cells, two odd observation cells.
    targets = torch.arange(13).repeat_interleave(repeats)
    features = torch.nn.functional.one_hot(targets, 13).float()
    indices = torch.arange(len(targets))
    return {"cells": {"features": features, "targets": targets,
                      "parity": (indices % 2).bool(), "indices": indices}}


def test_retention_fits_once_on_even_cells_and_scores_frozen_odd_cells(metrics):
    records = [cell_record()]
    head = metrics.fit_retention(records)
    assert head["status"] == "PASS"
    baseline = metrics.score_retention(records, head)
    assert baseline["status"] == "PASS" and baseline["macro_dice"] == 1.
    assert baseline["truth"] == [2] * 13 and baseline["feature_relative_change"] == 0.
    altered = copy.deepcopy(records)
    odd = altered[0]["cells"]["parity"]
    altered[0]["cells"]["features"][odd] = altered[0]["cells"]["features"][odd].roll(1, dims=1)
    frozen = copy.deepcopy(head)
    degraded = metrics.score_retention(altered, head)
    assert degraded["macro_dice"] == 0.
    assert degraded["feature_relative_change"] == pytest.approx(math.sqrt(2))
    assert torch.equal(head["weight"], frozen["weight"])
    assert torch.equal(head["mean"], frozen["mean"])


def test_fit_normalization_and_ridge_ignore_observation_features(metrics):
    original = [cell_record()]
    changed = copy.deepcopy(original)
    changed[0]["cells"]["features"][changed[0]["cells"]["parity"]] *= 1000
    a, b = metrics.fit_retention(original), metrics.fit_retention(changed)
    assert torch.equal(a["weight"], b["weight"])
    assert torch.equal(a["mean"], b["mean"]) and torch.equal(a["scale"], b["scale"])


def test_class_balanced_ridge_assigns_equal_bias_to_unequal_class_support(metrics):
    record = cell_record()
    cells = record["cells"]
    cells["features"].fill_(1.)
    extra = torch.zeros(80, dtype=torch.long)
    cells["targets"] = torch.cat((cells["targets"], extra))
    cells["features"] = torch.cat((cells["features"], torch.ones(80, 13)))
    cells["indices"] = torch.arange(132)
    cells["parity"] = (cells["indices"] % 2).bool()
    head = metrics.fit_retention([record])
    # No varying feature can distinguish hosts. Equal class loss gives uniform bias,
    # even with 42 fit cells in host0 versus two in every other host.
    assert head["weight"][-1].tolist() == pytest.approx([1 / 13] * 13)


def test_cell_selection_is_bounded_balanced_and_keeps_grid_parity(metrics):
    b = batch((1, 32, 64))
    b["valid"].fill_(1)
    b["entity"].fill_(1)  # SynthSeg2: left white matter, host0.
    b["entity"].flatten()[1024:] = 2  # SynthSeg41: right white matter, host1.
    record = metrics.measure_batch(ControlledModel(), b)
    cells = record["cells"]
    assert len(cells["targets"]) == 512
    assert torch.bincount(cells["targets"], minlength=13).tolist() == [256, 256] + [0] * 11
    assert torch.equal(cells["parity"], (cells["indices"] % 2).bool())
    assert int(cells["indices"][0]) == 0 and int(cells["indices"][-1]) == 2047


def test_host_majority_cells_with_less_than_half_valid_support_are_excluded(metrics):
    class PooledModel(ControlledModel):
        def forward(self, *args, **kwargs):
            out = super().forward(*args, **kwargs)
            for level in out["levels"]:
                level["feat"] = torch.ones(1, 2, 1, 1, 2)
                level["valid"] = torch.ones(1, 1, 1, 2)
            return out
    b = batch((1, 2, 8))
    b["entity"].fill_(1)
    b["a_ignore"].zero_()
    # First 2x4 cell has three valid voxels; second has four (exactly .5).
    b["valid"].zero_()
    b["valid"][0, 0, 0, :7] = 1
    b["point_weight"] = b["valid"].clone()
    b["instance"].zero_()
    b["n_instances"] = [0]
    b["host"] = [torch.empty(0, dtype=torch.long)]
    b["r_eligible"] = [torch.empty(0, dtype=torch.bool)]
    b["valid"][0, 0, 0, 3] = 0
    b["valid"][0, 0, 0, 7] = 1
    b["point_weight"] = b["valid"].clone()
    cells = metrics.measure_batch(PooledModel(), b)["cells"]
    assert cells["indices"].tolist() == [1]
    assert cells["targets"].tolist() == [0] and cells["parity"].tolist() == [True]


@pytest.mark.parametrize("ignored,feature_valid,expected", [(5, 1., []), (4, 1., [1]), (0, .5, [])])
def test_retention_pool_excludes_ignored_voxels_and_invalid_feature_cells(metrics, ignored, feature_valid, expected):
    class OneCellModel(ControlledModel):
        def forward(self, *args, **kwargs):
            out = super().forward(*args, **kwargs)
            for level in out["levels"]:
                level["feat"] = torch.ones(1, 2, 1, 1, 1)
                level["valid"] = torch.full((1, 1, 1, 1), feature_valid)
            return out
    b = batch()
    b["valid"].fill_(1)
    b["point_weight"].fill_(1)
    b["entity"].fill_(2)  # Usable voxels belong to right white matter, host1.
    b["entity"].flatten()[:ignored] = 1  # Ignored host0 must have no vote.
    b["a_ignore"].zero_()
    b["a_ignore"].flatten()[:ignored] = True
    cells = metrics.measure_batch(OneCellModel(), b)["cells"]
    # Five ignored voxels leave3/8 support; four leave exactly the accepted1/2.
    assert cells["targets"].tolist() == expected
    assert cells["indices"].tolist() == ([0] if expected else [])


@pytest.mark.parametrize("damage", ["class", "parity_support", "positions", "labels", "nan"])
def test_retention_refuses_changed_support_or_nonfinite_features(metrics, damage):
    records = [cell_record()]
    head = metrics.fit_retention(records)
    cells = records[0]["cells"]
    if damage == "class":
        cells["targets"][cells["targets"] == 12] = 11
        assert metrics.fit_retention(records)["status"] == "INSUFFICIENT"
    elif damage == "parity_support":
        assert metrics.fit_retention([cell_record(2)])["status"] == "INSUFFICIENT"
    elif damage == "positions":
        cells["indices"][0] = 1000
        with pytest.raises(ValueError):
            metrics.score_retention(records, head)
    elif damage == "labels":
        cells["targets"][0] = 1
        with pytest.raises(ValueError):
            metrics.score_retention(records, head)
    else:
        cells["features"][0, 0] = float("nan")
        with pytest.raises(ValueError, match="finite"):
            metrics.score_retention(records, head)


def measured(a, u, presence, r, ce):
    return {"a": {"macro": a, "truth": [1] * 32},
            "u": {"matched_mask_dice": u, "present_recall": presence, "instances": 4, "union_dice": .3},
            "r": {"macro_accuracy": r, "cross_entropy": ce, "eligible": 4,
                  "truth": [2, 2] + [0] * 12, "supported_hosts": 2,
                  "majority_macro_accuracy": .5}}


def test_verdict_requires_both_stages_gains_and_stage_three_ce_drop(metrics):
    initial = measured(.2, .3, .2, .25, 2.)
    two = measured(.26, .41, .5, .25, 2.)
    three = {"initial": two, "final": measured(.26, .41, .5, .6, 1.59)}
    retention = {"status": "PASS", "macro_dice": .8}
    result = metrics.capability_verdict(initial, two, three, retention,
                                         {"status": "PASS", "macro_dice": .751})
    assert result["status"] == "PASS"
    three["final"]["r"]["cross_entropy"] = 1.61
    assert metrics.capability_verdict(initial, two, three, retention, retention)["status"] == "FAIL"
    three["final"]["r"]["cross_entropy"] = 1.59
    two["u"]["present_recall"] = .49
    assert metrics.capability_verdict(initial, two, three, retention, retention)["status"] == "FAIL"


def test_verdict_refuses_widespread_masks_and_constant_host_baseline(metrics):
    a = measured(.2, .3, .2, .25, 2.)
    b = measured(.3, .5, .7, .6, 1.)
    c = {"initial": measured(.3, .5, .7, .25, 2.), "final": copy.deepcopy(b)}
    retention = {"status": "PASS", "macro_dice": .8}
    b["u"]["union_dice"] = .249
    assert metrics.capability_verdict(a, b, c, retention, retention)["status"] == "FAIL"
    b["u"]["union_dice"] = .25
    c["final"]["r"]["macro_accuracy"] = .59
    assert metrics.capability_verdict(a, b, c, retention, retention)["status"] == "FAIL"


def test_verdict_accepts_exact_prespecified_threshold_boundaries(metrics):
    initial = measured(.2, .3, .2, .25, 2.)
    final = measured(.25, .4, .5, .6, 1.6)
    retention = {"status": "PASS", "macro_dice": .8}
    result = metrics.capability_verdict(initial, final, {"initial": initial, "final": final},
                                         retention, {"status": "PASS", "macro_dice": .75})
    assert result["status"] == "PASS"


@pytest.mark.parametrize("damage", ["missing_a", "no_u", "no_r", "nan", "missing_readout", "retention_drop"])
def test_verdict_never_accepts_missing_support_nonfinite_values_or_lost_retention(metrics, damage):
    a = measured(.2, .3, .2, .25, 2.)
    b = measured(.3, .5, .7, .6, 1.)
    c = {"initial": measured(.3, .5, .7, .25, 2.), "final": copy.deepcopy(b)}
    r0 = {"status": "PASS", "macro_dice": .8}
    rf = {"status": "PASS", "macro_dice": .8}
    if damage == "missing_a":
        a["a"]["truth"][31] = 0
    elif damage == "no_u":
        b["u"]["instances"] = 0
    elif damage == "no_r":
        c["final"]["r"]["eligible"] = 0
    elif damage == "nan":
        b["a"]["macro"] = float("nan")
    elif damage == "missing_readout":
        rf["status"] = "INSUFFICIENT"
    else:
        rf["macro_dice"] = .74
    result = metrics.capability_verdict(a, b, c, r0, rf)
    assert result["status"] == ("FAIL" if damage == "retention_drop" else "INSUFFICIENT")

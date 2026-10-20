"""Formal orchestration must preserve recovery and enforce qualified stage transitions."""
from copy import deepcopy
import importlib

import pytest
import torch

from test_aur_unified_training import MODEL, factory as legacy_factory, assert_state_equal


CONTRACT = {"contract_sha256": "1" * 64, "runtime_sha256": "2" * 64}


def module():
    return importlib.import_module("anatobind.aur.unified_run")


def config(**changes):
    cfg = dict(schema_version=1, purpose="synthetic_test", model=deepcopy(MODEL), seed=53,
               steps=4, lr=1e-3, weight_decay=.1, points=32,
               lambda_r=1., lambda_h=.1, u_mask_profile="unmatched_empty_v1",
               mode_cycle=["2d", "3d"], batch_size=1, accumulation_steps=2,
               warmup_steps=1, backbone_warm_steps=1, backbone_warm_scale=.1,
               crops={"2d": [1,16,16], "3d": [4,16,16]},
               ssl=dict(mask_ratio=.4, block_mm=[4.,8.], contrast_weight=0.),
               validation_every=2, checkpoint_every=2, max_seconds=60.,
               planar_relation_policy="geometry_weak2d_v1", hcp_cap=.25)
    return {**cfg, **changes}


def factory(mode, draw):
    batch = legacy_factory(mode, draw)
    batch["source_labels"] = ["synthetic"]
    return batch


@pytest.mark.parametrize("bad", [dict(accumulation_steps=0), dict(mode_cycle=["3d"]),
    dict(crops={"2d":[2,16,16],"3d":[4,16,16]}), dict(max_seconds=float("nan")),
    dict(ssl=dict(mask_ratio=.4,block_mm=[4.,8.],contrast_weight=.1))])
def test_invalid_formal_configuration_fails_before_training(bad):
    with pytest.raises(ValueError, match="configuration"):
        module().validate_run_config(config(**bad))


def test_accumulated_cpu_resume_matches_uninterrupted_all_states(tmp_path):
    cls = module().UnifiedRun
    complete = cls("I", config(), CONTRACT)
    full_logs = [complete.step(factory) for _ in range(4)]
    full_path = complete.save(tmp_path / "full.pt")
    split = cls("I", config(), CONTRACT)
    split_logs = [split.step(factory)]
    path = split.save(tmp_path / "split.pt")
    torch.rand(9)
    restored = cls.resume(path, config(), CONTRACT)
    split_logs += [restored.step(factory) for _ in range(3)]
    assert full_logs == split_logs
    assert restored.cursors == {"2d":4,"3d":4}
    assert restored.exposures == {"2d":4,"3d":4}
    assert restored.source_exposures == {"synthetic":8}
    assert restored.source_mode_exposures == {"2d":{"synthetic":4},"3d":{"synthetic":4}}
    resumed_path = restored.save(tmp_path / "resumed.pt")
    assert_state_equal(torch.load(full_path, weights_only=True), torch.load(resumed_path, weights_only=True))


def test_validation_preserves_training_rng_cursors_and_weights():
    runner = module().UnifiedRun("I", config(), CONTRACT)
    before = runner.export()
    rng = runner._rng()
    report = runner.validate(factory, n_batches=2)
    assert set(report["modes"]) == {"2d","3d"}
    assert report["criterion"] >= 0
    assert runner.step_index == 0
    assert runner.cursors == {"2d":0,"3d":0}
    assert_state_equal(rng, runner._rng())
    assert_state_equal(before, runner.export())
    assert runner.model.training


def test_formal_stage_i_requires_explicit_registered_warm_start():
    with pytest.raises(ValueError, match="warm.start"):
        module().UnifiedRun("I", config(purpose="formal"), CONTRACT)


def test_warm_start_precedes_device_transfer_and_optimizer_construction(monkeypatch):
    import anatobind.aur.ssl.unified_init as init
    built = []
    original = torch.optim.AdamW
    def optimizer(*args, **kwargs):
        built.append(True)
        return original(*args, **kwargs)
    def warm_start(model, checkpoint, **kwargs):
        assert not built, "optimizer must not retain parameters from a prior device transfer"
        assert all(p.device.type == "cpu" for p in model.parameters())
        return {"kind":"synthetic_warm_start"}
    monkeypatch.setattr(torch.optim,"AdamW",optimizer)
    monkeypatch.setattr(init,"warm_start_stage_one",warm_start)
    run = module().UnifiedRun("I",config(purpose="formal"),CONTRACT,
                              warm_start={"checkpoint":"synthetic"})
    assert built and run.lineage == {"kind":"synthetic_warm_start"}
    model_ids = {id(p) for p in run.model.parameters() if p.requires_grad}
    assert {id(p) for g in run.optimizer.param_groups for p in g["params"]} == model_ids


def test_formal_stage_ii_rejects_probe_initializer_even_with_fake_pass():
    probe = module().UnifiedRun("I", config(purpose="resource_probe"), CONTRACT)
    for _ in range(4):
        probe.step(factory)
    with pytest.raises(ValueError, match="formal"):
        module().UnifiedRun("II", config(purpose="formal"), CONTRACT, initializer=probe.export(),
                            g1_report={"pass":True})


def test_sampler_is_source_then_patient_then_sequence_balanced_and_stateless():
    cls = module().PatientBalancedSampler
    rows = [dict(source="pdgm",patient="a",sequence="T1"),
            dict(source="pdgm",patient="a",sequence="T2"),
            dict(source="pdgm",patient="b",sequence="T1"),
            dict(source="hcp",patient="c",sequence="T1")]
    sampler = cls(rows, seed=5, hcp_cap=.25)
    assert sampler.source_probabilities == {"hcp":.25,"pdgm":.75}
    draws = [sampler.indices("2d", i, 2) for i in range(100)]
    assert draws == [sampler.indices("2d", i, 2) for i in range(100)]
    for indices in draws:
        identities = {(rows[i]["source"],rows[i]["patient"]) for i in indices}
        assert len(identities) == 2


def test_sampler_balances_sequences_before_repeated_sessions():
    rows=[dict(source='pdgm',patient='a',sequence='T1') for _ in range(9)]
    rows += [dict(source='pdgm',patient='a',sequence='T2')]
    sampler=module().PatientBalancedSampler(rows,seed=5)
    t2=sum(rows[sampler.indices('2d',i,1)[0]]['sequence']=='T2' for i in range(1000))
    assert 400<t2<600


def test_formal_single_mode_budget_is_rejected():
    with pytest.raises(ValueError,match='configuration'):
        module().validate_run_config(config(purpose='formal',steps=1,warmup_steps=0))


def test_unique_patient_batch_cannot_force_hcp_above_sampling_cap():
    rows=[dict(source='pdgm',patient='a',sequence='T1')]
    rows += [dict(source='hcp',patient=str(i),sequence='T1') for i in range(9)]
    s=module().PatientBalancedSampler(rows,seed=5,hcp_cap=.25)
    with pytest.raises(ValueError,match='non.HCP'):
        s.indices('2d',0,4)


def test_hcp_probability_remains_capped_after_small_source_exhausts():
    rows=[dict(source='pdgm',patient='a',sequence='T1')]
    rows += [dict(source='isles',patient=str(i),sequence='DWI') for i in range(20)]
    rows += [dict(source='hcp',patient=str(i),sequence='T1') for i in range(30)]
    s=module().PatientBalancedSampler(rows,seed=5,hcp_cap=.25)
    n=sum(rows[j]['source']=='hcp' for i in range(2000) for j in s.indices('2d',i,10))
    assert .23 < n/20000 < .27


def test_failed_batch_cannot_be_saved_as_resumable(tmp_path):
    run=module().UnifiedRun('I',config(),CONTRACT)
    def invalid(mode,draw):
        batch=factory(mode,draw)
        batch['view1'][:]=float('nan')
        return batch
    with pytest.raises(FloatingPointError,match='nonfinite'):
        run.step(invalid)
    with pytest.raises(ValueError,match='failed'):
        run.save(tmp_path/'state.pt')
    assert not (tmp_path/'state.pt').exists()


def test_resume_rejects_configuration_data_and_counter_tampering(tmp_path):
    cls = module().UnifiedRun
    run = cls("I", config(), CONTRACT)
    run.step(factory)
    path = run.save(tmp_path / "state.pt")
    with pytest.raises(ValueError, match="contract|configuration"):
        cls.resume(path, config(steps=6), CONTRACT)
    with pytest.raises(ValueError, match="contract|configuration"):
        cls.resume(path, config(), {**CONTRACT,"contract_sha256":"3"*64})
    payload = torch.load(path, weights_only=True)
    payload["cursors"]["2d"] += 1
    changed = tmp_path / "changed.pt"
    torch.save(payload, changed)
    with pytest.raises(ValueError, match="cursor"):
        cls.resume(changed, config(), CONTRACT)


@pytest.mark.parametrize('stage',['II','III'])
def test_supervised_stage_accumulation_resume_and_validation_preserve_full_state(tmp_path,stage):
    cls=module().UnifiedRun
    def initialize():
        first=cls('I',config(),CONTRACT)
        second=cls('II',config(),CONTRACT,initializer=first.export())
        return second if stage=='II' else cls('III',config(),CONTRACT,initializer=second.export())
    full=initialize()
    logs=[full.step(factory) for _ in range(4)]
    expected=full.save(tmp_path/'full.pt')
    partial=initialize()
    actual=[partial.step(factory)]
    before=partial.export()
    partial.validate(factory)
    assert_state_equal(before,partial.export())
    restored=cls.resume(partial.save(tmp_path/'partial.pt'),config(),CONTRACT)
    actual += [restored.step(factory) for _ in range(3)]
    assert logs==actual
    assert_state_equal(torch.load(expected,weights_only=True),
                       torch.load(restored.save(tmp_path/'resumed.pt'),weights_only=True))

"""Durable orchestration over the real small unified CPU training core."""
from copy import deepcopy
import importlib
import json
from pathlib import Path

import pytest
import torch

from test_aur_unified_run import config, factory
from test_aur_unified_training import assert_state_equal


def api():
    return importlib.import_module("anatobind.aur.unified_launcher")


def recipe(**over):
    return dict(schema_version=1, stage_config=config(accumulation_steps=1, **over), validation_batches=1)


def contract():
    return dict(version="synthetic", contract_sha256="1" * 64, summary={},
                train_ssl=[dict(source="synthetic", patient="train", role="train_ssl", split="train")],
                train_supervised=[dict(source="synthetic", patient="train", role="train_supervised", split="train")],
                validation_ssl=[dict(source="synthetic", patient="val", role="ssl_validation", split="val")],
                validation=[dict(source="synthetic", patient="val", role="development_validation", split="val")])


@pytest.fixture
def launcher(monkeypatch):
    module = api()
    monkeypatch.setattr(module, "validate_private_output", lambda path: Path(path).resolve())
    return module


def run(module, out, cfg=None, **kwargs):
    return module.run_stage(recipe() if cfg is None else cfg, contract(), "I", out,
                            train_factory=factory, validation_factory=factory, **kwargs)


def pointer(out, name):
    return json.loads((out / f"{name}.json").read_text())


def test_recipe_is_required_and_cli_does_not_echo_private_bad_arguments(capsys):
    module = api()
    with pytest.raises(ValueError, match="recipe"):
        module.validate_recipe({})
    with pytest.raises(SystemExit):
        module.parser().parse_args(["--stage", "PRIVATE_CASE_SECRET"])
    assert "PRIVATE_CASE_SECRET" not in capsys.readouterr().err
    assert not hasattr(module.parser().parse_args(["--config", "a", "--data-contract", "b", "--stage", "I", "--out", "c", "--device", "cpu"]), "synthetic")


def test_cli_rejects_synthetic_recipe_before_creating_output(tmp_path, capsys):
    module = api()
    cfg = tmp_path / "recipe.json"
    cfg.write_text(json.dumps(recipe()))
    assert module.main(["--config", str(cfg), "--data-contract", "PRIVATE_CASE_SECRET", "--stage", "I", "--out", str(tmp_path / "out"), "--device", "cpu"]) == 1
    assert "PRIVATE_CASE_SECRET" not in capsys.readouterr().out
    assert not (tmp_path / "out").exists()


def test_completed_run_selects_actual_best_and_preserves_last(launcher, tmp_path, monkeypatch):
    original = launcher.UnifiedRun.validate
    def worsening(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        result["criterion"] = self.step_index
        return result
    monkeypatch.setattr(launcher.UnifiedRun, "validate", worsening)
    out = tmp_path / "complete"
    result = run(launcher, out)
    assert result["status"] == "COMPLETE" and result["steps"] == 4
    best, last = pointer(out, "best"), pointer(out, "last")
    assert best["step"] == 2 and last["step"] == 4
    selected = torch.load(out / result["selected_export"], weights_only=True)
    assert selected["step"] == 2 and selected["complete"] is False
    assert "completion_certificate" not in selected  # synthetic work cannot release formal gates
    assert [row["step"] for row in launcher.read_jsonl(out / "steps.jsonl")] == [1, 2, 3, 4]


def test_timeout_saves_completed_boundary_and_resume_matches_full_state(launcher, tmp_path, monkeypatch):
    complete = tmp_path / "complete"
    run(launcher, complete)
    interrupted = tmp_path / "interrupted"
    clock = [0.]
    monkeypatch.setattr(launcher, "monotonic", lambda: clock[0])
    def expires(mode, draw):
        value = factory(mode, draw)
        clock[0] = 61.
        return value
    first = launcher.run_stage(recipe(), contract(), "I", interrupted, train_factory=expires, validation_factory=factory)
    assert first["status"] == "INTERRUPTED" and first["steps"] == 1
    assert not (interrupted / "completion.json").exists()
    assert not (interrupted / "selected_export.pt").exists()
    clock[0] = 0.
    restored = run(launcher, interrupted, resume=interrupted / pointer(interrupted, "last")["path"])
    assert restored["status"] == "COMPLETE"
    final_full = torch.load(complete / pointer(complete, "last")["path"], weights_only=True)
    final_restored = torch.load(interrupted / pointer(interrupted, "last")["path"], weights_only=True)
    initial=torch.load(interrupted/first['last']['path'],weights_only=True)
    assert final_restored['execution_id']==initial['execution_id']
    assert final_full['execution_id']!=final_restored['execution_id']
    # Separate output directories identify separate executions; all numerical
    # state and all other recovery metadata must still be exactly equal.
    final_full.pop('execution_id');final_restored.pop('execution_id')
    assert_state_equal(final_full, final_restored)
    assert [row["step"] for row in launcher.read_jsonl(interrupted / "steps.jsonl")] == [1, 2, 3, 4]


def test_final_step_timeout_resumes_missing_validation_without_new_updates(launcher,tmp_path,monkeypatch):
    out=tmp_path/'last_step_timeout'
    clock=[0.]
    monkeypatch.setattr(launcher,'monotonic',lambda:clock[0])
    def expires(mode,draw):
        batch=factory(mode,draw)
        if mode=='3d' and draw==1: clock[0]=61.
        return batch
    first=launcher.run_stage(recipe(),contract(),'I',out,train_factory=expires,validation_factory=factory)
    assert first['status']=='INTERRUPTED' and first['steps']==4
    assert [r['step'] for r in launcher.read_jsonl(out/'validation.jsonl')]==[2]
    clock[0]=0.
    def no_more_steps(*args): pytest.fail('completed optimizer budget repeated')
    result=launcher.run_stage(recipe(),contract(),'I',out,resume=out/pointer(out,'last')['path'],
                              train_factory=no_more_steps,validation_factory=factory)
    assert result['status']=='COMPLETE'
    assert [r['step'] for r in launcher.read_jsonl(out/'validation.jsonl')]==[2,4]


def test_recovery_rebuilds_best_from_committed_validation(launcher,tmp_path,monkeypatch):
    out=tmp_path/'best_pointer_crash'
    original=launcher._json
    failed=[False]
    def fail_best(path,value,**kw):
        if path.name=='best.json' and not failed[0]:
            failed[0]=True
            raise OSError('simulated pointer crash')
        return original(path,value,**kw)
    monkeypatch.setattr(launcher,'_json',fail_best)
    with pytest.raises(OSError): run(launcher,out)
    assert not (out/'best.json').exists()
    monkeypatch.setattr(launcher,'_json',original)
    result=run(launcher,out,resume=out/pointer(out,'last')['path'])
    val=launcher.read_jsonl(out/'validation.jsonl')
    assert result['best']['criterion']==min(r['criterion'] for r in val)


def test_launcher_saves_actual_source_bytes(launcher,tmp_path):
    out=tmp_path/'source'
    run(launcher,out)
    snapshot=pointer(out,'source_snapshot')
    assert all((out/'source_snapshot'/name).is_file() for name in snapshot)
    assert all(launcher.file_sha256(out/'source_snapshot'/name)==value for name,value in snapshot.items())


def test_resume_finishes_export_already_written_before_completion_marker(launcher,tmp_path,monkeypatch):
    out=tmp_path/'export_crash'
    original=launcher._json
    once=[False]
    def crash(path,value,**kwargs):
        if path.name=='completion.json' and not once[0]:
            once[0]=True
            raise OSError('completion marker not written')
        return original(path,value,**kwargs)
    monkeypatch.setattr(launcher,'_json',crash)
    with pytest.raises(OSError): run(launcher,out)
    digest=launcher.file_sha256(out/'selected_export.pt')
    monkeypatch.setattr(launcher,'_json',original)
    result=run(launcher,out,resume=out/pointer(out,'last')['path'])
    assert result['status']=='COMPLETE'
    assert launcher.file_sha256(out/'selected_export.pt')==digest


def test_resume_archives_uncommitted_tail_without_duplicate_canonical_steps(launcher, tmp_path, monkeypatch):
    out = tmp_path / "tail"
    clock = [0.]
    monkeypatch.setattr(launcher, "monotonic", lambda: clock[0])
    def expires(mode, draw):
        value = factory(mode, draw)
        clock[0] = 61.
        return value
    launcher.run_stage(recipe(), contract(), "I", out, train_factory=expires, validation_factory=factory)
    with (out / "steps.jsonl").open("a") as stream:
        stream.write(json.dumps(dict(step=2, uncommitted=True)) + "\n")
    clock[0] = 0.
    run(launcher, out, resume=out / pointer(out, "last")["path"])
    assert [row["step"] for row in launcher.read_jsonl(out / "steps.jsonl")] == [1, 2, 3, 4]
    archives = list(out.glob("recovery_steps_*.jsonl"))
    assert len(archives) == 1 and "uncommitted" in archives[0].read_text()


def test_resume_rejects_source_or_pointer_mismatch(launcher, tmp_path, monkeypatch):
    out = tmp_path / "bad"
    run(launcher, out)
    checkpoint = out / pointer(out, "last")["path"]
    snapshot = launcher.source_snapshot()
    monkeypatch.setattr(launcher, "source_snapshot", lambda: {**snapshot, "changed.py": "0" * 64})
    with pytest.raises(ValueError, match="source|configuration"):
        run(launcher, out, resume=checkpoint)
    monkeypatch.undo()
    monkeypatch.setattr(launcher, "validate_private_output", lambda path: Path(path).resolve())
    last = pointer(out, "last")
    last["sha256"] = "0" * 64
    (out / "last.json").write_text(json.dumps(last))
    with pytest.raises(ValueError, match="pointer"):
        run(launcher, out, resume=checkpoint)


def test_validation_uses_only_held_out_role_and_verifies_contract_twice(launcher, tmp_path, monkeypatch):
    records, verifications = [], []
    frozen = contract()
    monkeypatch.setattr(launcher, "read_contract", lambda path, verify=True: verifications.append(verify) or deepcopy(frozen))
    class Views:
        def __init__(self, rows, stage, cfg, **options):
            records.append((rows[0]["role"], options["validation"]))
        def __call__(self, mode, draw):
            return factory(mode, draw)
    monkeypatch.setattr(launcher, "QualifiedViews", Views)
    launcher.run_stage(recipe(), tmp_path / "contract", "I", tmp_path / "isolated")
    assert records == [("train_ssl", False), ("ssl_validation", True)]
    assert verifications == [True, True]
    frozen["validation_ssl"][0]["patient"] = "train"
    with pytest.raises(ValueError, match="partition"):
        launcher.run_stage(recipe(), tmp_path / "contract", "I", tmp_path / "overlap")


def test_generic_failure_has_private_detail_and_does_not_become_complete(launcher, tmp_path, capsys):
    out = tmp_path / "failed"
    def broken(mode, draw):
        raise RuntimeError("PRIVATE_CASE_SECRET")
    with pytest.raises(RuntimeError):
        launcher.run_stage(recipe(), contract(), "I", out, train_factory=broken, validation_factory=factory)
    assert pointer(out, "status")["status"] == "FAILED"
    assert not (out / "completion.json").exists()
    assert "PRIVATE_CASE_SECRET" in next(out.glob("failure_*.txt")).read_text()
    assert "PRIVATE_CASE_SECRET" not in capsys.readouterr().out


@pytest.fixture(scope='module')
def phase_states():
    from anatobind.aur.unified_run import UnifiedRun
    import tempfile
    identity={'contract_sha256':'1'*64,'runtime_sha256':'2'*64}
    run=UnifiedRun('I',config(accumulation_steps=1),identity)
    with tempfile.TemporaryDirectory() as directory:
        for _ in range(2): run.step(factory)
        selected=torch.load(run.save(Path(directory)/'selected.pt'),weights_only=True)
        for _ in range(2): run.step(factory)
        final=torch.load(run.save(Path(directory)/'final.pt'),weights_only=True)
    for state in (selected,final):
        state['purpose']=state['cfg']['purpose']='formal'
        state['execution_id']='4'*64
    return selected,final


@pytest.mark.parametrize("damage", [None, "selected_step", "selected_state", "final_hash", "final_budget", "purpose", "data",
                                   "incompatible_lineage","missing_resume_state","execution_id",
                                   "empty_optimizer","stale_optimizer","empty_rng"])
def test_completion_certificate_binds_selected_weights_to_actual_completed_phase(tmp_path, monkeypatch, damage,phase_states):
    artifacts = importlib.import_module("anatobind.aur.unified_artifacts")
    from anatobind.aur.unified_train import _state_hash
    from anatobind.aur.data_contract import file_sha256
    from anatobind.aur.unified_run import validate_run_config
    monkeypatch.setattr(artifacts, "validate_private_output", lambda path: Path(path).resolve())
    identity = {"contract_sha256": "1" * 64, "runtime_sha256": "2" * 64}
    selected,final=deepcopy(phase_states)
    final_path, selected_path = tmp_path / "final.pt", tmp_path / "selected.pt"
    if damage == "final_budget":
        final["step"] = 3
    elif damage=='incompatible_lineage': final['lineage']={'kind':'unrelated'}
    elif damage=='missing_resume_state':
        for key in ('optimizer','scheduler','rng','cursors','exposures','source_exposures','source_mode_exposures'):
            final.pop(key)
    elif damage=='execution_id': final['execution_id']='5'*64
    elif damage=='empty_optimizer': final['optimizer']['state']={}
    elif damage=='stale_optimizer':
        for state in final['optimizer']['state'].values(): state['step'].fill_(1)
    elif damage=='empty_rng': final['rng']['torch']=torch.empty(0,dtype=torch.uint8)
    torch.save(final, final_path)
    torch.save(selected, selected_path)
    certificate = dict(version=1, stage="I", purpose="formal", executed_steps=4, budget_steps=4,
                       selected_step=2, selected_model_state_sha256=_state_hash(selected["model_state_dict"]),
                       selected_backbone_state_sha256=_state_hash(selected["backbone_state_dict"]), data_contract=identity,
                       final_checkpoint_path=str(final_path.resolve()), final_checkpoint_sha256=file_sha256(final_path),
                       selected_checkpoint_path=str(selected_path.resolve()), selected_checkpoint_sha256=file_sha256(selected_path),
                       g1_report_sha256=None,execution_id='4'*64)
    initializer = deepcopy(selected)
    initializer.pop("cfg")
    initializer["completion_certificate"] = certificate
    if damage == "selected_step":
        certificate["selected_step"] = 3
    elif damage == "selected_state":
        key=next(k for k in initializer['model_state_dict'] if not k.startswith('backbone.'))
        initializer["model_state_dict"][key] += 1
    elif damage == "final_hash":
        certificate["final_checkpoint_sha256"] = "0" * 64
    elif damage == "purpose":
        initializer["purpose"] = "resource_probe"
    elif damage == "data":
        certificate["data_contract"] = {"contract_sha256": "0" * 64}
    assert artifacts.verify_completion_certificate(initializer, data_contract=identity) is (damage is None)

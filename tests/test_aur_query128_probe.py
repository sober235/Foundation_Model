"""Probe completion is distinct from formal phase certification."""
import importlib.util
import json
from pathlib import Path
import pytest
from test_aur_unified_run import config, factory, CONTRACT
from anatobind.aur.unified_run import UnifiedRun
from anatobind.aur.unified_launcher import atomic_torch_save
from anatobind.aur.data_contract import file_sha256

SCRIPT=Path(__file__).resolve().parents[1]/'scripts/aur_query128_probe.py'


def api():
    spec=importlib.util.spec_from_file_location('probe_under_test',SCRIPT)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def prepare_stage(tmp_path,stage):
    cfg=config(purpose='resource_probe',steps=12,accumulation_steps=1,validation_every=10,checkpoint_every=5)
    first=UnifiedRun('I',cfg,CONTRACT)
    initializer=first.export()
    if stage=='III':
        second=UnifiedRun('II',cfg,CONTRACT,initializer=initializer)
        for _ in range(8):second.step(factory)
        initializer=second.export()
    runner=UnifiedRun(stage,cfg,CONTRACT,initializer=initializer)
    runner.execution_id='a'*64
    out=tmp_path/stage;out.mkdir(mode=0o700)
    records=[];plan=[]
    for step in range(8):
        mode=runner.next_mode;record=runner.step(factory);records.append(record)
        plan.append({'mode':mode,'draw':step//2,'row_index':0,'expected_instances':2,
            'expected_r_eligible':2,'u_supervised':True,'stress':False})
    last_path=Path(runner.save(out/'last.pt'))
    exported=runner.export();export_path=out/'selected_export.pt';atomic_torch_save(exported,export_path)
    (out/'last.json').write_text(json.dumps({'path':last_path.name,'sha256':file_sha256(last_path),'step':8}))
    (out/'completion.json').write_text(json.dumps({'selected_export_sha256':file_sha256(export_path),'formal_gate_certificate':False}))
    (out/'run_config.json').write_text(json.dumps({'recipe':{'stage_config':cfg},'stage':stage,
        'data_contract':runner.data_contract,'execution_id':runner.execution_id}))
    (out/'summary_1.json').write_text(json.dumps({'status':'PROBE_COMPLETE','purpose':'resource_probe',
        'steps':8,'execution_cap':8,'scheduler_horizon_steps':12}))
    return api(),out,cfg,{'steps':plan},records


@pytest.mark.parametrize('stage',['II','III'])
def test_probe_completion_never_claims_formal_quality(tmp_path,stage):
    probe,out,cfg,plan,records=prepare_stage(tmp_path,stage)
    summary=probe.verify_probe_stage(out,stage,cfg,CONTRACT,plan,records)
    assert summary['steps']==8 and summary['scheduler_horizon_steps']==12
    assert summary['mode_exposures']=={'2d':4,'3d':4}
    assert 'formal_gate_certificate' not in summary


def test_probe_verifier_rejects_incomplete_or_nonfinite_steps(tmp_path):
    probe,out,cfg,plan,records=prepare_stage(tmp_path,'II')
    with pytest.raises(ValueError,match='probe stage boundary'):
        probe.verify_probe_stage(out,'II',cfg,CONTRACT,plan,records[:7])
    records[3]['loss']=float('nan')
    with pytest.raises(ValueError,match='capacity or loss'):
        probe.verify_probe_stage(out,'II',cfg,CONTRACT,plan,records)


def test_probe_entry_creates_private_owned_output_and_redacts_failures(tmp_path,monkeypatch):
    from types import SimpleNamespace
    probe=api();controller=SimpleNamespace(_private_path=lambda value:Path(value).resolve(),
        _supervise=lambda *a,**k:{'status':'FAILED','evidence':'NOT_EVIDENCE'})
    monkeypatch.setattr(probe,'_controller',lambda:controller)
    out=tmp_path/'private-probe';result=probe.run(tmp_path/'reg',tmp_path/'plan',out,'cpu')
    assert result=={'status':'FAILED','evidence':'NOT_EVIDENCE'} and out.is_dir()
    assert probe.run(tmp_path/'reg',tmp_path/'plan',out,'cpu')['status']=='FAILED'

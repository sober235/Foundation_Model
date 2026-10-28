"""A registered expansion is a fresh Stage II, never a relabeled parent or resume."""
from copy import deepcopy
import importlib
import json
from pathlib import Path
import pytest
import torch
from anatobind.aur.data_contract import file_sha256, json_digest
from anatobind.aur.unified_run import UnifiedRun, validate_run_config
from anatobind.aur.unified_train import _state_hash
from test_aur_unified_run import config, factory
from test_aur_unified_ssl_eval import report
from test_aur_unified_launcher import contract


def api():
    return importlib.import_module('anatobind.aur.query_transfer')


def setup_transfer(tmp_path,monkeypatch):
    mod=api()
    parent_root=tmp_path/'parent';child_root=tmp_path/'child'
    for root in (parent_root,child_root):
        for name in ('anatobind/aur/swin.py','anatobind/aur/unified_run.py',
                     'scripts/aur_unified_train.py','scripts/aur_unified_ssl_eval.py','scripts/aur_qualify_training_data.py'):
            p=root/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text('same source')
    (child_root/'anatobind/aur/unified_run.py').write_text('explicit transfer version')
    source=mod.runtime_snapshot(parent_root);target=mod.runtime_snapshot(child_root)
    monkeypatch.setattr(mod,'current_runtime_root',lambda:child_root)
    parent_identity={'contract_sha256':'1'*64,'runtime_sha256':json_digest(source),'geometry_version':'same_geometry'}
    parent_cfg=config();parent_cfg['model']['n_events']=64
    first=UnifiedRun('I',parent_cfg,parent_identity)
    parent=first.export();parent['purpose']='formal';parent['completion_certificate']={'selected_step':2,'selected_checkpoint_sha256':'5'*64}
    gate=report();gate.update(backbone_state_sha256=_state_hash(parent['backbone_state_dict']),checkpoint_step=2,checkpoint_sha256='5'*64)
    export_path=tmp_path/'export.pt';torch.save(parent,export_path)
    gate_path=tmp_path/'gate.json';gate_path.write_text(json.dumps(gate))
    cfg=validate_run_config(config(purpose='resource_probe',model={**parent['model_config'],'n_events':128},accumulation_steps=1))
    identity={**parent_identity,'runtime_sha256':json_digest(target)}
    record={'version':1,'policy':'query_capacity_64_to_128_v1','parent_runtime_root':str(parent_root),
        'parent_sources':source,'child_sources':target,'parent_identity':parent_identity,'child_identity':identity,
        'parent_export':{'path':str(export_path),'sha256':file_sha256(export_path)},
        'g1_report':{'path':str(gate_path),'sha256':file_sha256(gate_path)},'child_config_sha256':json_digest(cfg)}
    import anatobind.aur.unified_artifacts as artifacts
    monkeypatch.setattr(artifacts,'verify_completion_certificate',lambda p,*,data_contract: data_contract==parent_identity)
    return mod,parent,cfg,identity,gate,record


def test_transfer_preserves_parent_and_every_backbone_tensor_with_fresh_optimizer(tmp_path,monkeypatch):
    mod,parent,cfg,identity,gate,record=setup_transfer(tmp_path,monkeypatch)
    before=deepcopy(parent)
    runner=UnifiedRun('II',cfg,identity,initializer=parent,g1_report=gate,initialization_transfer=record)
    assert runner.model.events.M==128 and runner.step_index==0 and not runner.optimizer.state
    assert runner.lineage['kind']=='query_capacity_64_to_128_v1'
    assert runner.data_contract==identity and parent['data_contract']==before['data_contract']
    for name,value in parent['backbone_state_dict'].items():assert torch.equal(runner.model.backbone.state_dict()[name],value)
    assert _state_hash(parent['model_state_dict'])==_state_hash(before['model_state_dict'])
    runner.step(factory);path=runner.save(tmp_path/'step.pt')
    restored=UnifiedRun.resume(path,cfg,identity)
    assert restored.lineage==runner.lineage and restored.step_index==1
    with pytest.raises(ValueError,match='configuration'):
        UnifiedRun.resume(path,{**cfg,'model':{**cfg['model'],'n_events':64}},identity)


@pytest.mark.parametrize('damage',['data','geometry','model','backbone_source','extra_file','parent_bytes','gate_bytes',
                                  'recipe','parent_identity','policy','certificate','gate','weights'])
def test_transfer_rejects_unregistered_changes_before_model_build(tmp_path,monkeypatch,damage):
    mod,parent,cfg,identity,gate,record=setup_transfer(tmp_path,monkeypatch)
    if damage=='data':identity['contract_sha256']='9'*64
    elif damage=='geometry':identity['geometry_version']='other'
    elif damage=='model':cfg['model']['mask_dim']+=1;record['child_config_sha256']=json_digest(cfg)
    elif damage=='backbone_source':
        root=mod.current_runtime_root();(root/'anatobind/aur/swin.py').write_text('changed')
        record['child_sources']=mod.runtime_snapshot(root);identity['runtime_sha256']=json_digest(record['child_sources'])
    elif damage=='extra_file':
        root=mod.current_runtime_root();(root/'anatobind/extra.py').write_text('changed')
        record['child_sources']=mod.runtime_snapshot(root);identity['runtime_sha256']=json_digest(record['child_sources'])
    elif damage=='parent_bytes':Path(record['parent_export']['path']).write_bytes(b'changed')
    elif damage=='gate_bytes':Path(record['g1_report']['path']).write_text('{}')
    elif damage=='recipe':cfg['points']+=1
    elif damage=='parent_identity':record['parent_identity']={**record['parent_identity'],'runtime_sha256':'6'*64}
    elif damage=='policy':record['policy']='anything'
    elif damage=='certificate':
        import anatobind.aur.unified_artifacts as artifacts
        monkeypatch.setattr(artifacts,'verify_completion_certificate',lambda *a,**k:False)
    elif damage=='gate':gate['modes']['2d']['status']='FAIL'
    else:parent['backbone_state_dict'][next(iter(parent['backbone_state_dict']))]+=1
    monkeypatch.setattr(UnifiedRun,'_build',lambda self:pytest.fail('must reject before building'))
    with pytest.raises(ValueError,match='transfer'):
        UnifiedRun('II',cfg,identity,initializer=parent,g1_report=gate,initialization_transfer=record)


def test_default_transition_stays_strict_and_transfer_is_only_for_stage_ii(tmp_path,monkeypatch):
    _,parent,cfg,identity,gate,record=setup_transfer(tmp_path,monkeypatch)
    with pytest.raises(ValueError,match='mismatch'):
        UnifiedRun('II',cfg,identity,initializer=parent)
    with pytest.raises(ValueError,match='transfer'):
        UnifiedRun('III',cfg,identity,initializer=parent,initialization_transfer=record)


def test_resource_probe_execution_cap_preserves_schedule_and_never_issues_formal_completion(tmp_path,monkeypatch):
    from anatobind.aur import unified_launcher as launcher
    monkeypatch.setattr(launcher,'validate_private_output',lambda path:Path(path).resolve())
    monkeypatch.setattr(launcher,'read_contract',lambda path,verify=True:{**contract(),'contract_sha256':'1'*64})
    cfg=config(purpose='resource_probe',steps=12,accumulation_steps=1,validation_every=10,checkpoint_every=5)
    seen=[];train=lambda mode,draw:(seen.append((mode,draw)) or factory(mode,draw))
    out=tmp_path/'probe';result=launcher.run_stage({'schema_version':1,'stage_config':cfg,
        'validation_batches':1},'synthetic-contract.json','I',out,device='cpu',
        train_factory=train,validation_factory=factory,execution_cap=4)
    assert result['status']=='PROBE_COMPLETE' and result['steps']==4 and result['budget_steps']==12
    ck=torch.load(out/'selected_export.pt',map_location='cpu',weights_only=True)
    run_config=json.loads((out/'run_config.json').read_text())
    assert ck['purpose']=='resource_probe' and run_config['recipe']['stage_config']['steps']==12
    assert run_config['execution_cap']==4
    assert ck['step']==4 and not ck['complete']
    assert not ck.get('completion_certificate') and json.loads((out/'completion.json').read_text())['formal_gate_certificate'] is False
    assert [mode for mode,_ in seen]==['2d','3d','2d','3d']


@pytest.mark.parametrize('purpose,cap',[('formal',8),('resource_probe',13),('resource_probe',0),('resource_probe',1.5)])
def test_execution_cap_is_validated_separately_from_scheduler_horizon(tmp_path,purpose,cap):
    from anatobind.aur import unified_launcher as launcher
    cfg=config(purpose=purpose,steps=12,warmup_steps=4,accumulation_steps=1)
    with pytest.raises(ValueError,match='execution cap'):
        launcher.run_stage({'schema_version':1,'stage_config':cfg,'validation_batches':1},
            {**contract(),'contract_sha256':'1'*64},'I',tmp_path/'invalid',device='cpu',
            train_factory=factory,validation_factory=factory,execution_cap=cap)


def test_launcher_commits_exact_transfer_bytes_and_rechecks_at_close(tmp_path,monkeypatch):
    from anatobind.aur import unified_launcher as launcher
    parent=tmp_path/'parent.pt';torch.save({'synthetic':1},parent)
    gate=tmp_path/'gate.json';gate.write_text('{}')
    record={'policy':'example','parent_export':{'path':str(parent),'sha256':file_sha256(parent)},
            'g1_report':{'path':str(gate),'sha256':file_sha256(gate)}}
    path=tmp_path/'transfer.json';path.write_text(json.dumps(record))
    inputs=launcher._inputs(None,str(parent),str(gate),str(path))
    assert inputs['initialization_transfer']=={'path':str(path.resolve()),'sha256':file_sha256(path)}
    launcher._check_inputs(inputs)
    path.write_text('{"policy":"changed"}')
    with pytest.raises(ValueError,match='integrity'):
        launcher._check_inputs(inputs)


def test_launcher_reads_same_verified_transfer_bytes_once(tmp_path,monkeypatch):
    from anatobind.aur import unified_launcher as launcher
    parent=tmp_path/'parent.pt';torch.save({'synthetic':1},parent)
    gate=tmp_path/'gate.json';gate.write_text('{}')
    record={'policy':'example','parent_export':{'path':str(parent),'sha256':file_sha256(parent)},
            'g1_report':{'path':str(gate),'sha256':file_sha256(gate)}}
    path=tmp_path/'transfer.json';path.write_text(json.dumps(record))
    inputs=launcher._inputs(None,str(parent),str(gate),str(path))
    got={}
    def run(*args,**kwargs):got.update(kwargs);return 'synthetic runner'
    monkeypatch.setattr(launcher,'UnifiedRun',run)
    assert launcher._initialize('II',{}, {},inputs,'cpu')=='synthetic runner'
    assert got['initialization_transfer']==record
    path.write_text('{"policy":"changed"}')
    with pytest.raises(ValueError,match='transfer'):
        launcher._initialize('II',{}, {},inputs,'cpu')


def test_launcher_rejects_transfer_ref_different_from_committed_initializer(tmp_path):
    from anatobind.aur import unified_launcher as launcher
    first=tmp_path/'first.pt';second=tmp_path/'second.pt';torch.save({'data':1},first);torch.save({'data':1},second)
    gate=tmp_path/'gate.json';gate.write_text('{}')
    path=tmp_path/'transfer.json';path.write_text(json.dumps({'parent_export':{'path':str(second),'sha256':file_sha256(second)},
        'g1_report':{'path':str(gate),'sha256':file_sha256(gate)}}))
    inputs=launcher._inputs(None,str(first),str(gate),str(path))
    with pytest.raises(ValueError,match='transfer'):
        launcher._initialize('II',{}, {},inputs,'cpu')


def test_launcher_reads_initializer_and_g1_from_same_verified_bytes(tmp_path,monkeypatch):
    from anatobind.aur import unified_launcher as launcher
    first=tmp_path/'first.pt';torch.save({'data':1},first)
    gate=tmp_path/'gate.json';gate.write_text('{}')
    inputs=launcher._inputs(None,str(first),str(gate))
    torch.save({'data':2},first)
    with pytest.raises(ValueError,match='input'):
        launcher._initialize('II',{}, {},inputs,'cpu')


def test_worker_rejects_transfer_swapped_after_registration_before_launch(tmp_path):
    from anatobind.aur import unified_launcher as launcher
    path=tmp_path/'transfer.json';path.write_text('{"policy":"registered"}')
    reference={'path':str(path.resolve()),'sha256':file_sha256(path)}
    path.write_text('{"policy":"unregistered"}')
    with pytest.raises(ValueError,match='transfer'):
        launcher._inputs(None,None,None,reference)


def test_certificate_verifier_hashes_and_loads_one_buffer(tmp_path,monkeypatch):
    from anatobind.aur import unified_artifacts as artifacts
    path=tmp_path/'checkpoint.pt';torch.save({'expected':True},path)
    expected=file_sha256(path);bad=tmp_path/'bad.pt';torch.save({'expected':False},bad)
    monkeypatch.setattr(artifacts,'validate_private_output',lambda x:Path(x))
    # If production performs a hash followed by a second path load, the hash
    # hook substitutes a different checkpoint after returning the old digest.
    def unsafe_hash(p):
        path.write_bytes(bad.read_bytes());return expected
    monkeypatch.setattr(artifacts,'file_sha256',unsafe_hash)
    result=artifacts._checkpoint({'final_checkpoint_path':str(path),
        'final_checkpoint_sha256':expected},'final')
    assert result=={'expected':True}

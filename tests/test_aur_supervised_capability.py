"""Private finite TRAIN capability orchestration regression checks."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import time

import pytest
import torch

from test_aur_unified_run import config, factory, CONTRACT
from test_aur_unified_training import assert_state_equal

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts' / 'aur_supervised_capability.py'


def module():
    spec = importlib.util.spec_from_file_location('capability_controller', SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_parser_redacts_unknown_private_values():
    result = subprocess.run([sys.executable, str(SCRIPT), '--patient-secret-999'], capture_output=True, text=True)
    assert result.returncode != 0
    assert 'patient-secret' not in result.stdout + result.stderr
    assert 'Traceback' not in result.stdout + result.stderr


def test_existing_output_is_untouched_even_with_bad_registration(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, 'home', lambda: tmp_path)
    out = tmp_path / 'owner'
    out.mkdir()
    sentinel = out / 'status.json'
    sentinel.write_text('owner-state')
    before = list(out.iterdir())
    result = module().run_capability(tmp_path / 'private-case-secret.json', out)
    assert result['status'] == 'FAILED'
    assert sentinel.read_text() == 'owner-state'
    assert list(out.iterdir()) == before
    assert 'secret' not in json.dumps(result)


def test_normalized_output_rejects_git_and_external_home_paths(tmp_path):
    mod = module()
    with pytest.raises(ValueError, match='private'):
        mod._private_path('/var/tmp/case-secret')
    with pytest.raises(ValueError, match='private'):
        mod._private_path(SCRIPT.parent.parent / 'secret')


def test_production_recipes_inherit_parent_without_ssl_multiplier():
    mod = module()
    parent = config(points=16384, ssl_2d_lr_scale=5)
    recipes = mod.make_stage_configs(parent)
    assert recipes['II']['lr'] == {'backbone':5e-5,'heads':5e-4}
    assert recipes['III']['lr'] == {'backbone':1e-5,'heads':2.5e-4,'relation':5e-4}
    for cfg in recipes.values():
        assert cfg['steps'] == 512 and cfg['warmup_steps'] == 32
        assert cfg['backbone_warm_steps'] == 0 and cfg['backbone_warm_scale'] == 1
        assert cfg['points'] == 16384 and cfg['model'] == parent['model']
        assert 'ssl_2d_lr_scale' not in cfg
    assert parent['ssl_2d_lr_scale'] == 5


def test_measurement_restores_real_runner_rng_and_training_state():
    from anatobind.aur.unified_run import UnifiedRun
    first = UnifiedRun('I', config(), CONTRACT)
    runner = UnifiedRun('II', config(lr={'backbone':1e-3,'heads':1e-3}), CONTRACT, initializer=first.export())
    panel = {mode:[factory(mode,0)] for mode in ('2d','3d')}
    before_rng, before = runner._rng(), runner.export()
    records = module()._measure(runner, panel)
    assert set(records) == {'2d','3d'}
    assert_state_equal(before_rng, runner._rng())
    assert_state_equal(before, runner.export())
    assert runner.model.training


def test_tiny_real_ii_to_iii_recovers_all_state_and_never_certifies(tmp_path):
    from anatobind.aur.unified_run import UnifiedRun
    mod = module()
    first = UnifiedRun('I', config(), CONTRACT)
    parent = first.export()
    cfgs = {'II':config(purpose='resource_probe',lr={'backbone':1e-3,'heads':1e-3},checkpoint_every=2),
            'III':config(purpose='resource_probe',lr={'backbone':1e-3,'heads':1e-3,'relation':1e-3},checkpoint_every=2)}
    panels = {mode:[factory(mode,0)] for mode in ('2d','3d')}
    out = tmp_path / 'stages'
    out.mkdir()
    final, records, recovery = mod._train_stage('II',cfgs['II'],CONTRACT,parent,None,panels,out,'cpu',boundaries=(0,2,4))
    assert final['stage'] == 'II' and final['complete']
    assert recovery['verified'] is True and recovery['step'] == 2
    final3, records3, recovery3 = mod._train_stage('III',cfgs['III'],CONTRACT,final,None,panels,out,'cpu',boundaries=(0,2,4))
    assert final3['stage'] == 'III' and final3['step'] == 4
    assert recovery3['verified'] is True
    assert set(records) == set(records3) == {0,2,4}
    assert not list(out.rglob('*certificate*')) and not list(out.rglob('*selected_export*'))
    assert all(torch.isfinite(v).all() for v in final3['model_state_dict'].values())


def test_external_watchdog_terminates_blocked_worker(tmp_path):
    mod = module()
    started = time.monotonic()
    result = mod._supervise(lambda send: time.sleep(20), total_seconds=.15, phase_limits={})
    assert result == {'status':'TIMEOUT','evidence':'NOT_EVIDENCE'}
    assert time.monotonic() - started < 3


def registration_fixture(tmp_path):
    from anatobind.aur.unified_launcher import REPO, source_snapshot
    from anatobind.aur.data_contract import json_digest
    mod = module()
    inputs = {}
    for key in ('data_contract','parent_export','parent_completion','g1_report','g1_completion'):
        path = tmp_path / f'{key}.json'
        path.write_text('{}')
        inputs[key] = {'path':str(path.resolve()),'sha256':mod._sha(path)}
    return {'schema_version':1,'runtime_root':str(REPO.resolve()),'runtime_sha256':json_digest(source_snapshot()),
            'inputs':inputs,'controllers':{name:mod._sha(SCRIPT.with_name(name)) for name in
               ('aur_supervised_capability.py','aur_capability_metrics.py')},
            'stage_configs':{'II':{},'III':{}}}


@pytest.mark.parametrize('damage', ['runtime_hash','runtime_root','controller_hash','input_hash'])
def test_source_mismatch_refuses_before_contract_or_model(tmp_path,damage):
    mod = module()
    reg = registration_fixture(tmp_path)
    if damage == 'runtime_hash':
        reg['runtime_sha256'] = '0'*64
    elif damage == 'runtime_root':
        reg['runtime_root'] = str(tmp_path.resolve())
    elif damage == 'controller_hash':
        reg['controllers']['aur_supervised_capability.py'] = '0'*64
    else:
        reg['inputs']['parent_export']['sha256'] = '0'*64
    with pytest.raises(ValueError,match='changed|differs'):
        mod._validate_registration(reg)


def test_per_phase_watchdog_covers_blocked_work():
    mod = module()
    def work(send):
        send({'phase':'scan'})
        time.sleep(20)
    started = time.monotonic()
    result = mod._supervise(work,total_seconds=20,phase_limits={'scan':.15})
    assert result['status'] == 'TIMEOUT'
    assert time.monotonic()-started < 3


def test_gpu_admission_fails_when_visible_uuid_is_unresolved(monkeypatch):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','0')
    with pytest.raises(ValueError,match='GPU'):
        module()._gpu_admission('cuda:0')


def test_parent_completion_file_binds_embedded_certificate_and_selected_export():
    mod = module()
    parent = {'completion_certificate':{'version':1,'stage':'I','selected_step':1792}}
    completion = {'version':1,'stage':'I','selected_step':1792,
                  'selected_export_sha256':'7'*64,'formal_gate_certificate':True}
    assert mod._parent_certificate_matches(parent,completion,'7'*64)
    for key, value in [('selected_step',2048),('selected_export_sha256','8'*64),('formal_gate_certificate',False)]:
        changed = {**completion,key:value}
        assert not mod._parent_certificate_matches(parent,changed,'7'*64)
    assert not mod._parent_certificate_matches({'completion_certificate':{}},completion,'7'*64)


def synthetic_rows(tmp_path):
    import nibabel as nib
    import numpy as np
    rows = []
    for index, source in enumerate(('pdgm','bmsr','sibbms')):
        image, anatomy = tmp_path/f'image{index}.nii.gz', tmp_path/f'anatomy{index}.nii.gz'
        shape=(20,20,8)
        for path, values in ((image,np.arange(3200,dtype=np.float32).reshape(shape)),
                             (anatomy,np.full(shape,2,dtype=np.int16))):
            obj = nib.Nifti1Image(values,np.eye(4))
            obj.header.set_xyzt_units('mm')
            nib.save(obj,path)
        rows.append({'image':str(image),'anatomy':str(anatomy),'lesion':None,'source':source,
                     'patient':f'synthetic{index}','case':'synthetic','sequence':'T1',
                     'u_supervised':False,'a_supervised':True,'r_supervised':False,
                     'role':'train_supervised','split':'train'})
    return rows


def test_real_qualified_batch_metadata_freezes_with_safe_tensor_loading(tmp_path):
    from anatobind.aur.unified_views import QualifiedViews
    mod = module()
    views = QualifiedViews(synthetic_rows(tmp_path),'II',config(),augment=False,validation=False)
    original = views('2d',0)
    portable = mod._portable_batch(original)
    path = tmp_path/'batch.pt'
    torch.save(portable,path)
    frozen = torch.load(path,weights_only=True)
    assert torch.equal(frozen['source_affine'][0],torch.eye(4,dtype=torch.float64))
    assert torch.equal(frozen['entity'],original['entity'])
    assert mod._state_digest(frozen) == mod._state_digest(portable)


def test_missing_support_stops_real_train_pool_without_model(tmp_path):
    mod = module()
    frozen = {'train_supervised':synthetic_rows(tmp_path)}
    out = tmp_path/'pool-run'
    out.mkdir()
    panels, support = mod._freeze_panels({'stage_configs':{'II':config()}},frozen,out)
    assert panels is None and support['2d']['status'] == 'INSUFFICIENT'
    assert support['2d']['a_truth'][0] > 0
    assert sum(n>0 for n in support['2d']['a_truth']) == 1
    assert len(list((out/'pool').glob('2d_*.pt'))) == 64
    assert len(list((out/'pool').glob('3d_*.pt'))) == 64
    assert support['3d']['status'] == 'INSUFFICIENT'
    assert not (out/'II').exists() and not (out/'III').exists()


def test_non_train_rows_rejected_before_pool_creation(tmp_path):
    mod = module()
    rows = synthetic_rows(tmp_path)
    rows[0].update(role='development_validation',split='val')
    out = tmp_path/'bad-role'
    out.mkdir()
    with pytest.raises(ValueError,match='role|partition'):
        mod._freeze_panels({'stage_configs':{'II':config()}},{'train_supervised':rows},out)
    assert not (out/'pool').exists()


def test_g1_checkpoint_provenance_accepts_certified_checkpoint_or_export():
    mod=module()
    certificate={'selected_step':1792,'selected_checkpoint_sha256':'4'*64}
    assert mod._g1_parent_matches({'checkpoint_step':1792,'checkpoint_sha256':'4'*64},certificate,'5'*64)
    assert mod._g1_parent_matches({'checkpoint_step':1792,'checkpoint_sha256':'5'*64},certificate,'5'*64)
    assert not mod._g1_parent_matches({'checkpoint_step':2048,'checkpoint_sha256':'4'*64},certificate,'5'*64)
    assert not mod._g1_parent_matches({'checkpoint_step':1792,'checkpoint_sha256':'6'*64},certificate,'5'*64)


def test_frozen_batch_hash_refuses_modified_cache_before_use(tmp_path):
    mod=module()
    path=tmp_path/'batch.pt'
    original=factory('2d',0)
    torch.save(original,path)
    commitment={'file_sha256':mod._sha(path),'tensor_commitment':mod._state_digest(original)}
    assert torch.equal(mod._load_frozen_batch(path,commitment)['entity'],original['entity'])
    original['entity'].zero_()
    torch.save(original,path)
    with pytest.raises(ValueError,match='changed'):
        mod._load_frozen_batch(path,commitment)


def test_checkpoint_commitment_distinguishes_nested_cursor_structures():
    mod=module()
    assert mod._state_digest([[1],2]) != mod._state_digest([[1,2]])


def test_worker_dies_when_watchdog_parent_is_killed(tmp_path):
    import os
    code = '''import importlib.util, os, pathlib, time
spec=importlib.util.spec_from_file_location('controller', SCRIPT)
mod=importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
def blocked(send):
    pathlib.Path(PIDFILE).write_text(str(os.getpid()))
    time.sleep(30)
mod._supervise(blocked,total_seconds=30,phase_limits={})
'''.replace('SCRIPT',repr(str(SCRIPT))).replace('PIDFILE',repr(str(tmp_path/'worker.pid')))
    parent=subprocess.Popen([sys.executable,'-c',code],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    worker_pid=None
    try:
        deadline=time.monotonic()+3
        while time.monotonic()<deadline and not (tmp_path/'worker.pid').exists():
            time.sleep(.02)
        assert (tmp_path/'worker.pid').exists()
        worker_pid=int((tmp_path/'worker.pid').read_text())
        parent.kill()
        parent.wait(timeout=2)
        deadline=time.monotonic()+3
        while time.monotonic()<deadline:
            state=Path(f'/proc/{worker_pid}/stat')
            if not state.exists() or state.read_text().split()[2] == 'Z':
                break
            time.sleep(.02)
        else:
            pytest.fail('worker remained alive after watchdog parent death')
    finally:
        if parent.poll() is None:
            parent.kill()
            parent.wait(timeout=2)
        if worker_pid is not None:
            try:
                os.kill(worker_pid,9)
            except ProcessLookupError:
                pass


def test_retention_refusal_preserves_already_measured_private_evidence(tmp_path):
    from anatobind.aur.unified_run import UnifiedRun
    mod=module()
    parent=UnifiedRun('I',config(),CONTRACT).export()
    panels={mode:[factory(mode,0)] for mode in ('2d','3d')}
    def refuse(stage,step,records):
        raise mod._Insufficient()
    with pytest.raises(mod._Insufficient):
        mod._train_stage('II',config(),CONTRACT,parent,None,panels,tmp_path,'cpu',on_measure=refuse)
    assert (tmp_path/'II'/'measure_0000.pt').is_file()
    assert not list((tmp_path/'II').glob('checkpoint_*.pt'))


@pytest.mark.parametrize('changes',[{'batch_size':2},{'accumulation_steps':2},{'ssl_2d_lr_scale':1},
                                    {'points':32},{'model':{}}])
def test_parent_protocol_refuses_incompatible_supervised_recipe(changes):
    mod=module()
    parent_cfg=config(points=16384,batch_size=1,accumulation_steps=1,ssl_2d_lr_scale=5,
                      crops={'2d':[1,160,160],'3d':[128,160,160]})
    parent={'model_config':deepcopy(parent_cfg['model'])}
    mod._validate_parent_protocol(parent_cfg,parent)
    with pytest.raises(ValueError,match='LR5'):
        mod._validate_parent_protocol({**parent_cfg,**changes},parent)

"""Formal full-pool continuation must retain admission and finite stage boundaries."""
import importlib.util
import json
from pathlib import Path

import pytest
import torch

from test_aur_unified_run import config, factory
from test_aur_unified_launcher import contract

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/aur_formal_supervised.py'


def module():
    assert SCRIPT.exists(), 'The external formal continuation controller is missing'
    spec = importlib.util.spec_from_file_location('formal_supervised_external', SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def parent_config():
    from anatobind.aur.unified_run import validate_run_config
    return validate_run_config(config(purpose='formal', seed=20261019, points=16384,
        accumulation_steps=1, crops={'2d':[1,160,160], '3d':[128,160,160]}, ssl_2d_lr_scale=5))


def registration(tmp_path):
    from anatobind.aur.unified_launcher import REPO, source_snapshot
    mod = module()
    inputs = {}
    for key in mod.INPUTS:
        path = tmp_path / (key + '.json')
        path.write_text('{}')
        inputs[key] = {'path':str(path.resolve()), 'sha256':mod._sha(path)}
    return {'schema_version':1, 'runtime_root':str(REPO.resolve()),
        'runtime_sha256':mod._digest(source_snapshot()), 'inputs':inputs,
        'controllers':{name:mod._sha(SCRIPT.with_name(name)) for name in mod.CONTROLLERS},
        'stage_configs':mod.make_stage_configs(parent_config()), 'validation_batches':8}


def test_formal_recipes_are_fixed_and_remove_ssl_multiplier():
    mod = module()
    parent = parent_config()
    cfgs = mod.make_stage_configs(parent)
    assert parent['ssl_2d_lr_scale'] == 5
    assert cfgs['II']['steps'] == 4096 and cfgs['III']['steps'] == 2048
    assert cfgs['II']['warmup_steps'] == 128 and cfgs['III']['warmup_steps'] == 64
    assert cfgs['II']['max_seconds'] == 14400 and cfgs['III']['max_seconds'] == 7200
    assert cfgs['II']['lr'] == {'backbone':5e-5, 'heads':5e-4}
    assert cfgs['III']['lr'] == {'backbone':1e-5, 'heads':2.5e-4, 'relation':5e-4}
    for cfg in cfgs.values():
        assert cfg['purpose'] == 'formal' and 'ssl_2d_lr_scale' not in cfg
        assert cfg['validation_every'] == 512 and cfg['checkpoint_every'] == 256
        assert cfg['backbone_warm_steps'] == 0 and cfg['backbone_warm_scale'] == 1
        assert cfg['model'] == parent['model'] and cfg['mode_cycle'] == ['2d','3d']


@pytest.mark.parametrize('damage', ['input', 'runtime_hash', 'runtime_root', 'controller',
    'purpose', 'ssl_multiplier', 'steps', 'warmup', 'validation_batches', 'missing_recipe'])
def test_registration_refuses_changed_or_incomplete_protocol_before_data(tmp_path, damage):
    mod = module()
    reg = registration(tmp_path)
    if damage == 'input': reg['inputs']['parent_export']['sha256'] = '0'*64
    elif damage == 'runtime_hash': reg['runtime_sha256'] = '0'*64
    elif damage == 'runtime_root': reg['runtime_root'] = str(tmp_path)
    elif damage == 'controller': reg['controllers']['aur_supervised_capability.py'] = '0'*64
    elif damage == 'purpose': reg['stage_configs']['II']['purpose'] = 'resource_probe'
    elif damage == 'ssl_multiplier': reg['stage_configs']['II']['ssl_2d_lr_scale'] = 5
    elif damage == 'steps': reg['stage_configs']['III']['steps'] = 4096
    elif damage == 'warmup': reg['stage_configs']['III']['warmup_steps'] = 128
    elif damage == 'validation_batches': reg['validation_batches'] = 1
    else: reg['stage_configs']['II'].pop('points')
    with pytest.raises(ValueError): mod._validate_registration(reg)


def test_registration_accepts_exact_runtime_and_canonical_recipe(tmp_path):
    mod = module()
    assert mod._digest(mod._validate_registration(registration(tmp_path))) == registration(tmp_path)['runtime_sha256']


def test_parser_and_bad_registration_do_not_echo_private_values(tmp_path, monkeypatch, capsys):
    mod = module()
    with pytest.raises(SystemExit): mod.main(['--patient-secret-999'])
    assert 'patient-secret' not in capsys.readouterr().err
    monkeypatch.setattr(Path, 'home', lambda:tmp_path)
    result = mod.run_formal(tmp_path/'private-patient.json', tmp_path/'out')
    assert result['status'] == 'FAILED' and 'private-patient' not in json.dumps(result)
    assert json.loads((tmp_path/'out/terminal.json').read_text()) == result


def test_main_prints_only_anonymous_status_not_private_artifact_paths(monkeypatch,capsys):
    mod=module()
    monkeypatch.setattr(mod,'run_formal',lambda *a:{'status':'COMPLETE','evidence':'NOT_EVIDENCE',
        'stages':{'II':{'selected_export':'/home/private-patient/selected_export.pt','steps':4096}}})
    assert mod.main(['--registration','private-registration','--out','private-output'])==0
    printed=capsys.readouterr().out
    assert 'private-patient' not in printed and 'selected_export' not in printed
    assert json.loads(printed)=={'status':'COMPLETE','evidence':'NOT_EVIDENCE'}


def test_existing_output_is_untouched(tmp_path, monkeypatch):
    mod = module()
    monkeypatch.setattr(Path, 'home', lambda:tmp_path)
    out = tmp_path/'owned';out.mkdir();(out/'status.json').write_text('existing')
    assert mod.run_formal(tmp_path/'missing',out)['status'] == 'FAILED'
    assert list(out.iterdir()) == [out/'status.json']
    assert (out/'status.json').read_text() == 'existing'


def test_output_normalizes_and_rejects_git_or_home_escape(tmp_path, monkeypatch):
    mod = module(); monkeypatch.setattr(Path, 'home', lambda:tmp_path)
    assert mod._private_path(tmp_path/'unused/../safe') == tmp_path/'safe'
    with pytest.raises(ValueError):mod._private_path(tmp_path.parent/'outside')
    repo=tmp_path/'repo';repo.mkdir();(repo/'.git').write_text('gitdir: private')
    with pytest.raises(ValueError):mod._private_path(repo/'out')


@pytest.mark.parametrize('status', ['FAILED','TIMEOUT','INTERRUPTED'])
def test_stage_ii_failure_prevents_stage_iii(tmp_path,monkeypatch,status):
    mod=module(); calls=[]
    monkeypatch.setattr(mod,'_run_stage_job',lambda stage,*a,**k:calls.append(stage) or {'status':status})
    monkeypatch.setattr(mod,'_gpu_admission',lambda device:{'device':device})
    result=mod._execute_stages(registration(tmp_path),{},tmp_path,'cpu',lambda _:None)
    assert result['status']==status and calls==['II']


def test_stage_iii_receives_only_certified_formal_ii_selected_export(tmp_path,monkeypatch):
    mod=module(); calls=[];checks=[];guards=[]
    def run(stage,recipe,data,out,device,initializer,g1_report):
        calls.append((stage,initializer,g1_report,recipe))
        return {'status':'COMPLETE','steps':recipe['stage_config']['steps']}
    def verify(stage,out,cfg,identity):
        checks.append(stage)
        return {'selected_export':str(out/'selected_export.pt'),'selected_export_sha256':'a'*64,
                'completion_sha256':'b'*64,'steps':cfg['steps'],'selected_step':512}
    monkeypatch.setattr(mod,'_run_stage_job',run)
    monkeypatch.setattr(mod,'_validate_completed_stage',verify)
    monkeypatch.setattr(mod,'_gpu_admission',lambda device:guards.append(device) or {'device':device})
    reg=registration(tmp_path)
    result=mod._execute_stages(reg,{},tmp_path,'cpu',lambda _:None)
    assert result['status']=='COMPLETE' and checks==['II','III'] and len(guards)==2
    assert calls[0][1]==reg['inputs']['parent_export']['path']
    assert calls[0][2]==reg['inputs']['g1_report']['path']
    assert calls[1][1]==str(tmp_path/'II/selected_export.pt') and calls[1][2] is None
    assert all(c[3]['validation_batches']==8 for c in calls)


def test_changed_stage_certificate_prevents_iii(tmp_path,monkeypatch):
    mod=module();calls=[]
    monkeypatch.setattr(mod,'_run_stage_job',lambda stage,*a,**k:calls.append(stage) or {'status':'COMPLETE'})
    monkeypatch.setattr(mod,'_gpu_admission',lambda device:{'device':device})
    monkeypatch.setattr(mod,'_validate_completed_stage',lambda *a:(_ for _ in ()).throw(ValueError('invalid certificate')))
    with pytest.raises(ValueError):mod._execute_stages(registration(tmp_path),{},tmp_path,'cpu',lambda _:None)
    assert calls==['II']


def test_finite_watchdog_terminal_is_written_only_after_supervisor_returns(tmp_path,monkeypatch):
    mod=module();monkeypatch.setattr(Path,'home',lambda:tmp_path)
    def supervised(work,**kwargs):
        assert not (tmp_path/'run/terminal.json').exists()
        assert kwargs['total_seconds']==28800
        assert kwargs['phase_limits']['II']==16200 and kwargs['phase_limits']['III']==9000
        return {'status':'TIMEOUT','evidence':'NOT_EVIDENCE'}
    monkeypatch.setattr(mod,'_supervise',supervised)
    assert mod.run_formal(tmp_path/'registration',tmp_path/'run')['status']=='TIMEOUT'
    assert (tmp_path/'run/terminal.json').exists()


def test_stage_worker_uses_default_fullpool_factories(tmp_path,monkeypatch):
    import anatobind.aur.unified_launcher as launcher
    mod=module();seen=[]
    def run(*args,**kwargs):
        seen.append((args,kwargs));return {'status':'INTERRUPTED','steps':2}
    monkeypatch.setattr(launcher,'run_stage',run)
    result=mod._stage_work('II',{'schema_version':1},'contract',tmp_path,'cpu','parent','gate')
    assert result['status']=='INTERRUPTED'
    assert not {'train_factory','validation_factory','resume'} & set(seen[0][1])


def test_tiny_cpu_actual_stage_chain_uses_real_run_stage_without_formal_claim(tmp_path,monkeypatch):
    import os
    import tempfile
    import nibabel as nib
    import numpy as np
    import anatobind.aur.unified_launcher as launcher
    from anatobind.aur.unified_run import UnifiedRun
    mod=module();monkeypatch.setattr(mod,'PHASE_LIMITS',{'II':20,'III':20})
    # The regression includes a previously used Torch thread pool. Each stage
    # must exec a fresh interpreter and use real default QualifiedViews.
    torch.set_num_threads(2)
    large=torch.rand(256,256);large.copy_(large+1)
    root=Path.home()/'outputs/aur_lr5_formal_supervised_20261026/implementation'
    root.mkdir(parents=True,exist_ok=True,mode=0o700)
    with tempfile.TemporaryDirectory(prefix='synthetic-stage-',dir=root) as directory:
        private=Path(directory);frozen=contract()
        image=private/'image.nii.gz';anatomy=private/'anatomy.nii.gz';lesion=private/'lesion.nii.gz'
        shape=(16,16,8)
        labels=np.zeros(shape,np.int16);labels[5:9,5:9,2:6]=1
        for path,values in ((image,np.arange(np.prod(shape),dtype=np.float32).reshape(shape)),
                            (anatomy,np.full(shape,2,np.int16)),(lesion,labels)):
            obj=nib.Nifti1Image(values,np.eye(4));obj.header.set_xyzt_units('mm');nib.save(obj,path)
        for role in ('train_supervised','validation'):
            frozen[role][0].update(case='synthetic',sequence='T1',image=str(image),anatomy=str(anatomy),
                lesion=str(lesion),a_supervised=True,u_supervised=True,r_supervised=True,
                u_values=[1],a_ignore_values=[1])
        identity={'contract_sha256':frozen['contract_sha256'],'runtime_sha256':mod._digest(launcher.source_snapshot()),
                  'geometry_version':launcher.GEOMETRY_VERSION}
        first=UnifiedRun('I',config(),identity);parent=private/'parent.pt';torch.save(first.export(),parent)
        workers=[]
        for stage in ('II','III'):
            rates={'backbone':1e-3,'heads':1e-3}
            if stage=='III':rates['relation']=1e-3
            recipe={'schema_version':1,'stage_config':config(lr=rates),'validation_batches':1}
            result=mod._run_stage_job(stage,recipe,frozen,private/stage,'cpu',str(parent),None)
            assert result['status']=='COMPLETE' and result['steps']==4, (private/f'{stage}_worker.log').read_text()
            workers.append(result['worker_pid'])
            with pytest.raises(ProcessLookupError):os.kill(result['worker_pid'],0)
            parent=private/stage/'selected_export.pt';exported=torch.load(parent,weights_only=True)
            assert exported['stage']==stage and exported['purpose']=='synthetic_test'
            assert 'completion_certificate' not in exported
            assert len(launcher.read_jsonl(private/stage/'steps.jsonl'))==4
            with pytest.raises(ValueError,match='completion'):
                mod._validate_completed_stage(stage,private/stage,recipe['stage_config'],identity)
        assert workers[0]!=workers[1] and os.getpid() not in workers


def admission_fixture(tmp_path,monkeypatch):
    import anatobind.aur.data_contract as data
    import anatobind.aur.unified_artifacts as artifacts
    from anatobind.aur.unified_train import _state_hash
    from test_aur_unified_ssl_eval import report
    mod=module();reg=registration(tmp_path)
    identity={'contract_sha256':contract()['contract_sha256'],'runtime_sha256':reg['runtime_sha256']}
    final=tmp_path/'final.pt';torch.save({'cfg':parent_config()},final)
    certificate={'selected_step':2,'selected_checkpoint_sha256':'c'*64,'final_checkpoint_path':str(final)}
    parent={'stage':'I','data_contract':identity,'completion_certificate':certificate,
            'model_config':parent_config()['model'],'backbone_state_dict':{'weight':torch.ones(2)}}
    torch.save(parent,reg['inputs']['parent_export']['path'])
    gate=report();gate.update(backbone_state_sha256=_state_hash(parent['backbone_state_dict']),
        checkpoint_step=2,checkpoint_sha256='c'*64)
    Path(reg['inputs']['g1_report']['path']).write_text(json.dumps(gate))
    export_sha=mod._sha(reg['inputs']['parent_export']['path'])
    completion={**certificate,'selected_export_sha256':export_sha,'formal_gate_certificate':True}
    Path(reg['inputs']['parent_completion']['path']).write_text(json.dumps(completion))
    g1complete={'report_sha256':mod._sha(reg['inputs']['g1_report']['path']),
                'pass':True,'frozen_backbones_unchanged':True}
    Path(reg['inputs']['g1_completion']['path']).write_text(json.dumps(g1complete))
    for item in reg['inputs'].values():item['sha256']=mod._sha(item['path'])
    reads=[]
    def read(path,verify):reads.append(verify);return contract()
    monkeypatch.setattr(data,'read_contract',read)
    monkeypatch.setattr(artifacts,'verify_completion_certificate',lambda *a,**k:True)
    return mod,reg,reads


def test_admission_requires_complete_live_data_verifier_and_default_g1(tmp_path,monkeypatch):
    mod,reg,reads=admission_fixture(tmp_path,monkeypatch)
    assert mod._admit(reg)[2]['runtime_sha256']==reg['runtime_sha256']
    assert reads==[True]


@pytest.mark.parametrize('damage',['certificate','completion','g1_checkpoint','g1_mode','g1_default',
                                  'g1_completion','parent_runtime','parent_recipe'])
def test_admission_rejects_parent_or_g1_mismatch(tmp_path,monkeypatch,damage):
    import anatobind.aur.unified_artifacts as artifacts
    mod,reg,reads=admission_fixture(tmp_path,monkeypatch)
    if damage=='certificate':
        monkeypatch.setattr(artifacts,'verify_completion_certificate',lambda *a,**k:False)
    elif damage=='completion':
        path=Path(reg['inputs']['parent_completion']['path']);value=json.loads(path.read_text())
        value['selected_step']=3;path.write_text(json.dumps(value))
    elif damage.startswith('g1_'):
        path=Path(reg['inputs']['g1_completion' if damage=='g1_completion' else 'g1_report']['path'])
        value=json.loads(path.read_text())
        if damage=='g1_checkpoint':value['checkpoint_step']=3
        elif damage=='g1_mode':value['modes']['2d']['status']='FAIL'
        elif damage=='g1_default':value['default_recipe_match']=False
        else:value['pass']=False
        path.write_text(json.dumps(value))
    elif damage=='parent_runtime':
        path=Path(reg['inputs']['parent_export']['path']);value=torch.load(path,weights_only=True)
        value['data_contract']['runtime_sha256']='9'*64;torch.save(value,path)
    else:
        reg['stage_configs']['II']['weight_decay']=.2
    for item in reg['inputs'].values():item['sha256']=mod._sha(item['path'])
    with pytest.raises(ValueError):mod._admit(reg)
    assert reads==[True]


def test_registered_disk_source_cannot_replace_already_loaded_helper_bytes(tmp_path,monkeypatch):
    mod=module();reg=registration(tmp_path);original=mod._sha
    # Represent changed disk bytes and a newly updated registration, while the
    # module has already executed the earlier helper implementation.
    reg['controllers']['aur_supervised_capability.py']='a'*64
    monkeypatch.setattr(mod,'_sha',lambda p:'a'*64 if Path(p).name=='aur_supervised_capability.py' else original(p))
    with pytest.raises(ValueError,match='source|loaded'):mod._validate_registration(reg)


@pytest.mark.parametrize('busy',[False,True])
def test_gpu_guard_runs_after_startup_hashing_immediately_before_initialize(tmp_path,monkeypatch,busy):
    import anatobind.aur.unified_launcher as launcher
    mod=module();events=[]
    def initialize(*args,**kwargs):events.append('initialize');return {'status':'COMPLETE'}
    def run(*args,**kwargs):
        events.append('startup_hashes_checked')
        return launcher._initialize()
    def guard(device):
        events.append('gpu_checked')
        if busy:raise ValueError('GPU no longer idle')
        return {'device':device}
    monkeypatch.setattr(launcher,'_initialize',initialize)
    monkeypatch.setattr(launcher,'run_stage',run)
    monkeypatch.setattr(mod,'_gpu_admission',guard)
    if busy:
        with pytest.raises(ValueError,match='GPU'):mod._stage_work('II',{},'contract',tmp_path,'cpu','parent','gate')
        assert events==['startup_hashes_checked','gpu_checked']
    else:
        mod._stage_work('II',{},'contract',tmp_path,'cpu','parent','gate')
        assert events==['startup_hashes_checked','gpu_checked','initialize']
        assert (tmp_path/'gpu_initialization_admission.json').exists()
    assert launcher._initialize is initialize


def test_fresh_stage_timeout_waits_for_owned_child_exit(monkeypatch):
    import tempfile
    mod=module();monkeypatch.setattr(mod,'PHASE_LIMITS',{'II':.05,'III':.05})
    root=Path.home()/'outputs/aur_lr5_formal_supervised_20261026/implementation'
    original=mod.subprocess.Popen;children=[]
    def launch(*args,**kwargs):
        process=original(*args,**kwargs);children.append(process);return process
    monkeypatch.setattr(mod.subprocess,'Popen',launch)
    with tempfile.TemporaryDirectory(prefix='synthetic-timeout-',dir=root) as directory:
        result=mod._run_stage_job('II',{}, {},Path(directory)/'II','cpu',None,None)
        assert result['status']=='TIMEOUT' and len(children)==1
        assert children[0].poll() is not None


def test_fresh_stage_worker_dies_with_its_orchestrator():
    import hashlib
    import os
    import signal
    import subprocess
    import sys
    import tempfile
    import time
    mod=module();root=Path.home()/'outputs/aur_lr5_formal_supervised_20261026/implementation'
    with tempfile.TemporaryDirectory(prefix='synthetic-parent-death-',dir=root) as directory:
        private=Path(directory);request=private/'II_worker_request.json';marker=private/'worker.pid'
        value=dict(stage='II',recipe={},data_contract={},out=str(private/'II'),device='cpu',initializer=None,g1_report=None,
            controllers={name:hashlib.sha256(data).hexdigest() for name,data in mod._LOADED_SOURCES.items()})
        mod._json(request,value)
        worker_code='''import importlib.util,os,pathlib,sys,time
s=importlib.util.spec_from_file_location('external_formal',sys.argv[1]);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
def blocked(*args):
    pathlib.Path(sys.argv[5]).write_text(str(os.getpid()))
    time.sleep(30)
    return {'status':'COMPLETE'}
m._stage_work=blocked
raise SystemExit(m._stage_subprocess_main(sys.argv[2],sys.argv[3],int(sys.argv[4])))
'''
        parent_code='''import os,subprocess,sys
p=subprocess.Popen([sys.executable,'-c',sys.argv[1],sys.argv[2],sys.argv[3],sys.argv[4],str(os.getpid()),sys.argv[5]])
p.wait()
'''
        parent=subprocess.Popen([sys.executable,'-c',parent_code,worker_code,str(SCRIPT),str(request),mod._sha(request),str(marker)],
                                stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
        worker_pid=None
        def alive(pid):
            try:return Path(f'/proc/{pid}/stat').read_text().split()[2]!='Z'
            except FileNotFoundError:return False
        try:
            deadline=time.monotonic()+5
            while not marker.exists() and time.monotonic()<deadline:time.sleep(.02)
            assert marker.exists()
            worker_pid=int(marker.read_text());assert alive(worker_pid)
            parent.kill();parent.wait(timeout=2)
            deadline=time.monotonic()+3
            while alive(worker_pid) and time.monotonic()<deadline:time.sleep(.02)
            assert not alive(worker_pid)
        finally:
            if parent.poll() is None:parent.kill();parent.wait(timeout=2)
            if worker_pid is not None and alive(worker_pid):os.kill(worker_pid,signal.SIGKILL)


def test_stage_request_hash_and_parse_use_the_same_bytes(tmp_path,monkeypatch):
    import ctypes
    import hashlib
    import os
    from contextlib import contextmanager
    from types import SimpleNamespace
    mod=module();path=tmp_path/'II_worker_request.json'
    request=dict(stage='II',recipe={'marker':'committed'},data_contract={},out=str(tmp_path/'II'),
        device='cpu',initializer=None,g1_report=None,
        controllers={name:hashlib.sha256(data).hexdigest() for name,data in mod._LOADED_SOURCES.items()})
    mod._json(path,request);expected_sha=mod._sha(path)
    altered={**request,'recipe':{'marker':'uncommitted'}}
    changed=[];consumed=[];original_open=Path.open
    @contextmanager
    def swapping_open(member,*args,**kwargs):
        mode=args[0] if args else kwargs.get('mode','r')
        with original_open(member,*args,**kwargs) as stream:
            yield stream
        if member==path and mode=='rb' and not changed:
            changed.append(True)
            with original_open(path,'wb') as stream:stream.write(json.dumps(altered).encode())
    monkeypatch.setattr(Path,'open',swapping_open)
    monkeypatch.setattr(mod,'_private_path',lambda p:Path(p).resolve())
    monkeypatch.setattr(ctypes,'CDLL',lambda *a,**k:SimpleNamespace(prctl=lambda *a:0))
    monkeypatch.setattr(mod,'_stage_work',lambda stage,recipe,*args:consumed.append(recipe) or {'status':'COMPLETE'})
    assert mod._stage_subprocess_main(str(path),expected_sha,os.getppid())==0
    assert changed==[True] and json.loads(path.read_text())==altered
    assert consumed==[{'marker':'committed'}]


def test_parent_registration_hash_binds_parsed_bytes_and_closing_rejects_replacement(tmp_path,monkeypatch):
    from contextlib import contextmanager
    mod=module();reg=registration(tmp_path);path=tmp_path/'registration.json'
    path.write_text(json.dumps(reg));original_sha=mod._sha(path)
    changed={**reg,'schema_version':2};swapped=[];original_open=Path.open
    @contextmanager
    def swapping_open(member,*args,**kwargs):
        mode=args[0] if args else kwargs.get('mode','r')
        with original_open(member,*args,**kwargs) as stream:yield stream
        if member==path and mode in ('r','rb') and not swapped:
            swapped.append(True)
            with original_open(path,'w') as stream:json.dump(changed,stream)
    monkeypatch.setattr(Path,'open',swapping_open)
    monkeypatch.setattr(mod,'_admit',lambda value:({}, {'contract_sha256':'1'*64,'summary':{}}, {}))
    monkeypatch.setattr(mod,'_execute_stages',lambda *a:{'status':'INTERRUPTED','evidence':'NOT_EVIDENCE'})
    out=tmp_path/'out';out.mkdir()
    with pytest.raises(ValueError,match='closing input integrity'):
        mod._execute(path,out,'cpu',lambda _:None)
    assert swapped==[True]
    committed=json.loads((out/'input_commitments.json').read_text())
    assert committed['registration_sha256']==original_sha
    assert not (out/'result.json').exists()

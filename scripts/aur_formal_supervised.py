"""Bounded formal II/III continuation in an explicitly registered frozen runtime.

This external controller never adds its checkout to the runtime import path.
Each stage owns a separate finite worker so its CUDA context is gone before the
next strict idle-GPU admission. Outputs and terminal records are immutable.
"""
import argparse
from copy import deepcopy
import gc
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import subprocess


# Capture the controller source while its module body is executing. Comparing
# code objects also rejects a file changed between Python's load and this read.
_SELF_BYTES = Path(__file__).read_bytes()
if compile(_SELF_BYTES,__file__,'exec',dont_inherit=True)!=sys._getframe().f_code:
    raise ValueError('loaded controller source differs from disk')


def _helpers():
    path=Path(__file__).with_name('aur_supervised_capability.py')
    content=path.read_bytes()
    spec = importlib.util.spec_from_file_location('aur_formal_capability_helpers',
        path)
    module = importlib.util.module_from_spec(spec)
    exec(compile(content,str(path),'exec',dont_inherit=True),module.__dict__)
    return module,content


_base,_HELPER_BYTES = _helpers()
_LOADED_SOURCES={'aur_formal_supervised.py':_SELF_BYTES,'aur_supervised_capability.py':_HELPER_BYTES}
_sha, _digest, _read, _json = _base._sha, _base._digest, _base._read, _base._json
_private_path, _supervise = _base._private_path, _base._supervise
_gpu_admission = _base._gpu_admission
_parent_certificate_matches = _base._parent_certificate_matches
_g1_parent_matches = _base._g1_parent_matches
FAILED = {'status':'FAILED', 'evidence':'NOT_EVIDENCE'}
INPUTS = ('data_contract','parent_export','parent_completion','g1_report','g1_completion')
CONTROLLERS = ('aur_formal_supervised.py','aur_supervised_capability.py')
TOTAL_SECONDS = 28800
PHASE_LIMITS = {'admission':1800, 'II':16200, 'III':9000, 'close':1800}


def make_stage_configs(parent_cfg):
    result = {}
    for stage, steps, warmup, seconds, rates in (
        ('II',4096,128,14400,{'backbone':5e-5,'heads':5e-4}),
        ('III',2048,64,7200,{'backbone':1e-5,'heads':2.5e-4,'relation':5e-4})):
        cfg = deepcopy(parent_cfg)
        cfg.pop('ssl_2d_lr_scale',None)
        cfg.update(purpose='formal',steps=steps,lr=rates,warmup_steps=warmup,
                   backbone_warm_steps=0,backbone_warm_scale=1,
                   validation_every=512,checkpoint_every=256,max_seconds=seconds)
        result[stage] = cfg
    return result


def _validate_registration(reg):
    from anatobind.aur.unified_launcher import REPO, source_snapshot
    from anatobind.aur.unified_run import validate_run_config
    expected = {'schema_version','runtime_root','runtime_sha256','inputs','controllers',
                'stage_configs','validation_batches'}
    if (set(reg) not in (expected,expected|{'initialization_transfer'}) or reg['schema_version']!=1 or type(reg['validation_batches']) is not int
            or reg['validation_batches']!=8 or set(reg['inputs'])!=set(INPUTS)
            or set(reg['controllers'])!=set(CONTROLLERS) or set(reg['stage_configs'])!={'II','III'}):
        raise ValueError('invalid registration')
    transfer=reg.get('initialization_transfer')
    if transfer is not None:
        if (set(transfer)!={'path','sha256'} or str(Path(transfer['path']).expanduser().resolve())!=transfer['path']
                or _sha(transfer['path'])!=transfer['sha256']):
            raise ValueError('invalid transfer registration')
    if any((cfg['model']['n_events'] != 128 if transfer is not None else cfg['model']['n_events'] > 64)
           for cfg in reg['stage_configs'].values()):
        raise ValueError('invalid registration transfer capacity')
    for item in reg['inputs'].values():
        if set(item)!={'path','sha256'} or str(Path(item['path']).expanduser().resolve())!=item['path']:
            raise ValueError('invalid registered input')
        if _sha(item['path'])!=item['sha256']:
            raise ValueError('registered input changed')
    for name,digest in reg['controllers'].items():
        if (_sha(Path(__file__).with_name(name))!=digest
                or hashlib.sha256(_LOADED_SOURCES[name]).hexdigest()!=digest):
            raise ValueError('controller source changed')
    runtime = Path(reg['runtime_root']).resolve()
    if str(runtime)!=reg['runtime_root'] or REPO.resolve()!=runtime:
        raise ValueError('imported runtime differs from registration')
    snapshot = source_snapshot()
    if _digest(snapshot)!=reg['runtime_sha256']:
        raise ValueError('runtime source changed')
    # Refuse protocol errors before any medical array or checkpoint is loaded.
    for stage,cfg in reg['stage_configs'].items():
        if (validate_run_config(cfg)!=cfg or 'ssl_2d_lr_scale' in cfg
                or make_stage_configs(cfg)[stage]!=cfg or cfg['seed']!=20261019
                or cfg['points']!=16384 or cfg['batch_size']!=1 or cfg['accumulation_steps']!=1
                or cfg['crops']!={'2d':[1,160,160],'3d':[128,160,160]}):
            raise ValueError('invalid registered formal recipe')
    return snapshot


def _admit(reg):
    import torch
    from anatobind.aur.data_contract import read_contract
    from anatobind.aur.unified_artifacts import verify_completion_certificate
    from anatobind.aur.unified_run import validate_run_config
    from anatobind.aur.unified_train import _state_hash
    from anatobind.aur.ssl.unified_eval import validate_gate
    snapshot = _validate_registration(reg)
    paths = {name:item['path'] for name,item in reg['inputs'].items()}
    frozen = read_contract(paths['data_contract'],verify=True)
    def input_bytes(name):
        value=Path(paths[name]).read_bytes()
        if hashlib.sha256(value).hexdigest()!=reg['inputs'][name]['sha256']:
            raise ValueError('registered input changed')
        return value
    parent = torch.load(io.BytesIO(input_bytes('parent_export')),map_location='cpu',weights_only=True)
    parent_identity = deepcopy(parent['data_contract'])
    identity = deepcopy(parent_identity)
    if reg.get('initialization_transfer') is not None:
        identity['runtime_sha256']=reg['runtime_sha256']
    if (identity.get('contract_sha256')!=frozen['contract_sha256']
            or identity.get('runtime_sha256')!=reg['runtime_sha256'] or parent.get('stage')!='I'):
        raise ValueError('parent data or runtime identity differs')
    certificate = parent.get('completion_certificate',{})
    if (not _parent_certificate_matches(parent,json.loads(input_bytes('parent_completion')),reg['inputs']['parent_export']['sha256'])
            or not verify_completion_certificate(parent,data_contract=parent_identity)):
        raise ValueError('parent completion invalid')
    final_contents=Path(certificate['final_checkpoint_path']).read_bytes()
    if hashlib.sha256(final_contents).hexdigest()!=certificate['final_checkpoint_sha256']:
        raise ValueError('parent checkpoint changed')
    parent_cfg = torch.load(io.BytesIO(final_contents),map_location='cpu',weights_only=True)['cfg']
    _base._validate_parent_protocol(parent_cfg,parent)
    expected_configs=make_stage_configs(parent_cfg)
    if reg.get('initialization_transfer') is not None:
        for cfg in expected_configs.values():cfg['model']['n_events']=128
    if (parent_cfg!=validate_run_config(parent_cfg) or parent_cfg['seed']!=20261019
            or reg['stage_configs']!=expected_configs):
        raise ValueError('registered recipes differ from inherited protocol')
    gate,complete = json.loads(input_bytes('g1_report')),json.loads(input_bytes('g1_completion'))
    if (not validate_gate(gate,data_contract_sha256=frozen['contract_sha256'],
            backbone_state_sha256=_state_hash(parent['backbone_state_dict']),require_formal=True)
            or complete.get('report_sha256')!=reg['inputs']['g1_report']['sha256']
            or complete.get('pass') is not True or complete.get('frozen_backbones_unchanged') is not True
            or not _g1_parent_matches(gate,certificate,reg['inputs']['parent_export']['sha256'])):
        raise ValueError('parent G1 invalid')
    if reg.get('initialization_transfer') is not None:
        from anatobind.aur.query_transfer import validate_transfer
        ref=reg['initialization_transfer'];contents=Path(ref['path']).read_bytes()
        if hashlib.sha256(contents).hexdigest()!=ref['sha256']:
            raise ValueError('transfer registration changed')
        transfer=json.loads(contents)
        if (transfer.get('parent_export')!=reg['inputs']['parent_export'] or
                transfer.get('g1_report')!=reg['inputs']['g1_report']):
            raise ValueError('transfer sources differ from registration')
        validate_transfer(parent,reg['stage_configs']['II'],identity,gate,transfer)
    from anatobind.aur.unified_launcher import _validate_roles
    _validate_roles(frozen)
    return snapshot,frozen,identity


def _record_gpu_admission(device, destination):
    try:
        record = _gpu_admission(device)
    except ValueError as exc:
        if hasattr(exc, 'diagnostics'):
            _json(destination, exc.diagnostics)
        raise
    _json(destination, record)


def _stage_work(stage,recipe,data_contract,out,device,initializer,g1_report,initialization_transfer=None):
    import torch
    from anatobind.aur import unified_launcher as launcher
    torch.set_num_threads(2)
    original=launcher._initialize
    def admitted_initialize(*args,**kwargs):
        # run_stage has finished startup hashing and created its validated private
        # directory. Recheck the physical UUID at the actual model boundary.
        _record_gpu_admission(device, Path(out)/'gpu_initialization_admission.json')
        return original(*args,**kwargs)
    launcher._initialize=admitted_initialize
    try:
        extra={'initialization_transfer':initialization_transfer} if initialization_transfer is not None else {}
        return launcher.run_stage(recipe,data_contract,stage,out,device=device,
                                  initializer=initializer,g1_report=g1_report,**extra)
    finally:
        launcher._initialize=original


def _run_stage_job(stage,recipe,data_contract,out,device,initializer,g1_report,initialization_transfer=None):
    """Exec a clean interpreter: never perform Torch work after a nested fork."""
    out=_private_path(out)
    request=out.parent/f'{stage}_worker_request.json'
    result_path=out.parent/f'{stage}_worker_result.json'
    if stage not in ('II','III') or out.name!=stage:raise ValueError('invalid stage worker destination')
    _json(request,{'stage':stage,'recipe':recipe,'data_contract':data_contract,'out':str(out),
        'device':device,'initializer':initializer,'g1_report':g1_report,
        **({'initialization_transfer':initialization_transfer} if initialization_transfer is not None else {}),
        'controllers':{name:hashlib.sha256(value).hexdigest() for name,value in _LOADED_SOURCES.items()}})
    bootstrap=('import importlib.util,sys; '
        's=importlib.util.spec_from_file_location("aur_formal_stage_worker",sys.argv[1]); '
        'm=importlib.util.module_from_spec(s); s.loader.exec_module(m); '
        'raise SystemExit(m._stage_subprocess_main(sys.argv[2],sys.argv[3],int(sys.argv[4])))')
    worker=None
    with os.fdopen(os.open(out.parent/f'{stage}_worker.log',os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600),'w') as log:
        try:
            worker=subprocess.Popen([sys.executable,'-c',bootstrap,str(Path(__file__).resolve()),
                str(request),_sha(request),str(os.getpid())],stdout=log,stderr=log)
            worker.wait(timeout=PHASE_LIMITS[stage])
            if worker.returncode!=0 or not result_path.is_file():return dict(FAILED)
            record=_read(result_path)
            if record['worker_pid']!=worker.pid:return dict(FAILED)
            return {**record['result'],'worker_pid':worker.pid}
        except subprocess.TimeoutExpired:
            worker.kill();worker.wait(timeout=5)
            return {'status':'TIMEOUT','evidence':'NOT_EVIDENCE'}
        finally:
            if worker is not None and worker.poll() is None:
                worker.kill();worker.wait(timeout=5)


def _stage_subprocess_main(request_path,request_sha,parent_pid):
    """Internal exec target; install parent-death handling before importing Torch."""
    import ctypes
    import signal
    if ctypes.CDLL(None,use_errno=True).prctl(1,signal.SIGKILL,0,0,0)!=0 or os.getppid()!=parent_pid:
        return 1
    os.umask(0o077)
    try:
        path=_private_path(request_path)
        request_bytes=path.read_bytes()
        if hashlib.sha256(request_bytes).hexdigest()!=request_sha:raise ValueError('stage worker request changed')
        request=json.loads(request_bytes)
        expected={'stage','recipe','data_contract','out','device','initializer','g1_report','controllers'}
        stage=request['stage'];out=_private_path(request['out'])
        if (set(request) not in (expected,expected|{'initialization_transfer'}) or stage not in ('II','III') or out.name!=stage
                or out.parent!=path.parent or path.name!=f'{stage}_worker_request.json'
                or request['controllers']!={n:hashlib.sha256(v).hexdigest() for n,v in _LOADED_SOURCES.items()}):
            raise ValueError('stage worker identity differs')
        try:
            extra={'initialization_transfer':request['initialization_transfer']} if 'initialization_transfer' in request else {}
            result=_stage_work(stage,request['recipe'],request['data_contract'],out,request['device'],
                               request['initializer'],request['g1_report'],**extra)
        except BaseException:
            result=dict(FAILED)
        _json(out.parent/f'{stage}_worker_result.json',{'worker_pid':os.getpid(),'result':result})
        return 0
    except BaseException:
        return 1


def _validate_completed_stage(stage,out,cfg,identity):
    import torch
    from anatobind.aur.unified_artifacts import verify_completion_certificate
    path = out/'selected_export.pt'
    exported = torch.load(path,map_location='cpu',weights_only=True)
    completion = _read(out/'completion.json')
    digest = _sha(path)
    if (exported.get('stage')!=stage or not _parent_certificate_matches(exported,completion,digest)
            or not verify_completion_certificate(exported,data_contract=identity)):
        raise ValueError('formal stage completion invalid')
    certificate = exported['completion_certificate']
    final = torch.load(certificate['final_checkpoint_path'],map_location='cpu',weights_only=True)
    if (final['cfg']!=cfg or certificate['executed_steps']!=cfg['steps']
            or Path(certificate['final_checkpoint_path']).resolve().parent!=out.resolve()
            or Path(certificate['selected_checkpoint_path']).resolve().parent!=out.resolve()):
        raise ValueError('formal stage completion recipe differs')
    return {'selected_export':str(path.resolve()),'selected_export_sha256':digest,
            'completion_sha256':_sha(out/'completion.json'),'steps':cfg['steps'],
            'selected_step':certificate['selected_step']}


def _execute_stages(reg,identity,out,device,send):
    import torch
    initializer = reg['inputs']['parent_export']['path']
    stages = {}
    for stage in ('II','III'):
        send({'phase':stage})
        gc.collect()
        if torch.cuda.is_initialized():
            torch.cuda.empty_cache()
            raise ValueError('orchestrator must not own a CUDA context')
        _record_gpu_admission(device, out/f'gpu_admission_{stage}.json')
        recipe = {'schema_version':1,'stage_config':reg['stage_configs'][stage],
                  'validation_batches':reg['validation_batches']}
        extra=({'initialization_transfer':reg['initialization_transfer']}
               if stage=='II' and 'initialization_transfer' in reg else {})
        result = _run_stage_job(stage,recipe,reg['inputs']['data_contract']['path'],out/stage,device,
            initializer,reg['inputs']['g1_report']['path'] if stage=='II' else None,**extra)
        if result.get('status')!='COMPLETE':
            return {'status':result.get('status','FAILED'),'stage':stage,'evidence':'NOT_EVIDENCE'}
        stages[stage] = _validate_completed_stage(stage,out/stage,recipe['stage_config'],identity)
        initializer = stages[stage]['selected_export']
        _json(out/f'stage_{stage}_verified.json',stages[stage])
    return {'status':'COMPLETE','stages':stages,'evidence':'NOT_EVIDENCE'}


def _execute(registration,out,device,send):
    import torch
    # Bound CPU verification; stage model work executes in fresh interpreters.
    torch.set_num_threads(1)
    send({'phase':'admission'})
    registration_bytes=Path(registration).read_bytes()
    reg,reg_sha=json.loads(registration_bytes),hashlib.sha256(registration_bytes).hexdigest()
    snapshot,frozen,identity = _admit(reg)
    _json(out/'registration.json',reg)
    commitments = {'registration_sha256':reg_sha,'runtime_sha256':reg['runtime_sha256'],'inputs':reg['inputs'],
                   **({'initialization_transfer':reg['initialization_transfer']} if 'initialization_transfer' in reg else {})}
    _json(out/'input_commitments.json',commitments)
    _json(out/'runtime_sources.json',snapshot)
    archive=out/'controller_sources';archive.mkdir(mode=0o700)
    for name,digest in reg['controllers'].items():
        with os.fdopen(os.open(archive/name,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600),'wb') as stream:
            stream.write(_LOADED_SOURCES[name])
        if _sha(archive/name)!=digest:raise ValueError('source archive changed')
    _json(out/'data_coverage.json',frozen['summary'])
    result = _execute_stages(reg,identity,out,device,send)
    send({'phase':'close'})
    closing = _admit(reg)
    if (closing[0]!=snapshot or closing[1]['contract_sha256']!=frozen['contract_sha256']
            or closing[2]!=identity or _sha(registration)!=reg_sha
            or _read(out/'registration.json')!=reg or _read(out/'runtime_sources.json')!=snapshot
            or _read(out/'input_commitments.json')!=commitments
            or any(_sha(archive/name)!=digest for name,digest in reg['controllers'].items())):
        raise ValueError('closing input integrity changed')
    if result['status']=='COMPLETE':
        for stage in ('II','III'):
            verified = _validate_completed_stage(stage,out/stage,reg['stage_configs'][stage],identity)
            if verified!=result['stages'][stage] or _read(out/f'stage_{stage}_verified.json')!=verified:
                raise ValueError('closing stage integrity changed')
    _json(out/'closing_commitments.json',{**commitments,'complete_live_verification':True})
    _json(out/'result.json',result)
    return result


def run_formal(registration_path,out,device='cpu'):
    owned = False
    try:
        out = _private_path(out)
        if out.exists() or out.is_symlink():return dict(FAILED)
        registration=Path(registration_path).expanduser().resolve()
        if device not in ('cpu','cuda','cuda:0'):return dict(FAILED)
        out.parent.mkdir(mode=0o700,parents=True,exist_ok=True)
        out.mkdir(mode=0o700);owned=True
        result = _supervise(lambda send:_execute(registration,out,device,send),
                            total_seconds=TOTAL_SECONDS,phase_limits=PHASE_LIMITS)
        _json(out/'terminal.json',result)
        return result
    except BaseException:
        if owned:
            try:_json(out/'terminal.json',FAILED)
            except Exception:pass
        return dict(FAILED)


class _PrivateParser(argparse.ArgumentParser):
    def error(self,message):
        self.exit(2,'invalid formal continuation arguments\n')


def main(argv=None):
    parser=_PrivateParser(description='Bounded formal II/III continuation (NOT_EVIDENCE).')
    parser.add_argument('--registration',required=True)
    parser.add_argument('--out',required=True)
    parser.add_argument('--device',default='cpu')
    args=parser.parse_args(argv)
    result=run_formal(args.registration,args.out,args.device)
    print(json.dumps({key:result[key] for key in ('status','evidence')},sort_keys=True))
    return 0 if result['status']=='COMPLETE' else 1


if __name__=='__main__':
    raise SystemExit(main())

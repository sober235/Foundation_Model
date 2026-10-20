"""Private, bounded, restartable production launcher for qualified AUR stages."""
import argparse
from collections.abc import Mapping
from copy import deepcopy
import json
import os
from pathlib import Path
import tempfile
from time import monotonic
import traceback

import torch

from anatobind.aur.data_contract import (file_sha256, json_digest, read_contract, validate_private_output)
from anatobind.aur.unified_run import UnifiedRun, atomic_torch_save, validate_run_config
from anatobind.aur.unified_train import _state_hash
from anatobind.aur.unified_views import QualifiedViews

REPO = Path(__file__).resolve().parents[2]
GEOMETRY_VERSION = 'unified_real_plane_geometry_weak2d_v1'
_EXPORT_KEYS = ('format_version','purpose','stage','complete','step','execution_id','model_config','model_state_dict',
                'backbone_state_dict','data_contract','lineage','numerical_environment')


def validate_recipe(recipe):
    try:
        if (set(recipe) != {'schema_version','stage_config','validation_batches'} or recipe['schema_version'] != 1
                or type(recipe['validation_batches']) is not int or recipe['validation_batches'] < 1):
            raise ValueError()
        return {'schema_version':1, 'stage_config':validate_run_config(recipe['stage_config']),
                'validation_batches':recipe['validation_batches']}
    except (KeyError, TypeError, ValueError):
        raise ValueError('invalid complete launcher recipe') from None


def source_snapshot():
    """A fixed filesystem closure, independent of import order and stage recipe."""
    paths = sorted((REPO/'anatobind').rglob('*.py'))
    paths += [REPO/'scripts'/name for name in ('aur_unified_train.py','aur_unified_ssl_eval.py','aur_qualify_training_data.py')]
    return {str(path.relative_to(REPO)):file_sha256(path) for path in paths}


def _directory(path):
    missing, parent = [], path
    while not parent.exists():
        missing.append(parent)
        parent = parent.parent
    for member in reversed(missing):
        member.mkdir(mode=0o700)
    if path.stat().st_mode & 0o077:
        path.chmod(0o700)


def _atomic_bytes(path, content, *, immutable=False):
    _directory(path.parent)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent,prefix='.aur-launcher-',delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        if immutable:
            os.link(temporary,path)
        else:
            os.replace(temporary,path)
        descriptor = os.open(path.parent,os.O_RDONLY|os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _json(path, value, *, immutable=False):
    _atomic_bytes(path,(json.dumps(value,sort_keys=True,indent=1,allow_nan=False)+'\n').encode(),immutable=immutable)


def _read_json(path):
    return json.loads(path.read_text())


def _append(path, value):
    encoded = (json.dumps(value,sort_keys=True,allow_nan=False)+'\n').encode()
    with os.fdopen(os.open(path,os.O_WRONLY|os.O_APPEND|os.O_CREAT,0o600),'ab') as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def _unique(out, stem, suffix):
    path, attempt = out/f'{stem}_1{suffix}', 2
    while path.exists() or path.is_symlink():
        path, attempt = out/f'{stem}_{attempt}{suffix}', attempt+1
    return path


def _preserve_source(out,snapshot):
    for name,digest in snapshot.items():
        path=out/'source_snapshot'/name
        _atomic_bytes(path,(REPO/name).read_bytes(),immutable=True)
        if file_sha256(path)!=digest:
            raise ValueError('source changed while preserving execution snapshot')


def _validation_best(out,runner):
    """Reconstruct a best pointer from durable, checkpoint-bound validations."""
    best=None
    for row in read_jsonl(out/'validation.jsonl'):
        record=row.get('checkpoint')
        if (not isinstance(record,dict) or record.get('step')!=row['step']
                or Path(record.get('path','')).name!=record.get('path')
                or not 0<row['step']<=runner.step_index):
            raise ValueError('validation checkpoint commitment missing')
        path=out/record['path']
        if path.resolve().parent!=out or file_sha256(path)!=record['sha256']:
            raise ValueError('validation checkpoint commitment changed')
        state=torch.load(path,map_location='cpu',weights_only=True)
        if (state['step']!=row['step'] or state['cfg']!=runner.cfg
                or state['data_contract']!=runner.data_contract or state['stage']!=runner.stage
                or not isinstance(row['criterion'],(int,float)) or not torch.isfinite(torch.tensor(row['criterion']))):
            raise ValueError('validation checkpoint state mismatch')
        if best is None or row['criterion']<best['criterion']:
            best={**record,'criterion':row['criterion'],'selection':row['selection']}
    return best


def _validate_boundary(runner,out,factory,n_batches,best):
    last=_save_boundary(runner,out)
    report={'step':runner.step_index,**runner.validate(factory,n_batches),'checkpoint':last}
    _append(out/'validation.jsonl',report)
    if best is None or report['criterion']<best['criterion']:
        best={**last,'criterion':report['criterion'],'selection':report['selection']}
        _json(out/'best.json',best)
    return last,best


def _pointer(out, name):
    try:
        record = _read_json(out/f'{name}.json')
        path = out/record['path']
        if (Path(record['path']).name != record['path'] or path.resolve().parent != out
                or not path.is_file() or file_sha256(path) != record['sha256']
                or type(record['step']) is not int or record['step'] < 0):
            raise ValueError()
        return record, path
    except Exception:
        raise ValueError('invalid checkpoint pointer') from None


def _save_boundary(runner, out):
    if (out/'last.json').exists():
        record, path = _pointer(out,'last')
        if record['step'] == runner.step_index:
            return record
    path = _unique(out,f'checkpoint_step{runner.step_index}', '.pt')
    runner.save(path)
    record = {'path':path.name,'sha256':file_sha256(path),'step':runner.step_index}
    _json(out/'last.json',record)
    return record


def _recover_log(out, name, step, *, sequential=False):
    path = out/f'{name}.jsonl'
    if not path.exists():
        if sequential and step:
            raise ValueError('missing committed training log')
        return
    lines = path.read_bytes().splitlines(keepends=True)
    kept, tail, steps = [], [], []
    for index, line in enumerate(lines):
        try:
            row = json.loads(line)
            number = row['step']
            if type(number) is not int or number < 1:
                raise ValueError()
        except Exception:
            if index != len(lines)-1 or line.endswith(b'\n'):
                raise ValueError('invalid committed log') from None
            tail.append(line)
            continue
        if number > step:
            tail.append(line)
        elif tail:
            raise ValueError('invalid training log ordering')
        else:
            kept.append(line)
            steps.append(number)
    if (steps != sorted(set(steps)) or (sequential and steps != list(range(1,step+1)))):
        raise ValueError('committed log does not match resume boundary')
    if tail:
        archive = _unique(out,f'recovery_{name}', '.jsonl')
        _atomic_bytes(archive,b''.join(tail),immutable=True)
        _atomic_bytes(path,b''.join(kept))


def _validate_roles(contract):
    expected = {'train_ssl':('train_ssl','train'), 'validation_ssl':('ssl_validation','val'),
                'train_supervised':('train_supervised','train'), 'validation':('development_validation','val')}
    train, validation = set(), set()
    for key,(role,split) in expected.items():
        rows = contract.get(key)
        if not rows or any(row.get('role') != role or row.get('split') != split for row in rows):
            raise ValueError('invalid data partition')
        groups = {(row['source'],str(row['patient'])) for row in rows}
        (train if split=='train' else validation).update(groups)
    if train & validation:
        raise ValueError('training and validation partition overlap')


def _inputs(warm_start, initializer, g1_report):
    records = {}
    for name,value in [('warm_start',warm_start),('initializer',initializer),('g1_report',g1_report)]:
        if value is None:
            continue
        policy = None
        if name == 'warm_start':
            policy = value.get('policy','full_stage_one') if isinstance(value,Mapping) else 'full_stage_one'
            value = value['checkpoint'] if isinstance(value,Mapping) else value
        path = Path(value).expanduser().resolve()
        records[name] = {'path':str(path),'sha256':file_sha256(path)}
        if policy is not None:
            records[name]['policy'] = policy
    return records


def _check_inputs(records):
    if any(file_sha256(record['path']) != record['sha256'] for record in records.values()):
        raise ValueError('initialization input integrity changed')


def _initialize(stage, cfg, identity, inputs, device):
    initializer = torch.load(inputs['initializer']['path'],map_location='cpu',weights_only=True) if 'initializer' in inputs else None
    g1_report = _read_json(Path(inputs['g1_report']['path'])) if 'g1_report' in inputs else None
    warm_start = ({'checkpoint':inputs['warm_start']['path'],'policy':inputs['warm_start']['policy']} if 'warm_start' in inputs else None)
    return UnifiedRun(stage,cfg,identity,device=device,initializer=initializer,g1_report=g1_report,warm_start=warm_start)


def _finish_export(runner, out, best, last, inputs):
    selected_checkpoint = out/best['path']
    selected = torch.load(selected_checkpoint,map_location='cpu',weights_only=True)
    exported = {key:selected[key] for key in _EXPORT_KEYS}
    completion = {'version':1,'stage':runner.stage,'purpose':runner.cfg['purpose'],
        'execution_id':runner.execution_id,
        'executed_steps':runner.step_index,'budget_steps':runner.cfg['steps'],'selected_step':selected['step'],
        'selected_model_state_sha256':_state_hash(selected['model_state_dict']),
        'selected_backbone_state_sha256':_state_hash(selected['backbone_state_dict']),
        'data_contract':deepcopy(runner.data_contract),
        'final_checkpoint_path':str((out/last['path']).resolve()),'final_checkpoint_sha256':last['sha256'],
        'selected_checkpoint_path':str(selected_checkpoint.resolve()),'selected_checkpoint_sha256':best['sha256'],
        'g1_report_sha256':inputs.get('g1_report',{}).get('sha256')}
    if runner.cfg['purpose']=='formal':
        from anatobind.aur.unified_artifacts import verify_completion_certificate
        exported['completion_certificate'] = completion
        if not verify_completion_certificate(exported,data_contract=runner.data_contract):
            raise ValueError('formal phase completion verification failed')
    export_path = out/'selected_export.pt'
    if export_path.exists():
        existing=torch.load(export_path,map_location='cpu',weights_only=True)
        tensor_keys={'model_state_dict','backbone_state_dict'}
        if (set(existing)!=set(exported)
                or any(existing[k]!=v for k,v in exported.items() if k not in tensor_keys)
                or any(_state_hash(existing[k])!=_state_hash(exported[k]) for k in tensor_keys)):
            raise ValueError('existing selected export differs from committed validation')
    else:
        atomic_torch_save(exported,export_path)
    _json(out/'completion.json',{**completion,'selected_export_sha256':file_sha256(export_path),
                                'formal_gate_certificate':runner.cfg['purpose']=='formal'},immutable=True)
    return export_path.name


def run_stage(recipe, data_contract, stage, out, device='cpu', warm_start=None, initializer=None,
              g1_report=None, resume=None, *, train_factory=None, validation_factory=None):
    """Run one explicit stage; mapping contracts are for synthetic library tests.

    Source/data verification occurs on entry and close, never per optimizer
    step. max_seconds bounds the training/validation segment after startup.
    Data verification time is reported separately. An interrupted segment is
    resumable and produces no completion certificate or selected export.
    """
    recipe = validate_recipe(recipe)
    cfg = recipe['stage_config']
    if stage not in ('I','II','III'):
        raise ValueError('invalid stage')
    out = validate_private_output(out)
    verification_started = monotonic()
    if isinstance(data_contract,Mapping):
        if cfg['purpose'] != 'synthetic_test':
            raise ValueError('production requires a verified file data contract')
        frozen = deepcopy(data_contract)
    else:
        frozen = read_contract(data_contract,verify=True)
    _validate_roles(frozen)
    snapshot = source_snapshot()
    identity = {'contract_sha256':frozen['contract_sha256'],'runtime_sha256':json_digest(snapshot),
                'geometry_version':GEOMETRY_VERSION}
    supplied = _inputs(warm_start,initializer,g1_report)
    previous = None
    if resume is None:
        if out.exists():
            raise ValueError('run output already exists')
        inputs = supplied
    else:
        previous = _read_json(out/'run_config.json')
        inputs = previous['initialization_inputs']
        if any(inputs.get(key) != value for key,value in supplied.items()):
            raise ValueError('resume initialization configuration mismatch')
    _check_inputs(inputs)
    record = {'schema_version':1,'recipe':recipe,'stage':stage,'data_contract':identity,
              'device':str(torch.device(device)),'initialization_inputs':inputs,'source_snapshot_sha256':json_digest(snapshot)}
    record['execution_id']=json_digest({'output':str(out),'run':record})
    if previous is not None and (previous != record or _read_json(out/'source_snapshot.json') != snapshot):
        raise ValueError('resume source or configuration closure mismatch')
    if previous is None:
        _directory(out)
        _json(out/'run_config.json',record,immutable=True)
        _json(out/'source_snapshot.json',snapshot,immutable=True)
        _preserve_source(out,snapshot)
    elif any(file_sha256(out/'source_snapshot'/name)!=digest for name,digest in snapshot.items()):
        raise ValueError('saved source snapshot integrity changed')
    best, runner = None, None
    try:
        if resume is None:
            runner = _initialize(stage,cfg,identity,inputs,device)
            runner.execution_id=record['execution_id']
            last = _save_boundary(runner,out)
        else:
            last, path = _pointer(out,'last')
            if Path(resume).expanduser().resolve() != path.resolve():
                raise ValueError('resume must match the last committed checkpoint pointer')
            runner = UnifiedRun.resume(path,cfg,identity,device=device)
            if runner.execution_id!=record['execution_id']:
                raise ValueError('resume execution identity mismatch')
            if runner.stage != stage or runner.step_index != last['step']:
                raise ValueError('resume checkpoint pointer stage or step mismatch')
            _recover_log(out,'steps',runner.step_index,sequential=True)
            _recover_log(out,'validation',runner.step_index)
            if (out/'best.json').exists():
                best, best_path = _pointer(out,'best')
                if best['step'] > runner.step_index or not any(row['step']==best['step'] and row['criterion']==best['criterion'] for row in read_jsonl(out/'validation.jsonl')):
                    raise ValueError('best checkpoint pointer does not match committed validation')
                best_state = torch.load(best_path,map_location='cpu',weights_only=True)
                if best_state['step'] != best['step'] or best_state['cfg'] != cfg or best_state['data_contract'] != identity:
                    raise ValueError('best checkpoint pointer state mismatch')
            recovered=_validation_best(out,runner)
            if recovered is not None:
                best=recovered
                _json(out/'best.json',best)
            if (out/'completion.json').exists():
                completion = _read_json(out/'completion.json')
                if (runner.step_index != cfg['steps'] or file_sha256(out/'selected_export.pt') != completion['selected_export_sha256']):
                    raise ValueError('completed artifact integrity mismatch')
                return {'status':'COMPLETE','steps':runner.step_index,'selected_export':'selected_export.pt','best':best,'last':last}
        if train_factory is None:
            train_factory = QualifiedViews(frozen['train_ssl' if stage=='I' else 'train_supervised'],stage,cfg,augment=True,validation=False)
        if validation_factory is None:
            validation_factory = QualifiedViews(frozen['validation_ssl' if stage=='I' else 'validation'],stage,cfg,augment=False,validation=True)
        verification_seconds = monotonic()-verification_started
        started, interrupted = monotonic(), False
        _json(out/'status.json',{'status':'RUNNING','stage':stage,'step':runner.step_index})
        due=runner.step_index>0 and (runner.step_index%cfg['validation_every']==0 or runner.step_index==cfg['steps'])
        if due and not any(r['step']==runner.step_index for r in read_jsonl(out/'validation.jsonl')):
            last,best=_validate_boundary(runner,out,validation_factory,recipe['validation_batches'],best)
        while runner.step_index < cfg['steps']:
            if monotonic()-started >= cfg['max_seconds']:
                interrupted = True
                break
            if runner.device.type=='cuda':
                torch.cuda.synchronize(runner.device)
                torch.cuda.reset_peak_memory_stats(runner.device)
            before = monotonic()
            logged = runner.step(train_factory)
            if runner.device.type=='cuda':
                torch.cuda.synchronize(runner.device)
            logged['observed_wall_seconds'] = monotonic()-before
            if runner.device.type=='cuda':
                logged['peak_cuda_allocated_bytes'] = torch.cuda.max_memory_allocated(runner.device)
                logged['peak_cuda_reserved_bytes'] = torch.cuda.max_memory_reserved(runner.device)
            _append(out/'steps.jsonl',logged)
            print(json.dumps({key:logged[key] for key in ('step','mode','loss','exposures')},sort_keys=True),flush=True)
            if monotonic()-started >= cfg['max_seconds']:
                interrupted = True
                break
            validate = runner.step_index%cfg['validation_every']==0 or runner.step_index==cfg['steps']
            if validate:
                last,best=_validate_boundary(runner,out,validation_factory,recipe['validation_batches'],best)
            elif runner.step_index%cfg['checkpoint_every']==0:
                last = _save_boundary(runner,out)
        last = _save_boundary(runner,out)
        training_seconds = monotonic()-started
        verification_started = monotonic()
        closing = frozen if isinstance(data_contract,Mapping) else read_contract(data_contract,verify=True)
        if closing['contract_sha256'] != frozen['contract_sha256'] or source_snapshot() != snapshot:
            raise ValueError('data or source integrity changed during training')
        _check_inputs(inputs)
        verification_seconds += monotonic()-verification_started
        if not interrupted and not all(runner.exposures[mode]>0 for mode in ('2d','3d')):
            raise ValueError('completed phase requires training exposures in both modes')
        selected_export = None if interrupted else _finish_export(runner,out,best,last,inputs)
        summary = {'status':'INTERRUPTED' if interrupted else 'COMPLETE','stage':stage,'purpose':cfg['purpose'],
                   'steps':runner.step_index,'exposures':dict(runner.exposures),'source_exposures':dict(runner.source_exposures),
                   'source_mode_exposures':deepcopy(runner.source_mode_exposures),
                   'training_seconds':training_seconds,'verification_seconds':verification_seconds,
                   'selected_export':selected_export,'best':best,'last':last}
        _json(_unique(out,'summary','.json'),summary,immutable=True)
        _json(out/'status.json',{'status':summary['status'],'stage':stage,'step':runner.step_index})
        return summary
    except Exception:
        _atomic_bytes(_unique(out,'failure','.txt'),traceback.format_exc().encode(),immutable=True)
        _json(out/'status.json',{'status':'FAILED','stage':stage,'step':runner.step_index if runner is not None else 0})
        raise


class PrivateParser(argparse.ArgumentParser):
    def error(self, message):
        super().error('invalid_arguments')


def parser():
    result = PrivateParser(description='Qualified private single-process AUR training')
    for name in ('config','data-contract','out','device'):
        result.add_argument('--'+name,required=True)
    result.add_argument('--stage',choices=('I','II','III'),required=True)
    result.add_argument('--warm-start')
    result.add_argument('--warm-start-policy',choices=('full_stage_one','backbone_only'),default='full_stage_one')
    result.add_argument('--initializer')
    result.add_argument('--g1-report')
    result.add_argument('--resume')
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        recipe = validate_recipe(_read_json(Path(args.config)))
        if recipe['stage_config']['purpose'] not in ('formal','resource_probe'):
            raise ValueError('CLI requires a production purpose')
        warm_start = {'checkpoint':args.warm_start,'policy':args.warm_start_policy} if args.warm_start else None
        result = run_stage(recipe,args.data_contract,args.stage,args.out,device=args.device,warm_start=warm_start,
                           initializer=args.initializer,g1_report=args.g1_report,resume=args.resume)
        print(json.dumps({key:result[key] for key in ('status','steps')},sort_keys=True),flush=True)
        return 0 if result['status']=='COMPLETE' else 2
    except Exception:
        print(json.dumps({'status':'FAILED','reason':'qualified_training_failed'}),flush=True)
        return 1

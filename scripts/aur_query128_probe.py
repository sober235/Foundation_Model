"""Finite, source-bound 8+8 step capacity/resource probe; never formal evidence."""
import argparse
import hashlib
import importlib.util
import io
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

_SELF_BYTES=Path(__file__).read_bytes()
if compile(_SELF_BYTES,__file__,'exec',dont_inherit=True)!=sys._getframe().f_code:
    raise ValueError('probe controller source differs from disk')
CAPACITY_AUDIT=Path('/home/congcongliu/outputs/aur_capacity_audit_20261026/full_v1')


def _controller():
    path=Path(__file__).with_name('aur_formal_supervised.py')
    content=path.read_bytes()
    spec=importlib.util.spec_from_file_location('aur_query128_formal_controller',path)
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    if hashlib.sha256(content).hexdigest()!=hashlib.sha256(Path(__file__).with_name('aur_formal_supervised.py').read_bytes()).hexdigest():
        raise ValueError('controller changed')
    return mod


def _valid_plan(plan, registration, plan_bytes):
    if (set(plan)!={'version','purpose','formal_registration_sha256','audit_completion_sha256','seed',
            'mode_cycle','execution_cap_steps','scheduler_horizons','steps','selection_source',
            'model_forwards_during_plan_selection','training_updates_during_plan_selection','probe_controller_sha256'} or
            plan['version']!=2 or plan['purpose']!='resource_probe' or plan['mode_cycle']!=['2d','3d'] or
            plan['seed']!=20261019 or plan['execution_cap_steps']!=8 or len(plan['steps'])!=8 or
            plan['scheduler_horizons']!={'II':4096,'III':2048} or plan['model_forwards_during_plan_selection']!=0 or
            plan['training_updates_during_plan_selection']!=0 or plan['selection_source']!='frozen_train_roles_and_capacity_audit' or
            plan['formal_registration_sha256']!=hashlib.sha256(registration).hexdigest()):
        raise ValueError('invalid private probe plan')
    if plan['probe_controller_sha256']!=hashlib.sha256(_SELF_BYTES).hexdigest():
        raise ValueError('probe controller is not registered')
    audit_completion=(CAPACITY_AUDIT/'completion_anonymous.json').read_bytes()
    if hashlib.sha256(audit_completion).hexdigest()!=plan['audit_completion_sha256']:
        raise ValueError('capacity audit completion changed')
    index=json.loads((CAPACITY_AUDIT/'results_index_private.json').read_bytes())
    audited={}
    for name,digest in index.items():
        blob=(CAPACITY_AUDIT/name).read_bytes()
        if hashlib.sha256(blob).hexdigest()!=digest:
            raise ValueError('capacity audit row commitments changed')
        result=json.loads(blob)
        for observation in result['observations']:
            if observation['role']=='train_supervised':
                audited[(observation['mode'],observation['draw'])]=(observation['row_index'],observation['count'])
    counts={'2d':0,'3d':0};stress={65:0,85:0};r_modes={'2d':0,'3d':0}
    for number,step in enumerate(plan['steps']):
        if (set(step)!={'mode','draw','row_index','expected_instances','expected_r_eligible','u_supervised','stress'} or
                step['mode'] not in counts or type(step['draw']) is not int or step['draw']<0 or
                type(step['row_index']) is not int or step['row_index']<0 or
                type(step['expected_instances']) is not int or not 0<=step['expected_instances']<=128 or
                type(step['expected_r_eligible']) is not int or not 0<=step['expected_r_eligible']<=step['expected_instances'] or
                step['u_supervised'] is not True or type(step['stress']) is not bool or
                step['mode']!=('2d' if number%2==0 else '3d')):
            raise ValueError('invalid private probe observation')
        if audited.get((step['mode'],step['draw']))!=(step['row_index'],step['expected_instances']):
            raise ValueError('probe observation is not registered in the CPU capacity audit')
        counts[step['mode']]+=1;r_modes[step['mode']]+=step['expected_r_eligible']
        if step['stress']:
            if step['mode']!='3d' or step['expected_instances'] not in stress:
                raise ValueError('invalid probe capacity stress sample')
            stress[step['expected_instances']]+=1
    if (counts!={'2d':4,'3d':4} or any(n!=1 for n in stress.values()) or
            any(r_modes[m]<=0 for m in r_modes) or len({s['row_index'] for s in plan['steps']})<4):
        raise ValueError('incomplete probe mode, capacity or relation coverage')
    return hashlib.sha256(plan_bytes).hexdigest()


def verify_probe_stage(out, stage, cfg, identity, plan, results):
    """Verify an 8-step resource_probe boundary without creating a formal certificate."""
    out=Path(out)
    if stage not in ('II','III') or len(results)!=8:
        raise ValueError('invalid probe stage boundary')
    setup=json.loads((out/'run_config.json').read_bytes())
    cfg=json.loads(json.dumps(cfg,sort_keys=True))
    summary=json.loads(next(out.glob('summary_*.json')).read_bytes())
    completion=json.loads((out/'completion.json').read_bytes())
    if (summary.get('status')!='PROBE_COMPLETE' or summary.get('purpose')!='resource_probe' or
            summary.get('steps')!=8 or summary.get('execution_cap')!=8 or
            summary.get('scheduler_horizon_steps')!=cfg['steps'] or setup.get('recipe',{}).get('stage_config')!=cfg or
            setup.get('stage')!=stage or setup.get('data_contract')!=identity or
            completion.get('formal_gate_certificate') is not False):
        raise ValueError('probe stage execution identity differs')
    for i,(row,planned) in enumerate(zip(results,plan['steps'])):
        if (row.get('step')!=i+1 or row.get('mode')!=planned['mode'] or
                not math.isfinite(row.get('loss',float('nan'))) or row.get('u_matched_instances')!=planned['expected_instances'] or
                (stage=='III' and row.get('r_instances')!=planned['expected_r_eligible'])):
            raise ValueError('probe step capacity or loss differs')
    if ({m:results[-1]['exposures'][m] for m in ('2d','3d')}!={'2d':4,'3d':4} or
            any(sum(s['expected_r_eligible'] for s in plan['steps'] if s['mode']==m)<=0 for m in ('2d','3d'))):
        raise ValueError('probe requires two-mode exposure and R paths')
    if stage=='III' and any(sum(row.get('r_instances',0) for row in results if row['mode']==m)<=0 for m in ('2d','3d')):
        raise ValueError('Stage III probe did not execute both-mode R losses')
    pointer=json.loads((out/'last.json').read_bytes());checkpoint=out/pointer['path'];blob=checkpoint.read_bytes()
    if hashlib.sha256(blob).hexdigest()!=pointer['sha256']:
        raise ValueError('probe checkpoint hash differs')
    import torch
    from anatobind.aur.unified_run import validate_run_config
    from anatobind.aur.unified_train import _state_hash
    cfg=validate_run_config(cfg)
    state=torch.load(io.BytesIO(blob),map_location='cpu',weights_only=True)
    if (state.get('purpose')!='resource_probe' or state.get('stage')!=stage or state.get('step')!=8 or
            state.get('complete') is not False or state.get('cfg')!=cfg or state.get('data_contract')!=identity or
            state.get('execution_id')!=setup.get('execution_id')):
        raise ValueError('probe checkpoint identity differs')
    export_path=out/'selected_export.pt';export_bytes=export_path.read_bytes()
    if hashlib.sha256(export_bytes).hexdigest()!=completion.get('selected_export_sha256'):
        raise ValueError('probe export hash differs')
    exported=torch.load(io.BytesIO(export_bytes),map_location='cpu',weights_only=True)
    if (exported.get('purpose')!='resource_probe' or exported.get('stage')!=stage or exported.get('step')!=8 or
            exported.get('complete') is not False or 'completion_certificate' in exported or
            exported.get('execution_id')!=state['execution_id'] or exported.get('data_contract')!=identity):
        raise ValueError('probe export must remain explicitly uncertified')
    if (_state_hash(exported['model_state_dict'])!=_state_hash(state['model_state_dict']) or
            _state_hash(exported['backbone_state_dict'])!=_state_hash(state['backbone_state_dict'])):
        raise ValueError('probe selected export does not match selected checkpoint weights')
    return {'stage':stage,'steps':8,'probe_purpose':'resource_probe','scheduler_horizon_steps':cfg['steps'],'execution_id':state['execution_id'],
            'checkpoint_sha256':pointer['sha256'],'export_sha256':hashlib.sha256(export_bytes).hexdigest(),
            'completion_sha256':hashlib.sha256((out/'completion.json').read_bytes()).hexdigest(),
            'mode_exposures':results[-1]['exposures'],'losses_finite':True}


def _probe_worker(request_path, request_sha, parent_pid):
    import ctypes
    if ctypes.CDLL(None,use_errno=True).prctl(1,signal.SIGKILL,0,0,0)!=0 or os.getppid()!=parent_pid:return 1
    os.umask(0o077)
    try:
        path=Path(request_path);blob=path.read_bytes()
        if hashlib.sha256(blob).hexdigest()!=request_sha:raise ValueError
        req=json.loads(blob);stage=req['stage'];out=Path(req['out']);cfg=req['cfg']
        if hashlib.sha256(_SELF_BYTES).hexdigest()!=req['probe_script_sha256']:
            raise ValueError('probe controller request source differs')
        if stage=='III':
            previous=req['prior_probe_completion'];initializer=Path(req['initializer']).read_bytes()
            if (hashlib.sha256(initializer).hexdigest()!=req['initializer_sha256'] or
                    previous.get('stage')!='II' or previous.get('steps')!=8 or
                    previous.get('export_sha256')!=req['initializer_sha256'] or
                    previous.get('probe_purpose')!='resource_probe'):
                raise ValueError('probe III initializer is not the verified probe II export')
        else:initializer=None
        ctl_source=Path(req['formal_controller']).read_bytes();helper_source=Path(req['formal_helper']).read_bytes()
        if ({'aur_formal_supervised.py':hashlib.sha256(ctl_source).hexdigest(),
             'aur_supervised_capability.py':hashlib.sha256(helper_source).hexdigest()}!=req['controllers']):
            raise ValueError('probe admission controller changed')
        from anatobind.aur.data_contract import read_contract,file_sha256
        from anatobind.aur.unified_views import QualifiedViews
        from anatobind.aur import unified_launcher as launcher
        frozen=read_contract(req['data_contract'],verify=True)
        if frozen['contract_sha256']!=req['identity']['contract_sha256']:raise ValueError
        records=([x for x in req['plan']['steps'] if x['mode']=='2d'],[x for x in req['plan']['steps'] if x['mode']=='3d'])
        draws={'2d':[x['draw'] for x in records[0]],'3d':[x['draw'] for x in records[1]]}
        offsets={'2d':0,'3d':0};consumed=[]
        view=QualifiedViews(frozen['train_supervised'],stage,cfg,augment=True,validation=False)
        def factory(mode,_draw):
            index=offsets[mode];entry=records[0 if mode=='2d' else 1][index]
            row_index=view.sampler.indices(mode,entry['draw'],cfg['batch_size'])[0]
            if row_index!=entry['row_index']:raise ValueError('probe observation row differs')
            batch=view(mode,entry['draw'])
            count=int(batch['n_instances'][0]);r_count=int(batch['r_eligible'][0].sum())
            if (count!=entry['expected_instances'] or r_count!=entry['expected_r_eligible'] or
                    not bool(batch['u_supervised'][0])):raise ValueError('probe target capacity/R qualification differs')
            consumed.append((mode,entry));offsets[mode]+=1
            return batch
        import importlib.util
        spec=importlib.util.spec_from_file_location('probe_guard_controller',req['formal_controller'])
        ctl=importlib.util.module_from_spec(spec);spec.loader.exec_module(ctl)
        if {n:hashlib.sha256(v).hexdigest() for n,v in ctl._LOADED_SOURCES.items()}!=req['controllers']:
            raise ValueError('loaded admission controller changed')
        original=launcher._initialize
        def guarded(*args,**kwargs):
            ctl._record_gpu_admission(req['device'],out/'gpu_initialization_admission.json')
            return original(*args,**kwargs)
        launcher._initialize=guarded
        try:
            result=launcher.run_stage({'schema_version':1,'stage_config':cfg,'validation_batches':8},
                req['data_contract'],stage,out,device=req['device'],initializer=req['initializer'],
                g1_report=req['g1_report'],train_factory=factory,execution_cap=8,
                **({'initialization_transfer':req['initialization_transfer']} if req.get('initialization_transfer') else {}))
        finally:launcher._initialize=original
        if result.get('status')!='PROBE_COMPLETE' or len(consumed)!=8 or offsets!={'2d':4,'3d':4}:raise ValueError('probe execution incomplete')
        steps=[json.loads(x) for x in (out/'steps.jsonl').read_text().splitlines()]
        verified=verify_probe_stage(out,stage,cfg,req['identity'],req['plan'],steps)
        with (out/'probe_stage_complete.json').open('x') as f:json.dump(verified,f,sort_keys=True);f.write('\n')
        return 0
    except BaseException:
        return 1


def _execute(registration, plan_path, out, device, send):
    ctl=_controller();reg_bytes=Path(registration).read_bytes();reg=json.loads(reg_bytes)
    plan_bytes=Path(plan_path).read_bytes();plan=json.loads(plan_bytes);plan_sha=_valid_plan(plan,reg_bytes,plan_bytes)
    snapshot,frozen,identity=ctl._admit(reg)
    if not reg.get('initialization_transfer'):raise ValueError('128-query transfer is required')
    ctl._json(out/'registration.json',reg);ctl._json(out/'plan_private.json',plan)
    commitments={'registration_sha256':hashlib.sha256(reg_bytes).hexdigest(),'plan_sha256':plan_sha,
        'runtime_sha256':reg['runtime_sha256'],'input_commitments':reg['inputs'],
        'transfer_sha256':reg['initialization_transfer']['sha256']}
    ctl._json(out/'commitments.json',commitments)
    run_cfgs={}
    for stage in ('II','III'):
        cfg=dict(reg['stage_configs'][stage]);cfg['purpose']='resource_probe';run_cfgs[stage]=cfg
    transfer_bytes=Path(reg['initialization_transfer']['path']).read_bytes()
    if hashlib.sha256(transfer_bytes).hexdigest()!=reg['initialization_transfer']['sha256']:
        raise ValueError('registered transfer bytes changed')
    probe_transfer=json.loads(transfer_bytes)
    probe_transfer=dict(probe_transfer);probe_transfer['child_config_sha256']=ctl._digest(run_cfgs['II'])
    transfer_path=out/'probe_transfer_private.json';ctl._json(transfer_path,probe_transfer)
    states={};initializer=reg['inputs']['parent_export']['path']
    for stage in ('II','III'):
        send({'phase':stage});ctl._record_gpu_admission(device,out/f'gpu_admission_{stage}.json')
        stage_out=out/stage
        request=out/f'{stage}_request.json'
        prior_complete=None;initializer_sha=None
        if stage=='III':
            prior_path=out/'II/probe_stage_complete.json';prior_bytes=prior_path.read_bytes()
            prior_complete=json.loads(prior_bytes)
            initializer_path=out/'II/selected_export.pt';initializer_bytes=initializer_path.read_bytes()
            initializer_sha=hashlib.sha256(initializer_bytes).hexdigest()
            if (prior_complete.get('stage')!='II' or prior_complete.get('export_sha256')!=initializer_sha or
                    prior_complete.get('steps')!=8 or prior_complete.get('probe_purpose')!='resource_probe'):
                raise ValueError('verified probe II is not a valid probe III initializer')
            immutable=out/'probe_II_export_for_III.pt'
            with immutable.open('xb') as f:f.write(initializer_bytes);f.flush();os.fsync(f.fileno())
            immutable.chmod(0o400);initializer=str(immutable)
        req={'stage':stage,'out':str(stage_out),'cfg':run_cfgs[stage],'data_contract':reg['inputs']['data_contract']['path'],
            'identity':identity,'plan':plan,'device':device,'initializer':initializer,'g1_report':reg['inputs']['g1_report']['path'] if stage=='II' else None,
            'initialization_transfer':{'path':str(transfer_path.resolve()),'sha256':hashlib.sha256(transfer_path.read_bytes()).hexdigest()} if stage=='II' else None,
            'initializer_sha256':initializer_sha,'prior_probe_completion':prior_complete,
            'formal_controller':str(Path(ctl.__file__).resolve()),
            'formal_helper':str(Path(ctl.__file__).with_name('aur_supervised_capability.py').resolve()),
            'controllers':reg['controllers'],'probe_script_sha256':hashlib.sha256(_SELF_BYTES).hexdigest()}
        ctl._json(request,req);blob=request.read_bytes();sha=hashlib.sha256(blob).hexdigest();worker=None
        log=out/f'{stage}_worker.log'
        with log.open('x') as stream:
            boot=('import importlib.util,sys; s=importlib.util.spec_from_file_location("query128_probe",sys.argv[1]); '
                  'm=importlib.util.module_from_spec(s);s.loader.exec_module(m);raise SystemExit(m._probe_worker(sys.argv[2],sys.argv[3],int(sys.argv[4])))')
            worker=subprocess.Popen([sys.executable,'-c',boot,str(Path(__file__).resolve()),str(request),sha,str(os.getpid())],
                cwd=Path.cwd(),stdout=stream,stderr=stream,start_new_session=False)
            try:worker.wait(timeout=1800)
            except subprocess.TimeoutExpired:worker.kill();worker.wait(timeout=5);raise TimeoutError
        if worker.returncode!=0:raise ValueError('probe stage worker failed')
        steps=[json.loads(x) for x in (stage_out/'steps.jsonl').read_text().splitlines()]
        states[stage]=verify_probe_stage(stage_out,stage,run_cfgs[stage],identity,plan,steps)
    send({'phase':'close'})
    closed=ctl._admit(reg)
    if closed[0]!=snapshot or closed[1]['contract_sha256']!=frozen['contract_sha256'] or Path(registration).read_bytes()!=reg_bytes or Path(plan_path).read_bytes()!=plan_bytes:
        raise ValueError('probe closing commitments changed')
    result={'status':'COMPLETE','purpose':'resource_probe','evidence':'NOT_EVIDENCE','stages':states,
            'model_forwards_are_probe_training_only':True,'formal_completion_certificate':False,
            'limited_total_budget_projection':'unsupported from eight selected observations'}
    ctl._json(out/'result.json',result);ctl._json(out/'closing_commitments.json',commitments)
    return result


def run(registration,plan,out,device='cuda:0'):
    out=None
    try:
        ctl=_controller();out=ctl._private_path(out)
        if out.exists():return {'status':'FAILED','evidence':'NOT_EVIDENCE'}
        out.parent.mkdir(mode=0o700,parents=True,exist_ok=True);out.mkdir(mode=0o700)
        return ctl._supervise(lambda send:_execute(registration,plan,out,device,send),total_seconds=5400,
                              phase_limits={'II':2000,'III':2000,'close':600,'admission':1000})
    except BaseException:
        return {'status':'FAILED','evidence':'NOT_EVIDENCE'}


class Parser(argparse.ArgumentParser):
    def error(self,message):self.exit(2,'invalid resource probe arguments\n')


def main(argv=None):
    p=Parser();p.add_argument('--registration',required=True);p.add_argument('--plan',required=True);p.add_argument('--out',required=True);p.add_argument('--device',default='cuda:0');a=p.parse_args(argv)
    result=run(a.registration,a.plan,a.out,a.device);print(json.dumps({k:result[k] for k in ('status','evidence')},sort_keys=True));return 0 if result['status']=='COMPLETE' else 1


if __name__=='__main__':raise SystemExit(main())

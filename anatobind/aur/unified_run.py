"""Qualified single-process training, stage gates, accumulation and strict recovery.

The established pilot supplies the unchanged model construction and supervised
loss mechanism. This orchestration adds an explicit recipe, image-only Stage I
factories, patient-balanced draws, validation and formal transition contracts.
"""
from collections import Counter, defaultdict
from copy import deepcopy
import os
from pathlib import Path
import random
import tempfile

import numpy as np
import torch
import torch.distributed as dist

from anatobind.aur import train as T
from anatobind.aur.model import DEFAULTS as MODEL_DEFAULTS
from anatobind.aur.model import AnatoBindBrain
from anatobind.aur.ssl.model import StageOne
from anatobind.aur.ssl.model import DEFAULTS as SSL_DEFAULTS
from anatobind.aur.unified import inactive_adapter_parameters
from anatobind.aur.unified_train import (UnifiedPilot, _builtin, _cpu_copy,
                                       _numerical_environment, _state_hash)

FORMAT_VERSION = 2
_REQUIRED = {'schema_version','purpose','model','seed','steps','lr','weight_decay','points',
    'lambda_r','lambda_h','u_mask_profile','mode_cycle','batch_size','accumulation_steps',
    'warmup_steps','backbone_warm_steps','backbone_warm_scale','crops','ssl',
    'validation_every','checkpoint_every','max_seconds','planar_relation_policy','hcp_cap'}
_SOURCES = {'pdgm','bmsr','isles','sibbms','hcp','synthetic'}


def validate_run_config(cfg):
    """Resolve a complete, finite, versioned recipe; never accept pilot presets."""
    from anatobind.aur import losses as L
    try:
        if not _REQUIRED <= set(cfg) or set(cfg) - _REQUIRED - {'ssl_2d_lr_scale'}:
            raise ValueError
        out = _builtin(cfg)
        # Preserve omitted keys for historical format2 canonical certificates.
        scale = out.get('ssl_2d_lr_scale',1)
        if type(scale) not in (int,float) or scale <= 0:
            raise ValueError
        if out['schema_version'] != 1 or out['purpose'] not in ('formal','resource_probe','synthetic_test'):
            raise ValueError
        if set(out['model']) - set(MODEL_DEFAULTS):
            raise ValueError
        out['model'] = _builtin({**MODEL_DEFAULTS, **out['model']})
        if out['model']['dual_input'] is not True or out['mode_cycle'] != ['2d','3d']:
            raise ValueError
        for key in ('steps','points','batch_size','accumulation_steps','validation_every','checkpoint_every'):
            if type(out[key]) is not int or out[key] < 1:
                raise ValueError
        if out['steps']<2:
            raise ValueError
        for key in ('seed','warmup_steps','backbone_warm_steps'):
            if type(out[key]) is not int or out[key] < 0:
                raise ValueError
        if out['warmup_steps'] >= out['steps']:
            raise ValueError
        for key in ('weight_decay','lambda_r','lambda_h'):
            if type(out[key]) not in (int,float) or out[key] < 0:
                raise ValueError
        if not 0 < out['backbone_warm_scale'] <= 1 or not 0 < out['hcp_cap'] <= 1 or out['max_seconds'] <= 0:
            raise ValueError
        if out['u_mask_profile'] not in L.U_MASK_PROFILES or out['planar_relation_policy'] != 'geometry_weak2d_v1':
            raise ValueError
        if set(out['crops']) != {'2d','3d'}:
            raise ValueError
        patch = out['model']['patch']
        for mode, shape in out['crops'].items():
            stride = [1,*patch[1:]] if mode == '2d' else patch
            if (len(shape) != 3 or any(type(n) is not int or n <= 0 or n % s for n,s in zip(shape,stride))
                    or (mode == '2d' and shape[0] != 1)):
                raise ValueError
        allowed = {'mask_ratio','block_mm','contrast_weight','contrast_dim','temperature'}
        if set(out['ssl']) - allowed:
            raise ValueError
        out['ssl'] = _builtin({**{k:SSL_DEFAULTS[k] for k in allowed}, **out['ssl']})
        if (not 0 < out['ssl']['mask_ratio'] < 1 or out['ssl']['contrast_weight'] < 0
                or out['ssl']['temperature'] <= 0 or type(out['ssl']['contrast_dim']) is not int
                or out['ssl']['contrast_dim'] <= 0 or len(out['ssl']['block_mm']) != 2
                or not 0 < out['ssl']['block_mm'][0] <= out['ssl']['block_mm'][1]
                or (out['ssl']['contrast_weight'] > 0 and out['batch_size'] < 2)):
            raise ValueError
        rates = out['lr'].values() if isinstance(out['lr'],dict) else [out['lr']]
        if not all(type(v) in (int,float) and v > 0 for v in rates):
            raise ValueError
        return out
    except (TypeError,KeyError,ValueError,OverflowError):
        raise ValueError('invalid qualified training configuration') from None


def _restore_rng(obj, rng):
    torch.set_rng_state(rng['torch'])
    if rng['cuda'] is not None:
        torch.cuda.set_rng_state_all(rng['cuda'])
    random.setstate(rng['python'])
    n = rng['numpy']
    np.random.set_state((n['algorithm'],np.asarray(n['keys'],dtype=np.uint32),n['pos'],n['has_gauss'],n['cached']))
    obj.point_generator.set_state(rng['points'])
    obj.mask_generator.set_state(rng['masks'])


def atomic_torch_save(payload, path):
    path = Path(path)
    if path.exists() or path.is_symlink():
        raise FileExistsError('checkpoint already exists')
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent,prefix='.aur-qualified-',delete=False) as stream:
            temporary = stream.name
            torch.save(payload,stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary,path)
        fd = os.open(path.parent,os.O_RDONLY|os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    finally:
        if temporary is not None:
            os.unlink(temporary)
    return str(path)


class UnifiedRun(UnifiedPilot):
    """Reuse the verified losses; own the formal schedule and complete state."""
    def __init__(self, stage, cfg, data_contract, device='cpu', initializer=None, g1_report=None, warm_start=None):
        self._setup(stage,cfg,data_contract,device)
        if self.cfg['purpose'] == 'formal' and stage == 'I' and warm_start is None:
            raise ValueError('formal Stage I requires a registered warm-start')
        if stage != 'I':
            previous = 'I' if stage == 'II' else 'II'
            if (not isinstance(initializer,dict) or initializer.get('format_version') != FORMAT_VERSION
                    or initializer.get('stage') != previous):
                raise ValueError('invalid previous-stage initializer')
            if self.cfg['purpose'] == 'formal':
                if initializer.get('purpose') != 'formal':
                    raise ValueError('formal transition requires a formal source stage')
                from anatobind.aur.unified_artifacts import verify_completion_certificate
                if not verify_completion_certificate(initializer,data_contract=self.data_contract):
                    raise ValueError('formal transition requires a verified completed formal source stage')
                if stage == 'II':
                    from anatobind.aur.ssl.unified_eval import validate_gate
                    if not validate_gate(g1_report,data_contract_sha256=self.data_contract['contract_sha256'],
                            backbone_state_sha256=_state_hash(initializer['backbone_state_dict']),require_formal=True):
                        raise ValueError('both qualified G1 modes must pass before formal Stage II')
            if initializer['model_config'] != self.cfg['model'] or initializer['data_contract'] != self.data_contract:
                raise ValueError('initializer configuration or data contract mismatch')
        self._warm_start = warm_start if stage == 'I' else None
        self._build()
        if stage == 'I':
            if warm_start is None:
                self.lineage = {'kind':'random','seed':self.cfg['seed'],'formal':False}
            else:
                self.lineage = self._warm_start_lineage
        else:
            state = initializer['backbone_state_dict'] if stage == 'II' else initializer['model_state_dict']
            (self.model.backbone if stage == 'II' else self.model).load_state_dict(state,strict=True)
            self.lineage = {'kind':f'stage_{previous}_export','source':deepcopy(initializer['lineage']),
                            'source_state_sha256':_state_hash(state),'g1':_builtin(g1_report) if stage == 'II' else None}

    def _setup(self, stage, cfg, data_contract, device):
        if stage not in ('I','II','III') or int(os.environ.get('WORLD_SIZE','1')) != 1 or (dist.is_initialized() and dist.get_world_size()!=1):
            raise ValueError('qualified training requires one process and a valid stage')
        self.stage,self.cfg,self.data_contract = stage,validate_run_config(cfg),_builtin(data_contract)
        if stage != 'I' and self.cfg.get('ssl_2d_lr_scale',1) != 1:
            raise ValueError('nonunit SSL adapter learning-rate scale requires Stage I')
        if (not isinstance(self.data_contract.get('contract_sha256'),str)
                or len(self.data_contract['contract_sha256']) != 64):
            raise ValueError('invalid data contract hash')
        self.device = torch.device(device)
        self.numerical_environment = _numerical_environment(self.device)
        self.step_index = 0
        self.cursors,self.exposures = {'2d':0,'3d':0},{'2d':0,'3d':0}
        self.source_exposures = {}
        self.source_mode_exposures = {'2d':{},'3d':{}}
        self.failed = False

    def _build(self):
        seed=self.cfg['seed']
        random.seed(seed)
        np.random.seed(seed % (2**32))
        torch.manual_seed(seed)
        if self.stage == 'I':
            keys=('embed','depths','heads','window','patch','use_checkpoint','dual_input')
            self.model=StageOne(**{k:self.cfg['model'][k] for k in keys},**self.cfg['ssl'])
            if getattr(self,'_warm_start',None) is not None:
                from anatobind.aur.ssl.unified_init import warm_start_stage_one
                self._warm_start_lineage=warm_start_stage_one(self.model,self._warm_start['checkpoint'],
                    policy=self._warm_start.get('policy','full_stage_one'),seed=seed)
        else:
            self.model=AnatoBindBrain(**self.cfg['model'])
        # Device transfer may replace parameter objects. Construct the optimizer
        # only after both initialization and the final transfer.
        self.model.to(self.device).train()
        names=['ssl'] if self.stage=='I' else ['backbone','heads']+(['relation'] if self.stage=='III' else [])
        lr=self.cfg['lr']
        if isinstance(lr,dict) and set(lr)!=set(names):
            raise ValueError('learning-rate configuration does not match stage')
        rates=lr if isinstance(lr,dict) else dict.fromkeys(names,float(lr))
        groups=([{'name':'ssl','params':list(self.model.parameters()),'lr':rates['ssl']}] if self.stage=='I'
                else T.param_groups(self.model,self.stage,rates))
        if self.stage == 'I' and self.cfg.get('ssl_2d_lr_scale',1) != 1:
            names_2d = ('backbone.patch_embed_2d.weight','backbone.patch_embed_2d.bias',
                        'decoder.output_2d.weight','decoder.output_2d.bias')
            named = dict(self.model.named_parameters())
            groups = [
                {'name':'ssl','params':[p for name,p in named.items() if name not in names_2d],
                 'lr':rates['ssl']},
                {'name':'ssl_2d','params':[named[name] for name in names_2d],
                 'lr':rates['ssl']*self.cfg['ssl_2d_lr_scale']},
            ]
        self.optimizer=torch.optim.AdamW(groups,weight_decay=self.cfg['weight_decay'],betas=(.9,.95))
        schedules = [T.group_lr_lambda(g['name'],self.stage,self.cfg['warmup_steps'],self.cfg['steps'],
                    self.cfg['backbone_warm_steps'],self.cfg['backbone_warm_scale']) for g in self.optimizer.param_groups]
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer,schedules)
        self.point_generator=torch.Generator(device=self.device).manual_seed(seed+101)
        self.mask_generator=torch.Generator().manual_seed(seed+202)

    def _check_numerical_environment(self):
        if self.numerical_environment != _numerical_environment(self.device):
            raise ValueError('qualified numerical environment changed')

    def _loss(self,batch,mode):
        if self.stage != 'I':
            return self._supervised(batch,mode)
        result = self.model(batch,self.mask_generator,input_mode=mode)
        return result['loss'],{k:float(result[k]) for k in ('mim','contrast','hidden_share','hidden_voxels')}

    def _batch(self,batch,mode):
        if (batch.get('input_mode') != mode or batch['valid'].shape[0] != self.cfg['batch_size']
                or tuple(batch['valid'].shape[1:]) != tuple(self.cfg['crops'][mode])
                or not torch.isfinite(batch['valid']).all()
                or not (batch['valid'].flatten(1).sum(1)>0).all()):
            raise ValueError('batch configuration mismatch or invalid observation')
        if self.stage != 'I':
            for b,count in enumerate(batch['n_instances']):
                if count > self.cfg['model']['n_events'] or tuple(batch['r_eligible'][b].shape)!=(count,):
                    raise ValueError('instance or relation eligibility capacity mismatch')
        return T.to_device(batch,self.device)

    def step(self,batch_factory):
        if self.failed or self.step_index >= self.cfg['steps']:
            raise ValueError('training budget exhausted or failed step requires investigation')
        self._check_numerical_environment()
        mode,draw = self.next_mode,self.cursors[self.next_mode]
        logs,total = Counter(),0.
        sources = Counter()
        self.optimizer.zero_grad(set_to_none=True)
        try:
            for micro in range(self.cfg['accumulation_steps']):
                batch = batch_factory(mode,draw+micro)
                labels = batch.get('source_labels')
                if (not isinstance(labels,list) or len(labels)!=self.cfg['batch_size']
                        or any(s not in _SOURCES for s in labels)):
                    raise ValueError('batch source provenance missing or invalid')
                batch = self._batch(batch,mode)
                loss,logged = self._loss(batch,mode)
                if not torch.isfinite(loss):
                    raise FloatingPointError('nonfinite qualified loss')
                (loss/self.cfg['accumulation_steps']).backward()
                total += float(loss.detach())/self.cfg['accumulation_steps']
                logs.update(logged)
                sources.update(labels)
            for p in inactive_adapter_parameters(self.model,mode):
                p.grad = None
            torch.nn.utils.clip_grad_norm_(self.model.parameters(),1.,error_if_nonfinite=True)
            self.optimizer.step()
            self.scheduler.step()
        except Exception:
            self.failed = True
            self.optimizer.zero_grad(set_to_none=True)
            raise
        self.step_index += 1
        self.cursors[mode] += self.cfg['accumulation_steps']
        self.exposures[mode] += self.cfg['batch_size']*self.cfg['accumulation_steps']
        self.source_exposures = dict(Counter(self.source_exposures)+sources)
        self.source_mode_exposures[mode] = dict(Counter(self.source_mode_exposures[mode])+sources)
        count_keys = {'r_samples','r_instances','r_correct','u_matched_instances','hidden_voxels'}
        logged = {k:(v if k in count_keys else v/self.cfg['accumulation_steps']) for k,v in logs.items()}
        return {**logged,'mode':mode,'step':self.step_index,'draw_index':draw,'loss':total,
            'exposures':dict(self.exposures),'source_exposures':dict(self.source_exposures),
            'source_mode_exposures':deepcopy(self.source_mode_exposures),
            'lr':{g['name']:g['lr'] for g in self.optimizer.param_groups}}

    @torch.no_grad()
    def validate(self,batch_factory,n_batches=1):
        if type(n_batches) is not int or n_batches<1:
            raise ValueError('invalid validation budget')
        self._check_numerical_environment()
        rng = self._rng()
        training = self.model.training
        modes = {}
        self.model.eval()
        try:
            self.point_generator.manual_seed(self.cfg['seed']+9001)
            self.mask_generator.manual_seed(self.cfg['seed']+9002)
            for mode in ('2d','3d'):
                values = []
                for draw in range(n_batches):
                    batch = self._batch(batch_factory(mode,draw),mode)
                    loss,logged = self._loss(batch,mode)
                    if not torch.isfinite(loss):
                        raise FloatingPointError('nonfinite validation loss')
                    values.append(float(loss))
                modes[mode] = {'loss':sum(values)/len(values),'batches':len(values),
                               'observations':len(values)*self.cfg['batch_size']}
        finally:
            _restore_rng(self,rng)
            self.model.train(training)
        return {'criterion':sum(v['loss'] for v in modes.values())/2,'modes':modes,
                'selection':'mean_loss_equal_modes','evidence':'NOT_EVIDENCE'}

    def export(self):
        self._check_numerical_environment()
        if self.failed:
            raise ValueError('failed step cannot become a resumable stage export')
        return {'format_version':FORMAT_VERSION,'purpose':self.cfg['purpose'],'stage':self.stage,
            'execution_id':getattr(self,'execution_id',None),
            'complete':self.step_index==self.cfg['steps'],'step':self.step_index,
            'model_config':deepcopy(self.cfg['model']),'model_state_dict':_cpu_copy(self.model.state_dict()),
            'backbone_state_dict':_cpu_copy(self.model.backbone.state_dict()),
            'data_contract':deepcopy(self.data_contract),'lineage':deepcopy(self.lineage),
            'numerical_environment':deepcopy(self.numerical_environment)}

    def save(self,path):
        payload = {**self.export(),'cfg':deepcopy(self.cfg),'device_type':self.device.type,
            'next_mode':self.next_mode,'cursors':dict(self.cursors),'exposures':dict(self.exposures),
            'source_exposures':dict(self.source_exposures),'optimizer':_cpu_copy(self.optimizer.state_dict()),
            'source_mode_exposures':deepcopy(self.source_mode_exposures),
            'scheduler':self.scheduler.state_dict(),'rng':self._rng()}
        return atomic_torch_save(payload,path)

    @classmethod
    def resume(cls,path,cfg,data_contract,device='cpu'):
        ck = torch.load(path,map_location='cpu',weights_only=True)
        if (ck.get('format_version')!=FORMAT_VERSION or ck.get('cfg')!=validate_run_config(cfg)
                or ck.get('data_contract')!=_builtin(data_contract)):
            raise ValueError('resume configuration or data contract mismatch')
        step,cycle = ck.get('step'),ck['cfg']['mode_cycle']
        if type(step) is not int or not 0<=step<=ck['cfg']['steps']:
            raise ValueError('invalid resume step')
        whole,remainder = divmod(step,len(cycle))
        cursors = {m:(whole*cycle.count(m)+cycle[:remainder].count(m))*ck['cfg']['accumulation_steps'] for m in ('2d','3d')}
        if (ck.get('cursors')!=cursors or ck.get('next_mode')!=cycle[remainder]
                or ck.get('exposures')!={m:n*ck['cfg']['batch_size'] for m,n in cursors.items()}):
            raise ValueError('resume cursor or exposure mismatch')
        sources = ck.get('source_exposures')
        if (not isinstance(sources,dict) or any(k not in _SOURCES or type(v) is not int or v<0 for k,v in sources.items())
                or sum(sources.values())!=sum(ck['exposures'].values())):
            raise ValueError('resume source exposure mismatch')
        mode_sources=ck.get('source_mode_exposures')
        if not isinstance(mode_sources,dict) or set(mode_sources)!={'2d','3d'}:
            raise ValueError('resume mode source exposure mismatch')
        for mode,values in mode_sources.items():
            if (not isinstance(values,dict) or any(k not in _SOURCES or type(v) is not int or v<0 for k,v in values.items())
                    or sum(values.values())!=ck['exposures'][mode]):
                raise ValueError('resume mode source exposure mismatch')
        if dict(Counter(mode_sources['2d'])+Counter(mode_sources['3d']))!=sources:
            raise ValueError('resume source totals mismatch')
        obj = cls.__new__(cls)
        obj._setup(ck['stage'],cfg,data_contract,device)
        if ck.get('device_type')!=obj.device.type or ck.get('numerical_environment')!=obj.numerical_environment:
            raise ValueError('resume numerical environment mismatch')
        obj._build()
        obj.model.load_state_dict(ck['model_state_dict'],strict=True)
        obj.optimizer.load_state_dict(ck['optimizer'])
        obj.scheduler.load_state_dict(ck['scheduler'])
        obj.lineage=deepcopy(ck['lineage'])
        obj.step_index,obj.cursors,obj.exposures = step,dict(cursors),dict(ck['exposures'])
        obj.source_exposures=dict(sources)
        obj.source_mode_exposures=deepcopy(mode_sources)
        obj.execution_id=ck.get('execution_id')
        _restore_rng(obj,ck['rng'])
        return obj


class PatientBalancedSampler:
    """Source/patient/sequence draws; HCP caps per-draw probability, not realized small-batch counts."""
    def __init__(self,rows,seed,hcp_cap=.25):
        self.rows,self.seed = list(rows),int(seed)
        self.groups = defaultdict(lambda:defaultdict(lambda:defaultdict(list)))
        for i,r in enumerate(self.rows):
            self.groups[r['source']][str(r['patient'])][r['sequence']].append(i)
        if not self.groups or not 0<hcp_cap<=1:
            raise ValueError('empty sampler or invalid source cap')
        names=sorted(self.groups)
        if names==['hcp'] and hcp_cap<1:
            raise ValueError('non-HCP patients required for the source cap')
        if 'hcp' in names and len(names)>1:
            share=min(1/len(names),hcp_cap)
            self.source_probabilities={s:share if s=='hcp' else (1-share)/(len(names)-1) for s in names}
        else:
            self.source_probabilities={s:1/len(names) for s in names}

    def indices(self,mode,draw,batch_size):
        if mode not in ('2d','3d') or type(draw) is not int or draw<0 or type(batch_size) is not int or batch_size<1:
            raise ValueError('invalid sampler draw')
        if sum(len(p) for p in self.groups.values())<batch_size:
            raise ValueError('insufficient distinct patients for batch')
        if ('hcp' in self.groups and len(self.groups)>1
                and sum(len(p) for s,p in self.groups.items() if s!='hcp')<batch_size):
            raise ValueError('insufficient non-HCP patients to preserve the source probability cap')
        rng=np.random.default_rng([self.seed,int(mode=='3d'),draw])
        used,result=set(),[]
        for _ in range(batch_size):
            available={s:[p for p in sorted(ps) if (s,p) not in used] for s,ps in self.groups.items()}
            names=sorted(s for s,ps in available.items() if ps)
            # Separate the HCP draw before renormalizing other available sources;
            # exhausting a small disease source must not increase HCP's chance.
            if 'hcp' in names and len(self.groups)>1 and rng.random()<self.source_probabilities['hcp']:
                source='hcp'
            else:
                candidates=[s for s in names if s!='hcp'] if len(self.groups)>1 else names
                probabilities=np.array([self.source_probabilities[s] for s in candidates])
                source=candidates[int(rng.choice(len(candidates),p=probabilities/probabilities.sum()))]
            pts=available[source]
            patient=pts[int(rng.integers(len(pts)))]
            by_sequence=self.groups[source][patient]
            sequences=sorted(by_sequence)
            choices=by_sequence[sequences[int(rng.integers(len(sequences)))]]
            result.append(choices[int(rng.integers(len(choices)))])
            used.add((source,patient))
        return result

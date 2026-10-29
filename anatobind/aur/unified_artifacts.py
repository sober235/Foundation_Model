"""Verify selected formal weights against real, immutable completion boundaries."""
from pathlib import Path
import hashlib
import io

import torch
from collections import Counter
import random
import numpy as np

from anatobind.aur.data_contract import file_sha256, validate_private_output
from anatobind.aur.unified_train import _state_hash


def _checkpoint(certificate, kind):
    value = certificate[f'{kind}_checkpoint_path']
    path = validate_private_output(value)
    if not isinstance(value, str) or str(path) != value or not path.is_file():
        raise ValueError('invalid completion artifact')
    contents=path.read_bytes()
    if hashlib.sha256(contents).hexdigest() != certificate[f'{kind}_checkpoint_sha256']:
        raise ValueError('invalid completion artifact')
    return torch.load(io.BytesIO(contents), map_location='cpu', weights_only=True)


def verify_completion_certificate(initializer, *, data_contract):
    """Return False on any mismatch, without exposing private artifact details.

    A selected export keeps its actual step and completeness flag. A separate
    certificate proves the phase completed and binds the selection to a real
    earlier checkpoint, rather than declaring the best step to be the last.
    """
    try:
        certificate = initializer['completion_certificate']
        execution_id=certificate['execution_id']
        if (not isinstance(execution_id,str) or len(execution_id)!=64
                or initializer['execution_id']!=execution_id):
            return False
        if (certificate['version'] != 1 or certificate['purpose'] != 'formal'
                or initializer['purpose'] != 'formal' or initializer['format_version'] != 2
                or initializer['stage'] not in ('I', 'II', 'III')
                or certificate['stage'] != initializer['stage']
                or certificate['data_contract'] != data_contract or initializer['data_contract'] != data_contract):
            return False
        budget, executed, selected_step = (certificate[k] for k in ('budget_steps', 'executed_steps', 'selected_step'))
        if (any(type(v) is not int for v in (budget, executed, selected_step))
                or not 0 < selected_step <= executed == budget or initializer['step'] != selected_step
                or initializer['complete'] is not (selected_step == budget)):
            return False
        final, selected = _checkpoint(certificate, 'final'), _checkpoint(certificate, 'selected')
        for checkpoint in (final, selected):
            if (checkpoint['format_version'] != 2 or checkpoint['purpose'] != 'formal'
                    or checkpoint['stage'] != initializer['stage'] or checkpoint['data_contract'] != data_contract
                    or checkpoint['model_config'] != initializer['model_config']
                    or checkpoint['cfg']['model'] != initializer['model_config']
                    or checkpoint['cfg']['purpose'] != 'formal' or checkpoint['cfg']['steps'] != budget
                    or checkpoint['execution_id']!=execution_id or checkpoint['lineage']!=initializer['lineage']
                    or checkpoint['numerical_environment']!=initializer['numerical_environment']
                    or not _full_boundary(checkpoint)):
                return False
        if (final['step'] != budget or final['complete'] is not True
                or selected['step'] != selected_step or selected['complete'] is not (selected_step == budget)
                or selected['cfg'] != final['cfg']):
            return False
        for prefix in ('model', 'backbone'):
            expected = certificate[f'selected_{prefix}_state_sha256']
            if (_state_hash(initializer[f'{prefix}_state_dict']) != expected
                    or _state_hash(selected[f'{prefix}_state_dict']) != expected):
                return False
        return all(torch.equal(initializer['model_state_dict']['backbone.' + key], value)
                   for key, value in initializer['backbone_state_dict'].items())
    except Exception:
        return False


def _full_boundary(checkpoint):
    """Require a resumable execution boundary, rather than an export with step flags."""
    from anatobind.aur.unified_run import validate_run_config
    cfg=checkpoint['cfg']
    if cfg!=validate_run_config(cfg): return False
    step=checkpoint['step']
    if type(step) is not int or not 0<step<=cfg['steps']: return False
    whole,remainder=divmod(step,len(cfg['mode_cycle']))
    draws={m:(whole*cfg['mode_cycle'].count(m)+cfg['mode_cycle'][:remainder].count(m))*cfg['accumulation_steps']
           for m in ('2d','3d')}
    exposures={m:n*cfg['batch_size'] for m,n in draws.items()}
    if (checkpoint['cursors']!=draws or checkpoint['exposures']!=exposures
            or checkpoint['next_mode']!=cfg['mode_cycle'][remainder]
            or checkpoint['scheduler']['last_epoch']!=step
            or checkpoint['device_type'] not in ('cpu','cuda')): return False
    for mode in ('2d','3d'):
        counts=checkpoint['source_mode_exposures'][mode]
        if any(type(v) is not int or v<0 for v in counts.values()) or sum(counts.values())!=exposures[mode]: return False
    if dict(Counter(checkpoint['source_mode_exposures']['2d'])+Counter(checkpoint['source_mode_exposures']['3d']))!=checkpoint['source_exposures']: return False
    optimizer=checkpoint['optimizer']
    states=optimizer['state']
    if not isinstance(states,dict) or not states or not optimizer['param_groups']: return False
    ids=[p for group in optimizer['param_groups'] for p in group['params']]
    if len(ids)!=len(set(ids)) or not set(states)<=set(ids): return False
    optimizer_steps=[]
    for value in states.values():
        if not {'step','exp_avg','exp_avg_sq'}<=set(value): return False
        number=float(value['step'])
        if not np.isfinite(number) or number!=int(number) or not 0<number<=step: return False
        a,b=value['exp_avg'],value['exp_avg_sq']
        if not isinstance(a,torch.Tensor) or not isinstance(b,torch.Tensor) or a.shape!=b.shape or a.dtype!=b.dtype: return False
        if not torch.isfinite(a).all() or not torch.isfinite(b).all(): return False
        optimizer_steps.append(number)
    # Shared parameters receive a gradient on every completed optimizer step.
    if max(optimizer_steps)!=step: return False
    rng=checkpoint['rng']
    if not {'torch','cuda','python','numpy','points','masks'}<=set(rng): return False
    for name in ('torch','points','masks'):
        if (not isinstance(rng[name],torch.Tensor) or rng[name].dtype!=torch.uint8
                or rng[name].ndim!=1 or not rng[name].numel()): return False
    for name in ('torch','masks'):
        torch.Generator(device='cpu').set_state(rng[name])
    if checkpoint['device_type']=='cpu':
        torch.Generator(device='cpu').set_state(rng['points'])
    elif (not isinstance(rng['cuda'],list) or not rng['cuda']
            or any(not isinstance(t,torch.Tensor) or t.dtype!=torch.uint8 or t.ndim!=1 or not t.numel() for t in rng['cuda'])
            or rng['points'].numel()!=rng['cuda'][0].numel()): return False
    random.Random().setstate(rng['python'])
    n=rng['numpy']
    np.random.RandomState().set_state((n['algorithm'],np.asarray(n['keys'],dtype=np.uint32),n['pos'],n['has_gauss'],n['cached']))
    return True

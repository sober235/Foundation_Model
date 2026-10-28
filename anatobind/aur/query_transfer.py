"""Audited Stage I backbone transfer for the registered 64-to-128 expansion."""
from copy import deepcopy
from pathlib import Path
import json
import hashlib
import io
import torch
from anatobind.aur.data_contract import file_sha256, json_digest
from anatobind.aur.unified_train import _state_hash

POLICY = 'query_capacity_64_to_128_v1'
CHANGED = frozenset({'anatobind/aur/unified_data.py','anatobind/aur/planar_relations.py',
                     'anatobind/aur/unified_run.py','anatobind/aur/unified_launcher.py',
                     'anatobind/aur/query_transfer.py','anatobind/aur/unified_artifacts.py',
                     'anatobind/aur/eval.py','anatobind/aur/unified_evaluation.py'})


def current_runtime_root():
    return Path(__file__).resolve().parents[2]


def runtime_snapshot(root):
    root = Path(root).resolve()
    members = sorted((root/'anatobind').rglob('*.py'))
    members += [root/'scripts'/name for name in ('aur_unified_train.py','aur_unified_ssl_eval.py','aur_qualify_training_data.py')]
    result = {}
    for p in members:
        if not p.resolve().is_relative_to(root) or not p.is_file():
            raise ValueError('invalid source snapshot')
        result[str(p.relative_to(root))] = file_sha256(p)
    return result


def validate_transfer(parent, cfg, identity, gate, record):
    """Validate actual parent bytes and source identities before model construction."""
    try:
        required={'version','policy','parent_runtime_root','parent_sources','child_sources','parent_identity',
                  'child_identity','parent_export','g1_report','child_config_sha256'}
        if set(record)!=required or record['version']!=1 or record['policy']!=POLICY:
            raise ValueError
        old, new = record['parent_identity'], record['child_identity']
        if (old != parent['data_contract'] or new != identity or
                old['contract_sha256'] != new['contract_sha256'] or old['geometry_version'] != new['geometry_version'] or
                json_digest(cfg) != record['child_config_sha256']):
            raise ValueError
        parent_root=Path(record['parent_runtime_root']).resolve()
        if str(parent_root)!=record['parent_runtime_root']:
            raise ValueError
        source=runtime_snapshot(parent_root);target=runtime_snapshot(current_runtime_root())
        if (source!=record['parent_sources'] or target!=record['child_sources'] or
                json_digest(source)!=old['runtime_sha256'] or json_digest(target)!=new['runtime_sha256'] or
                any(source.get(k)!=target.get(k) for k in set(source)|set(target) if k not in CHANGED)):
            raise ValueError
        old_model=deepcopy(parent['model_config']);new_model=deepcopy(cfg['model'])
        if old_model.pop('n_events',None)!=64 or new_model.pop('n_events',None)!=128 or old_model!=new_model:
            raise ValueError
        buffers={}
        for name in ('parent_export','g1_report'):
            ref=record[name]
            if set(ref)!={'path','sha256'} or str(Path(ref['path']).resolve())!=ref['path']:
                raise ValueError
            buffers[name]=Path(ref['path']).read_bytes()
            if hashlib.sha256(buffers[name]).hexdigest()!=ref['sha256']:
                raise ValueError
        saved=torch.load(io.BytesIO(buffers['parent_export']),map_location='cpu',weights_only=True)
        if (saved.get('stage')!='I' or saved.get('purpose')!='formal' or saved.get('format_version')!=2 or
                saved.get('data_contract')!=old or saved.get('model_config')!=parent['model_config'] or
                saved.get('completion_certificate')!=parent['completion_certificate'] or
                _state_hash(saved['backbone_state_dict'])!=_state_hash(parent['backbone_state_dict']) or
                _state_hash(saved['model_state_dict'])!=_state_hash(parent['model_state_dict'])):
            raise ValueError
        if json.loads(buffers['g1_report'])!=gate:
            raise ValueError
        from anatobind.aur.unified_artifacts import verify_completion_certificate
        from anatobind.aur.ssl.unified_eval import validate_gate
        cert=parent['completion_certificate']
        if (not verify_completion_certificate(parent,data_contract=old) or
                not validate_gate(gate,data_contract_sha256=old['contract_sha256'],
                                  backbone_state_sha256=_state_hash(parent['backbone_state_dict']),require_formal=True) or
                gate.get('checkpoint_step')!=cert['selected_step'] or
                gate.get('checkpoint_sha256') not in (cert['selected_checkpoint_sha256'],record['parent_export']['sha256'])):
            raise ValueError
        return {'kind':POLICY,'parent_export_sha256':record['parent_export']['sha256'],
                'g1_report_sha256':record['g1_report']['sha256'],'parent_identity':deepcopy(old),
                'child_identity':deepcopy(new),'child_config_sha256':record['child_config_sha256'],
                'transfer_record_sha256':json_digest(record),'backbone_state_sha256':_state_hash(parent['backbone_state_dict'])}
    except Exception:
        raise ValueError('invalid registered query capacity transfer') from None

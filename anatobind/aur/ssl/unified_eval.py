"""Frozen dual-input G1 engineering probes against pseudo-labels, NOT_EVIDENCE.

Both arms consume exactly the same privately frozen observations. Only linear
or logistic readouts train; no backbone parameter or gradient changes. The
legacy evaluator supplies the readout, Dice, AUC and bootstrap primitives.
"""
from collections import Counter
from copy import deepcopy
import math
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from anatobind.aur.data_contract import (file_sha256, read_contract, validate_private_output,
                                          write_private_json)
from anatobind.aur.labels import N_HOSTS, entity_to_synthseg, host_map
from anatobind.aur.ssl import eval as E
from anatobind.aur.swin import SwinBackbone
from anatobind.aur.unified_data import load_unified_volume, make_unified_crop
from anatobind.aur.unified_train import _BACKBONE_KEYS, _state_hash

VERSION = 'dual_g1_v1'
DEFAULT_RECIPE = {'crops': {'2d': [1, 160, 160], '3d': [128, 160, 160]},
                  'n_calibration': 96, 'n_validation': 48, 'n_lesion': 48,
                  'seeds': [0, 1, 2], 'readout_epochs': 300, 'n_bootstrap': 1000,
                  'random_seed': 0, 'dice_gain': .05, 'auc_tolerance': .02,
                  'selection_seeds': [11, 12], 'crop_seeds': [21, 22, 23, 24]}


def _finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def report_gate(mode_metrics, *, counts_complete=True):
    """Recompute PASS/FAIL/INSUFFICIENT; missing/nonfinite AUC never passes."""
    insufficient = {'status': 'INSUFFICIENT', 'host_pass': False, 'lesion_pass': False,
                    'macro_dice_gain': None, 'gain_ci_low_min': None,
                    'lesion_auc_ssl': None, 'lesion_auc_random': None,
                    'evidence': 'NOT_EVIDENCE'}
    try:
        arms = mode_metrics['arms']
        if set(arms)!={'ssl','random'}:
            return insufficient
        for name in ('ssl', 'random'):
            arm = arms[name]
            values = [arm['host']['macro_dice_mean'], arm['lesion']['auc_mean']]
            host_seeds, lesion_seeds = arm['host']['seeds'], arm['lesion']['seeds']
            if not host_seeds or len(host_seeds) != len(lesion_seeds):
                return insufficient
            values += [entry['macro_dice'] for entry in host_seeds]
            values += [entry['auc'] for entry in lesion_seeds]
            if not all(_finite(v) and 0 <= v <= 1 for v in values):
                return insufficient
            if (not math.isclose(arm['host']['macro_dice_mean'],sum(e['macro_dice'] for e in host_seeds)/len(host_seeds),abs_tol=1e-12,rel_tol=0)
                    or not math.isclose(arm['lesion']['auc_mean'],sum(e['auc'] for e in lesion_seeds)/len(lesion_seeds),abs_tol=1e-12,rel_tol=0)):
                return insufficient
            if any(type(entry[k]) is not int or entry[k] <= 0 for entry in lesion_seeds
                   for k in ('n_train_positive', 'n_train_negative', 'n_val_positive', 'n_val_negative')):
                return insufficient
        intervals = mode_metrics['gain_bootstrap_by_seed']
        if not counts_complete or len(intervals) != len(arms['ssl']['host']['seeds']):
            return insufficient
        if any(len(entry['ci95']) != 2 or not all(_finite(v) for v in entry['ci95'])
               or entry['ci95'][0] > entry['ci95'][1] for entry in intervals):
            return insufficient
        gain = arms['ssl']['host']['macro_dice_mean'] - arms['random']['host']['macro_dice_mean']
        low = min(entry['ci95'][0] for entry in intervals)
        ssl, random = arms['ssl']['lesion']['auc_mean'], arms['random']['lesion']['auc_mean']
        host_pass = gain >= .05 and low > 0
        lesion_pass = ssl >= random - .02
        return {'status': 'PASS' if host_pass and lesion_pass else 'FAIL',
                'host_pass': bool(host_pass), 'lesion_pass': bool(lesion_pass),
                'macro_dice_gain': float(gain), 'gain_ci_low_min': float(low),
                'lesion_auc_ssl': float(ssl), 'lesion_auc_random': float(random),
                'evidence': 'NOT_EVIDENCE'}
    except (KeyError, TypeError, ValueError):
        return insufficient


def validate_gate(report, *, data_contract_sha256, backbone_state_sha256, require_formal=True):
    """Boolean stage-transition validator bound to exact data and backbone hashes."""
    try:
        if (report['version'] != VERSION or report['data_contract_sha256'] != data_contract_sha256
                or report['backbone_state_sha256'] != backbone_state_sha256
                or set(report['modes']) != {'2d', '3d'}):
            return False
        if require_formal and (report['recipe'] != DEFAULT_RECIPE or not report['default_recipe_match']):
            return False
        recipe = report['recipe']
        expected = {'host_calibration': recipe['n_calibration'], 'host_validation': recipe['n_validation'],
                    'lesion_calibration': recipe['n_lesion'], 'lesion_validation': recipe['n_lesion']}
        for mode in report['modes'].values():
            supported = (mode['counts'] == expected
                         and len(mode['support']['host_validation_cells']) == N_HOSTS
                         and all(type(v) is int and v > 0 for v in mode['support']['host_validation_cells']))
            if not supported or mode['status'] != 'PASS' or report_gate(mode)['status'] != 'PASS':
                return False
            seeds = recipe['seeds']
            for arm in mode['arms'].values():
                if any([entry['seed'] for entry in arm[probe]['seeds']] != seeds for probe in ('host', 'lesion')):
                    return False
            if [entry['seed'] for entry in mode['gain_bootstrap_by_seed']] != seeds:
                return False
        return True
    except (KeyError, TypeError, ValueError):
        return False


def equivalent_diameter(measure, input_mode):
    """Area-derived circle diameter in 2D; volume-derived sphere in 3D."""
    if not _finite(measure) or measure < 0 or input_mode not in ('2d', '3d'):
        raise ValueError('Invalid mode-specific lesion measure')
    return 2 * math.sqrt(measure / math.pi) if input_mode == '2d' else (6 * measure / math.pi) ** (1 / 3)


def size_bin(diameter):
    return '<5' if diameter < 5 else '5-10' if diameter <= 10 else '>10'


def _group(row):
    return str(row['source']), str(row['patient'])


def select_rows(contract, n_calibration, n_validation, seed, lesion_only=False):
    """Pick one sequence per source-qualified patient, without split leakage."""
    train, validation = contract['train_supervised'], contract['validation']
    if ({_group(r) for r in train} & {_group(r) for r in validation}
            or any(r.get('split') != 'train' for r in train)
            or any(r.get('split') != 'val' for r in validation)):
        raise ValueError('G1 patient roles are inconsistent')
    rng = np.random.default_rng(seed)
    def pick(rows, count):
        groups = {}
        for row in rows:
            if row.get('a_supervised', False) and (not lesion_only or row.get('u_supervised', False)):
                groups.setdefault(_group(row), []).append(row)
        keys = sorted(groups)
        chosen = rng.choice(len(keys), size=min(count, len(keys)), replace=False) if keys else []
        return [groups[keys[i]][int(rng.integers(len(groups[keys[i]])))] for i in chosen]
    return pick(train, n_calibration), pick(validation, n_validation)


@torch.no_grad()
def cell_features(backbone, crop, device):
    """Use the true mode stem; return detached F2+F3+F4 and F2 validity."""
    levels = backbone(crop['image'][None].to(device), crop['valid'][None].to(device),
                      crop['coords'][None].to(device), crop['local'][None].to(device),
                      input_mode=crop['input_mode'])
    f2 = levels[1]['feat']
    parts = [f2] + [F.interpolate(levels[k]['feat'].float(), size=f2.shape[2:],
                                   mode='trilinear', align_corners=False) for k in (2, 3)]
    return torch.cat(parts, 1)[0].half().cpu(), levels[1]['valid'][0].cpu()


def bootstrap_difference(per_crop_a, per_crop_b, rows, n_boot=1000, seed=0):
    groups = [_group(row) for row in rows]
    result = E.bootstrap_difference(per_crop_a, per_crop_b, groups, n_boot=n_boot, seed=seed)
    return {**result, 'patient_groups': len(set(groups))}


def _private_torch(path, value):
    with os.fdopen(os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600), 'wb') as stream:
        torch.save(value, stream)


def _freeze_crops(rows, shape, mode, seed, lesion, out, tag):
    crops = []
    for i, row in enumerate(rows):
        vol = load_unified_volume(row)
        crop = make_unified_crop(vol, np.random.default_rng([seed, i]), mode, shape,
                                 do_augment=False, lesion_centred=lesion)
        crop['source'] = row['source']
        crop['patient'] = row['patient']
        crops.append(crop)
        _private_torch(out / f'{tag}_{i:03d}_private.pt', crop)
    return crops


def _gather(backbone, crops, device):
    data = {'features': [], 'targets': [], 'valids': [], 'roi_targets': [],
            'roi_valids': [], 'roi_normal_valids': [], 'lesions': [], 'meta': []}
    for crop in crops:
        feat, valid = cell_features(backbone, crop, device)
        stride = E.stride_of(crop['instance'].shape, feat.shape[1:])
        if any(g * s != n for g, s, n in zip(feat.shape[1:], stride, crop['instance'].shape)):
            raise ValueError('G1 feature grid does not tile the observation')
        hosts = torch.from_numpy(host_map(entity_to_synthseg(crop['entity'].numpy())).astype(np.int64))
        onehot = F.one_hot(hosts, N_HOSTS + 1).permute(3, 0, 1, 2).float()[None]
        mask = ((crop['valid'] > .5) & ~crop['a_ignore']).float()[None, None]
        pool = lambda x: F.avg_pool3d(x, kernel_size=stride, stride=stride)[0]
        anatomy_fraction = pool(mask)[0]
        targets = pool(onehot * mask).argmax(0)
        data['features'].append(feat)
        data['targets'].append(targets)
        data['valids'].append((valid > .5) & (anatomy_fraction >= .5))
        data['roi_targets'].append(pool(onehot).argmax(0))
        data['roi_valids'].append(valid > .5)
        # U positives intentionally overlap A-ignore. They retain feature
        # validity, while normal controls exclude every ignored source voxel,
        # including rejected/small abnormalities absent from instance targets.
        ignored = F.max_pool3d(crop['a_ignore'].float()[None, None],
                              kernel_size=stride, stride=stride)[0, 0] > 0
        data['roi_normal_valids'].append((valid > .5) & ~ignored)
        data['lesions'].append(pool((crop['instance'] > 0).float()[None, None])[0])
        data['meta'].append(crop)
    return data


def _roi_pairs(data, seed):
    rng = np.random.default_rng(seed)
    xs, ys, diameters, support = [], [], [], []
    for feat, les, target, valid, normal_valid, crop in zip(data['features'], data['lesions'],
            data['roi_targets'], data['roi_valids'], data['roi_normal_valids'], data['meta']):
        inst = crop['instance']
        stride = E.stride_of(inst.shape, feat.shape[1:])
        flat = feat.permute(1, 2, 3, 0).reshape(-1, feat.shape[0]).float()
        for k in range(1, int(inst.max()) + 1):
            mask = inst == k
            cells = F.max_pool3d(mask.float()[None, None], kernel_size=stride, stride=stride)[0, 0] > 0
            cells &= valid
            if not cells.any():
                continue
            host = int(torch.mode(target[cells]).values)
            normal = (target == host) & normal_valid & (les == 0) & ~cells
            n = int(cells.sum())
            idx = torch.nonzero(normal.flatten()).flatten()
            if len(idx) < n:
                continue
            take = idx[torch.as_tensor(rng.choice(len(idx), size=n, replace=False))]
            xs.extend([flat[cells.flatten()].mean(0), flat[take].mean(0)])
            ys.extend([1, 0])
            unit = crop['pixel_area_mm2'] if crop['input_mode'] == '2d' else crop['voxel_volume_mm3']
            diameter = equivalent_diameter(float(mask.sum()) * unit, crop['input_mode'])
            diameters.extend([diameter, diameter])
            support.extend([{'source': crop['source'], 'host_map_class': host, 'size_bin': size_bin(diameter)}] * 2)
    channels = data['features'][0].shape[0] if data['features'] else 0
    pairs = (torch.stack(xs) if xs else torch.zeros(0, channels),
             torch.tensor(ys, dtype=torch.long), np.asarray(diameters))
    return pairs, support


def _lesion_probe(cal, val, seed):
    from sklearn.linear_model import LogisticRegression
    xc, yc, _ = cal
    xv, yv, diameters = val
    result = {'auc': None, 'n_train_samples': len(yc), 'n_val_samples': len(yv),
              'n_train_positive': int((yc == 1).sum()), 'n_train_negative': int((yc == 0).sum()),
              'n_val_positive': int((yv == 1).sum()), 'n_val_negative': int((yv == 0).sum()), 'by_size': {}}
    scores = None
    if (len(yc) >= 4 and len(yv) >= 2 and len(set(yc.tolist())) == 2
            and torch.isfinite(xc).all() and torch.isfinite(xv).all()):
        mean, std = xc.mean(0, keepdim=True), xc.std(0, keepdim=True).clamp(min=1e-3)
        clf = LogisticRegression(C=1.0, max_iter=2000, random_state=seed)
        clf.fit(((xc - mean) / std).numpy(), yc.numpy())
        scores = clf.decision_function(((xv - mean) / std).numpy())
        result['auc'] = E.auc(scores, yv.numpy())
    for name in ('<5', '5-10', '>10'):
        selected = np.array([size_bin(d) == name for d in diameters], dtype=bool)
        pos, neg = int((yv[selected] == 1).sum()), int((yv[selected] == 0).sum())
        result['by_size'][name] = {'auc': E.auc(scores[selected], yv.numpy()[selected])
                                 if scores is not None and selected.any() else None,
                                 'n_samples': int(selected.sum()), 'n_positive': pos,
                                 'n_negative': neg, 'n_pairs': min(pos, neg)}
    return result


def _clean(value):
    if isinstance(value, dict): return {key: _clean(v) for key, v in value.items()}
    if isinstance(value, (tuple, list)): return [_clean(v) for v in value]
    if isinstance(value, np.generic): return _clean(value.item())
    if isinstance(value, float) and not math.isfinite(value): return None
    return value


def _mode_report(backbones, shared, rows, recipe, device, out, mode):
    counts = {name: len(shared[key]) for name, key in (('host_calibration', 'cal'),
              ('host_validation', 'val'), ('lesion_calibration', 'lcal'), ('lesion_validation', 'lval'))}
    report = {'counts': counts, 'arms': {}, 'gain_bootstrap_by_seed': [],
              'support': {'host_validation_cells': [0] * N_HOSTS, 'by_source': {}, 'lesion_pairs': {}}}
    if not all(counts.values()):
        report.update(report_gate(report, counts_complete=False))
        return report
    per_crop = {}
    for name, backbone in backbones.items():
        data = {key: _gather(backbone, crops, device) for key, crops in shared.items()}
        _private_torch(out / f'{mode}_{name}_features_private.pt', data)
        cal, val = data['cal'], data['val']
        if not all(any(v.any() for v in part['valids']) for part in (cal, val)):
            report.update(report_gate(report, counts_complete=False))
            return report
        if name == 'ssl':
            cells = sum(torch.bincount(t[v], minlength=N_HOSTS + 1)
                        for t, v in zip(val['targets'], val['valids']))
            report['support']['host_validation_cells'] = cells[1:].tolist()
            for row, target, valid in zip(rows['val'], val['targets'], val['valids']):
                source = row['source']
                item = report['support']['by_source'].setdefault(source, {'host_validation_crops': 0,
                    'host_cells': [0] * N_HOSTS})
                item['host_validation_crops'] += 1
                local = torch.bincount(target[valid], minlength=N_HOSTS + 1)[1:].tolist()
                item['host_cells'] = [a + b for a, b in zip(item['host_cells'], local)]
        arm = {'host': {'seeds': []}, 'lesion': {'seeds': []}}
        for seed in recipe['seeds']:
            # The legacy linear constructor consumes CPU RNG before its
            # explicit probe initialization. Preserve the training caller's
            # generator and avoid touching any unrelated visible CUDA device.
            with torch.random.fork_rng(devices=[]):
                head = E.train_readout(cal['features'], cal['targets'], cal['valids'], seed,
                                       device, epochs=recipe['readout_epochs'])
            pc = E.score_readout(head, val['features'], val['targets'], val['valids'])
            dice, macro = E.dice_from_counts(pc)
            arm['host']['seeds'].append({'seed': seed, 'macro_dice': macro,
                                         'per_host_dice': dice.tolist()})
            per_crop[(name, seed)] = pc
            lcal, lcal_support = _roi_pairs(data['lcal'], seed)
            lval, lval_support = _roi_pairs(data['lval'], seed)
            arm['lesion']['seeds'].append({'seed': seed, **_lesion_probe(lcal, lval, seed)})
            if name == 'ssl':
                for split, support in (('calibration', lcal_support), ('validation', lval_support)):
                    report['support']['lesion_pairs'][f'{split}_seed_{seed}'] = {
                        'by_source': dict(Counter(r['source'] for r in support)),
                        'by_host_map_class': {str(k): v for k, v in Counter(r['host_map_class'] for r in support).items()},
                        'by_size': {key: sum(r['size_bin'] == key for r in support) for key in ('<5', '5-10', '>10')}}
        arm['host']['macro_dice_mean'] = float(np.mean([r['macro_dice'] for r in arm['host']['seeds']]))
        aucs = [r['auc'] for r in arm['lesion']['seeds']]
        arm['lesion']['auc_mean'] = float(np.mean(aucs)) if all(_finite(v) for v in aucs) else None
        report['arms'][name] = arm
    report['gain_bootstrap_by_seed'] = [dict(seed=seed, **bootstrap_difference(
        per_crop[('ssl', seed)], per_crop[('random', seed)], rows['val'],
        n_boot=recipe['n_bootstrap'], seed=seed)) for seed in recipe['seeds']]
    expected = {'host_calibration': recipe['n_calibration'], 'host_validation': recipe['n_validation'],
                'lesion_calibration': recipe['n_lesion'], 'lesion_validation': recipe['n_lesion']}
    complete = counts == expected and all(report['support']['host_validation_cells'])
    report.update(report_gate(report, counts_complete=complete))
    return report


def run(contract, checkpoint, out_dir, device='cpu', crops=None, model_config=None,
        n_calibration=96, n_validation=48, n_lesion=48, seeds=(0, 1, 2),
        readout_epochs=300, n_bootstrap=1000, random_seed=0):
    """Write one immutable private dual G1 report, from a verified data contract.

    Smaller explicitly supplied recipes support engineering tests; their report
    records actual counts and cannot satisfy require_formal stage transitions.
    Trust is limited to project-produced Stage I format-version-2 checkpoints.
    """
    out = validate_private_output(out_dir)
    if out.exists():
        raise ValueError('G1 output already exists; choose a new private version')
    recipe = deepcopy(DEFAULT_RECIPE)
    recipe.update(crops={mode: list(shape) for mode, shape in (crops or recipe['crops']).items()},
                  n_calibration=n_calibration, n_validation=n_validation, n_lesion=n_lesion,
                  seeds=list(seeds), readout_epochs=readout_epochs, n_bootstrap=n_bootstrap,
                  random_seed=random_seed)
    if (set(recipe['crops']) != {'2d', '3d'} or recipe['crops']['2d'][0] != 1
            or any(type(recipe[key]) is not int or recipe[key] <= 0 for key in
                   ('n_calibration', 'n_validation', 'n_lesion', 'readout_epochs', 'n_bootstrap'))
            or not recipe['seeds'] or len(set(recipe['seeds'])) != len(recipe['seeds'])
            or any(type(seed) is not int or seed < 0 for seed in [random_seed, *recipe['seeds']])):
        raise ValueError('Invalid dual G1 recipe')
    data = read_contract(contract, verify=True)
    payload = torch.load(checkpoint, map_location='cpu', weights_only=True)
    if payload.get('format_version') != 2 or payload.get('stage') != 'I':
        raise ValueError('G1 requires a unified Stage I checkpoint')
    cfg = payload['cfg']['model']
    if model_config is not None and any(model_config.get(k) != cfg.get(k) for k in _BACKBONE_KEYS):
        raise ValueError('G1 model configuration differs from checkpoint')
    kwargs = {key: cfg[key] for key in _BACKBONE_KEYS}
    if kwargs['dual_input'] is not True:
        raise ValueError('G1 requires a dual-input backbone')
    state = payload['backbone_state_dict']
    embedded = {key[len('backbone.'):]: value for key, value in payload['model_state_dict'].items()
                if key.startswith('backbone.')}
    if _state_hash(state) != _state_hash(embedded):
        raise ValueError('G1 checkpoint backbone commitments differ')
    with torch.random.fork_rng(devices=[]):
        ssl = SwinBackbone(**kwargs)
        ssl.load_state_dict(state, strict=True)
        torch.random.default_generator.manual_seed(random_seed)
        random = SwinBackbone(**kwargs)
    backbones = {'ssl': ssl, 'random': random}
    for bb in backbones.values():
        bb.to(device).eval()
        bb.requires_grad_(False)
    states_before = {name: _state_hash(bb.state_dict()) for name, bb in backbones.items()}
    cal, val = select_rows(data, n_calibration, n_validation, 11)
    lcal, lval = select_rows(data, n_lesion, n_lesion, 12, lesion_only=True)
    selections = {'cal': cal, 'val': val, 'lcal': lcal, 'lval': lval}
    # Freeze membership, physical recipe and exact checkpoint before dense reads.
    write_private_json(out / 'selection_private.json', selections)
    write_private_json(out / 'frozen.json', {'version': VERSION, 'recipe': recipe,
        'data_contract_sha256': data['contract_sha256'], 'backbone_state_sha256': states_before['ssl'],
        'checkpoint_sha256': file_sha256(checkpoint), 'evidence': 'NOT_EVIDENCE'})
    report = {'version': VERSION, 'data_contract_sha256': data['contract_sha256'],
              'backbone_state_sha256': states_before['ssl'], 'checkpoint_sha256': file_sha256(checkpoint),
              'checkpoint_step': payload['step'], 'recipe': recipe,
              'default_recipe_match': recipe == DEFAULT_RECIPE, 'modes': {}, 'evidence': 'NOT_EVIDENCE'}
    for mode in ('2d', '3d'):
        shared = {key: _freeze_crops(rows, recipe['crops'][mode], mode, seed,
                                   key.startswith('l'), out, mode + '_' + key)
                  for (key, rows), seed in zip(selections.items(), recipe['crop_seeds'])}
        report['modes'][mode] = _mode_report(backbones, shared, selections, recipe, device, out, mode)
    if any(_state_hash(bb.state_dict()) != states_before[name]
           or any(p.grad is not None for p in bb.parameters()) for name, bb in backbones.items()):
        raise RuntimeError('G1 frozen backbone state changed')
    # Recheck all committed inputs after collection; no integrity shortcut for
    # developer validation. Changes leave the frozen/partial evidence intact.
    if read_contract(contract, verify=True)['contract_sha256'] != data['contract_sha256']:
        raise ValueError('G1 data contract changed')
    if file_sha256(checkpoint) != report['checkpoint_sha256']:
        raise ValueError('G1 checkpoint changed')
    report = _clean(report)
    write_private_json(out / 'g1_report.json', report)
    write_private_json(out / 'complete.json', {'version': VERSION,
        'report_sha256': file_sha256(out / 'g1_report.json'), 'frozen_backbones_unchanged': True,
        'pass': validate_gate(report, data_contract_sha256=data['contract_sha256'],
                              backbone_state_sha256=states_before['ssl'])})
    return report

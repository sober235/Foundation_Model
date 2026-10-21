"""Dual-input G1 must never promote missing support or NaN measurements."""
from copy import deepcopy
import importlib
import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch

from anatobind.aur.swin import SwinBackbone


def api():
    try:
        return importlib.import_module('anatobind.aur.ssl.unified_eval')
    except ModuleNotFoundError:
        pytest.fail('dual-mode frozen G1 evaluator is missing')


def metrics():
    arms = {}
    for name, macro, auc in [('ssl', .7, .8), ('random', .6, .81)]:
        arms[name] = {'host': {'macro_dice_mean': macro,
            'seeds': [{'seed': s, 'macro_dice': macro} for s in (0, 1, 2)]},
            'lesion': {'auc_mean': auc, 'seeds': [{'seed': s, 'auc': auc,
                'n_train_positive': 2, 'n_train_negative': 2,
                'n_val_positive': 1, 'n_val_negative': 1} for s in (0, 1, 2)]}}
    return {'arms': arms, 'gain_bootstrap_by_seed': [{'seed': s, 'ci95': [.01, .2]} for s in (0, 1, 2)]}


def report():
    e = api()
    mode = {'counts': {'host_calibration': 96, 'host_validation': 48,
                       'lesion_calibration': 48, 'lesion_validation': 48},
            'support': {'host_validation_cells': [2] * 13}, **metrics()}
    mode.update(e.report_gate(mode))
    return {'version': 'dual_g1_v1', 'data_contract_sha256': '1' * 64,
            'backbone_state_sha256': '2' * 64, 'recipe': deepcopy(e.DEFAULT_RECIPE),
            'default_recipe_match': True, 'modes': {'2d': deepcopy(mode), '3d': deepcopy(mode)}}


def test_supported_finite_metrics_can_pass_both_frozen_conditions():
    gate = api().report_gate(metrics())
    assert gate['status'] == 'PASS'
    assert gate['host_pass'] and gate['lesion_pass']
    assert gate['macro_dice_gain'] == pytest.approx(.1)


@pytest.mark.parametrize('field', ['auc_mean', 'seed_auc', 'positive_support', 'negative_support', 'host_macro', 'ci'])
@pytest.mark.parametrize('arm', ['ssl', 'random'])
def test_nonfinite_and_missing_auc_or_host_support_are_insufficient(field, arm):
    m = metrics()
    if field == 'auc_mean': m['arms'][arm]['lesion']['auc_mean'] = float('nan')
    elif field == 'seed_auc': m['arms'][arm]['lesion']['seeds'][0]['auc'] = float('nan')
    elif field == 'positive_support': m['arms'][arm]['lesion']['seeds'][0]['n_val_positive'] = 0
    elif field == 'negative_support': m['arms'][arm]['lesion']['seeds'][0]['n_val_negative'] = 0
    elif field == 'host_macro': m['arms'][arm]['host']['macro_dice_mean'] = None
    else: m['gain_bootstrap_by_seed'][0]['ci95'][0] = float('nan')
    assert api().report_gate(m)['status'] == 'INSUFFICIENT'


@pytest.mark.parametrize('change', ['gain', 'ci', 'auc'])
def test_finite_underperforming_metrics_fail(change):
    m = metrics()
    if change == 'gain':
        m['arms']['ssl']['host']['macro_dice_mean'] = .64
        for r in m['arms']['ssl']['host']['seeds']: r['macro_dice']=.64
    elif change == 'ci': m['gain_bootstrap_by_seed'][0]['ci95'][0] = 0.0
    else:
        m['arms']['ssl']['lesion']['auc_mean'] = .78
        for r in m['arms']['ssl']['lesion']['seeds']: r['auc']=.78
    assert api().report_gate(m)['status'] == 'FAIL'


def test_gate_thresholds_are_inclusive_for_gain_and_auc_but_strict_for_bootstrap():
    m = metrics()
    m['arms']['ssl']['host']['macro_dice_mean'] = .75
    m['arms']['random']['host']['macro_dice_mean'] = .70
    m['arms']['ssl']['lesion']['auc_mean'] = .79
    for name in ('ssl','random'):
        for r in m['arms'][name]['host']['seeds']: r['macro_dice']=m['arms'][name]['host']['macro_dice_mean']
        for r in m['arms'][name]['lesion']['seeds']: r['auc']=m['arms'][name]['lesion']['auc_mean']
    assert api().report_gate(m)['status'] == 'PASS'


@pytest.mark.parametrize('probe,field',[('host','macro_dice'),('lesion','auc')])
def test_seed_measurements_must_agree_with_declared_aggregate(probe,field):
    m=metrics()
    m['arms']['ssl'][probe]['seeds'][0][field]=.1
    assert api().report_gate(m)['status']=='INSUFFICIENT'


@pytest.mark.parametrize('change', ['mode', 'hash', 'counts', 'recipe', 'status', 'host_support', 'forged_pass'])
def test_transition_checks_recompute_the_gate_and_verify_provenance(change):
    e = api(); r = report()
    if change == 'mode': del r['modes']['2d']
    elif change == 'hash': r['backbone_state_sha256'] = '3' * 64
    elif change == 'counts': r['modes']['2d']['counts']['host_validation'] = 2
    elif change == 'recipe': r['recipe']['readout_epochs'] = 2
    elif change == 'status': r['modes']['2d']['status'] = 'INSUFFICIENT'
    elif change == 'host_support': r['modes']['2d']['support']['host_validation_cells'][0] = 0
    else: r['modes']['2d']['arms']['random']['lesion']['auc_mean'] = None
    assert not e.validate_gate(r, data_contract_sha256='1' * 64, backbone_state_sha256='2' * 64)


def test_transition_accepts_only_a_complete_supported_dual_report():
    assert api().validate_gate(report(), data_contract_sha256='1' * 64, backbone_state_sha256='2' * 64)


def test_planar_lesion_size_uses_area_and_volumetric_size_uses_volume():
    e = api()
    assert e.equivalent_diameter(4 * math.pi, '2d') == pytest.approx(4.)
    assert e.equivalent_diameter(4 * math.pi, '3d') == pytest.approx(24 ** (1 / 3))
    assert [e.size_bin(x) for x in (4.99, 5., 10., 10.01)] == ['<5', '5-10', '5-10', '>10']


def test_source_qualified_patient_bootstrap_does_not_merge_same_named_patients():
    e = api()
    a = [(torch.ones(13), torch.ones(13), torch.ones(13)),
         (torch.zeros(13), torch.ones(13), torch.ones(13))]
    b = [(torch.zeros(13), torch.ones(13), torch.ones(13))] * 2
    rows = [{'source': 'one', 'patient': 'same'}, {'source': 'two', 'patient': 'same'}]
    out = e.bootstrap_difference(a, b, rows, n_boot=100, seed=0)
    assert out['patient_groups'] == 2
    assert out['ci95'] == [0., 1.]


def test_cell_features_uses_actual_planar_stem_and_has_no_backbone_gradients():
    e = api()
    bb = SwinBackbone(embed=32, depths=(1, 1, 1, 1), heads=(1, 2, 4, 8),
                      window=(2, 2, 2), dual_input=True)
    for mode, shape in [('2d', (1, 16, 16)), ('3d', (4, 16, 16))]:
        c = {'image': torch.zeros(1, *shape), 'valid': torch.ones(shape),
             'coords': torch.zeros(3, *shape), 'local': torch.zeros(3, *shape), 'input_mode': mode}
        f, v = e.cell_features(bb, c, 'cpu')
        assert f.shape[1:] == v.shape
        assert f.shape[1] == 1
        assert not f.requires_grad
    assert all(p.grad is None for p in bb.parameters())


def test_selection_is_disjoint_and_uses_one_source_qualified_patient_observation():
    rows = [{'source': s, 'patient': p, 'case': p, 'sequence': q,
             'source_sequence': q, 'split': split, 'a_supervised': True, 'u_supervised': True}
            for split, p in [('train', 'p'), ('val', 'v')]
            for s in ('a', 'b') for q in ('T1', 'T2')]
    selected = api().select_rows({'train_supervised': rows[:0] + [r for r in rows if r['split'] == 'train'],
                                  'validation': [r for r in rows if r['split'] == 'val']}, 96, 48, seed=11)
    assert [len(part) for part in selected] == [2, 2]
    conflict = {'train_supervised': [rows[0]], 'validation': [dict(rows[0], split='val')]}
    with pytest.raises(ValueError, match='patient'):
        api().select_rows(conflict, 1, 1, seed=0)


def test_cli_redacts_invalid_argument_values(capsys):
    import runpy
    try:
        module = runpy.run_path(str(Path(__file__).resolve().parents[1] / 'scripts/aur_unified_ssl_eval.py'))
    except FileNotFoundError:
        pytest.fail('dual-mode G1 CLI is missing')
    with pytest.raises(SystemExit):
        module['main'](['--contract', 'private-value', '--checkpoint', 'private-value',
                        '--out', 'private-value', '--calibration', 'private-value'])
    assert 'private-value' not in capsys.readouterr().err


def test_missing_observations_write_insufficient_report_without_promoting(tmp_path, monkeypatch):
    from anatobind.aur.data_contract import private_output
    from anatobind.aur.unified_train import _state_hash
    e = api()
    config = dict(embed=32, depths=[1] * 4, heads=[1, 2, 4, 8], window=[2] * 3,
                  patch=[2, 4, 4], dual_input=True, use_checkpoint=False)
    bb = SwinBackbone(**config)
    state = bb.state_dict()
    checkpoint = tmp_path / 'stage_i.pt'
    torch.save({'format_version': 2, 'stage': 'I', 'cfg': {'model': config, 'ssl': {}},
                'step': 1, 'backbone_state_dict': state,
                'model_state_dict': {'backbone.' + k: v for k, v in state.items()}}, checkpoint)
    monkeypatch.setattr(e, 'read_contract', lambda *a, **kw: {'train_supervised': [], 'validation': [],
        'contract_sha256': '1' * 64})
    monkeypatch.setattr(e, 'validate_private_output', lambda p: private_output(p, private_root=tmp_path))
    result = e.run(tmp_path / 'contract.json', checkpoint, tmp_path / 'g1',
                   crops={'2d': (1, 16, 16), '3d': (4, 16, 16)},
                   n_calibration=2, n_validation=2, n_lesion=2, seeds=(0,),
                   readout_epochs=1, n_bootstrap=5)
    assert result['version'] == 'dual_g1_v1'
    assert result['backbone_state_sha256'] == _state_hash(state)
    assert result['default_recipe_match'] is False
    assert {m['status'] for m in result['modes'].values()} == {'INSUFFICIENT'}
    saved = json.loads((tmp_path / 'g1/g1_report.json').read_text())
    assert saved == result
    assert (tmp_path / 'g1').stat().st_mode & 0o777 == 0o700
    assert (tmp_path / 'g1/g1_report.json').stat().st_mode & 0o777 == 0o600


def test_logistic_probe_includes_exact_ten_mm_in_the_middle_size_bin():
    x = torch.tensor([[-2.], [2.], [-1.], [1.]])
    y = torch.tensor([0, 1, 0, 1])
    cal = (x, y, np.full(4, 10.))
    val = (x[:2], y[:2], np.full(2, 10.))
    out = api()._lesion_probe(cal, val, seed=0)
    assert out['by_size']['5-10']['n_samples'] == 2
    assert out['by_size']['5-10']['n_pairs'] == 1
    assert out['by_size']['>10']['n_samples'] == 0


def test_roi_normal_cells_exclude_every_a_ignore_voxel_but_keep_positive_lesions(monkeypatch):
    from anatobind.aur.labels import entity_map
    e = api()
    shape = (1, 2, 6)
    instance = torch.zeros(shape, dtype=torch.long)
    instance[:, :, :2] = 1
    ignore = instance.bool()
    ignore[0, 0, 2] = True  # A partially ignored cell is also not normal tissue.
    crop = {'instance': instance, 'a_ignore': ignore, 'valid': torch.ones(shape),
            'entity': torch.from_numpy(entity_map(np.full(shape, 2, np.int16))),
            'source': 'synthetic', 'patient': 'synthetic', 'input_mode': '2d',
            'pixel_area_mm2': 1.}
    features = torch.tensor([[[[10., 50., 20.]]]])
    monkeypatch.setattr(e, 'cell_features', lambda *args: (features, torch.ones(1, 1, 3)))
    data = e._gather(None, [crop], 'cpu')
    assert not data['valids'][0][0, 0, 0]  # A readout cannot use the ignored U cell.
    for seed in range(8):
        (x, y, _), support = e._roi_pairs(data, seed)
        assert y.tolist() == [1, 0]
        assert x[:, 0].tolist() == [10., 20.]
        assert len(support) == 2


def test_missing_anatomy_qualification_is_not_selected_for_any_g1_probe():
    e = api()
    rows = [{'source': 'synthetic', 'patient': 'synthetic', 'split': 'train',
             'u_supervised': True}]
    for lesion_only in (False, True):
        assert e.select_rows({'train_supervised': rows, 'validation': []}, 1, 1,
                             seed=0, lesion_only=lesion_only) == ([], [])


@pytest.mark.parametrize("planar", [False, True])
def test_tiny_cli_reads_real_frozen_observations_and_reports_both_modes(tmp_path, monkeypatch, capsys, planar):
    import nibabel as nib
    import runpy
    from anatobind.aur import data_contract as dc
    e = api()
    rows = []
    for i in range(4):
        seg = np.full((32, 32, 8), 2, np.int16); seg[16:] = 41
        lesion = np.zeros(seg.shape, np.int16); lesion[4:8, 4:8, 2:6] = 1
        image = np.random.default_rng(i).normal(100, 5, seg.shape).astype(np.float32)
        image[lesion > 0] += 50
        paths = {}
        for name, array in (('image', image), ('anatomy', seg), ('lesion', lesion)):
            path = tmp_path / f'synthetic_{i}_{name}.nii.gz'
            obj = nib.Nifti1Image(array, np.eye(4)); obj.header.set_xyzt_units('mm')
            nib.save(obj, path); paths[name] = str(path)
        rows.append(dict(paths, source='synthetic', patient=f'p{i}', case=f'c{i}',
            sequence='FLAIR', source_sequence='FLAIR', split='train' if i < 2 else 'val',
            a_supervised=True, u_supervised=True, r_supervised=True, u_values=[1], a_ignore_values=[1],
            source_hashes={k:dc.file_sha256(v) for k,v in paths.items()}))
    contract_dir = tmp_path / 'contract'; contract_dir.mkdir(mode=0o700)
    roles = {'train_supervised': rows[:2], 'validation': rows[2:], 'train_ssl': [], 'validation_ssl': []}
    for role, filename in dc.ROLE_FILES.items(): dc.write_private_json(contract_dir / filename, roles[role])
    outputs = {name: dc.file_sha256(contract_dir / name) for name in dc.ROLE_FILES.values()}
    dc.write_private_json(contract_dir/'anonymous_summary.json',{})
    outputs['anonymous_summary.json']=dc.file_sha256(contract_dir/'anonymous_summary.json')
    contract = {'version': dc.VERSION, 'inputs': {}, 'source_files': {value: dc.file_sha256(value)
        for row in rows for key, value in row.items() if key in ('image', 'anatomy', 'lesion')},
        'source_snapshot': {}, 'outputs': outputs, 'summary': {}}
    dc.write_private_json(contract_dir / 'contract.json', contract)
    dc.write_private_json(contract_dir / 'complete.json', {'version': dc.VERSION,
        'contract_sha256': dc.file_sha256(contract_dir / 'contract.json'), 'outputs': outputs})
    def synthetic_contract(path, verify=True):
        dc.verify_contract(path, check_paths=False)  # Synthetic fixtures have their own source root.
        return dc.read_contract(path, verify=False)
    monkeypatch.setattr(e, 'read_contract', synthetic_contract)
    monkeypatch.setattr(e, 'validate_private_output', lambda p: dc.private_output(p, private_root=tmp_path))
    cfg = dict(embed=32, depths=[1] * 4, heads=[1, 2, 4, 8], window=[2] * 3,
               patch=[2, 4, 4], dual_input=True, use_checkpoint=False)
    if planar:
        cfg["planar_merge"] = True
    bb = SwinBackbone(**cfg); state = bb.state_dict()
    checkpoint = tmp_path / 'stage_i.pt'
    torch.save({'format_version': 2, 'stage': 'I', 'cfg': {'model': cfg, 'ssl': {}}, 'step': 1,
        'backbone_state_dict': state, 'model_state_dict': {'backbone.' + k: v for k, v in state.items()}}, checkpoint)
    cli = runpy.run_path(str(Path(__file__).resolve().parents[1] / 'scripts/aur_unified_ssl_eval.py'))
    out = tmp_path / 'g1'
    built = []
    def observed_backbone(**kwargs):
        backbone = SwinBackbone(**kwargs)
        built.append(backbone)
        return backbone
    monkeypatch.setattr(e, 'SwinBackbone', observed_backbone)
    rng_before = torch.get_rng_state().clone()
    assert cli['main'](['--contract', str(contract_dir), '--checkpoint', str(checkpoint), '--out', str(out),
                        '--crop-2d', '1', '32', '32', '--crop-3d', '8', '32', '32',
                        '--calibration', '2', '--validation', '2', '--lesion', '2', '--seeds', '0',
                        '--readout-epochs', '2', '--bootstrap', '10']) == 0
    assert torch.equal(torch.get_rng_state(), rng_before)
    assert len(built) == 2
    assert set(built[0].state_dict()) == set(built[1].state_dict()) == set(state)
    assert sum('merge_2d' in k for k in built[1].state_dict()) == (9 if planar else 0)
    emitted = json.loads(capsys.readouterr().out)
    assert emitted['default_recipe_match'] is False
    assert set(emitted['modes']) == {'2d', '3d'}
    result = json.loads((out / 'g1_report.json').read_text())
    for mode in result['modes'].values():
        assert set(mode['arms']) == {'ssl', 'random'}
        assert mode['counts']['host_calibration'] == 2
        assert mode['status'] == 'INSUFFICIENT'  # Only two of the thirteen host classes have support.
        assert mode['support']['host_validation_cells'][0] > 0
    assert json.loads((out / 'complete.json').read_text())['frozen_backbones_unchanged'] is True

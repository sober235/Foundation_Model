"""Frozen development-only evaluation, independent planar/volume denominators."""
import importlib
import json
from pathlib import Path
import runpy

import nibabel as nib
import numpy as np
import pytest
import torch

from test_aur_unified_infer import tiny_model, certified_export, supervised_boundaries


def api():
    try:
        return importlib.import_module('anatobind.aur.unified_evaluation')
    except ModuleNotFoundError:
        pytest.fail('Mode-explicit development evaluator is missing')


@pytest.fixture
def qualified(tmp_path, monkeypatch):
    from anatobind.aur import data_contract as dc
    from test_aur_data_contract import committed_empty_roles
    home = tmp_path/'private'; home.mkdir()
    monkeypatch.setattr(dc, 'private_output', lambda p: dc.validate_private_output(p, private_root=home))
    monkeypatch.setattr(dc, 'source_path', lambda p: Path(p).resolve())
    out = home/'contract'; out.mkdir()
    shape = (16, 16, 4)
    affine = np.diag([.5, .5, 1., 1.])
    image = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    anatomy = np.full(shape, 2, np.int16)
    lesion = np.zeros(shape, np.float32)
    # One qualified3D source; one one-pixel ignored2D component; one plane-only
    # target (4 pixels =1mm2, source volume1mm3<10).
    lesion[3:9, 3:9, :] = .99999994
    lesion[12, 12, 2] = 1
    lesion[0:2, 12:14, 2] = 1
    paths = {}
    for name, values in [('image', image), ('anatomy', anatomy), ('lesion', lesion)]:
        path = home/(name+'.nii')
        obj = nib.Nifti1Image(values, affine); obj.header.set_xyzt_units('mm'); nib.save(obj, path)
        paths[name] = str(path)
    row = dict(source='isles', patient='synthetic_private_patient', case='synthetic_private_case',
               sequence='DWI', source_sequence='DWI', role='development_validation', split='val',
               a_supervised=True, u_supervised=True, r_supervised=True, u_values=[1], a_ignore_values=[1],
               source_hashes={name: dc.file_sha256(path) for name, path in paths.items()}, **paths)
    outputs = committed_empty_roles(dc, out)
    (out/'validation.json').write_text(json.dumps([row]))
    outputs['validation.json'] = dc.file_sha256(out/'validation.json')
    frozen = out/'execution_source/qualified.py'; frozen.parent.mkdir(); frozen.write_text('qualified source')
    outputs['execution_source/qualified.py'] = dc.file_sha256(frozen)
    contract = dict(version=dc.VERSION, inputs={}, source_files={p: dc.file_sha256(p) for p in paths.values()},
        source_snapshot={'qualified.py': dc.file_sha256(frozen)}, outputs=outputs,
        summary={}, environment=dc.decoder_environment())
    dc.write_private_json(out/'contract.json', contract)
    dc.write_private_json(out/'complete.json', dict(version=dc.VERSION,
        contract_sha256=dc.file_sha256(out/'contract.json'), outputs=outputs))
    return home, out, row


def test_freeze_is_header_only_central_complete_and_verifies_role_membership(qualified, monkeypatch):
    ue = api(); home, path, row = qualified
    # Header-only freeze must not touch proxies (including canonical reorientation).
    def forbid(*args, **kwargs):
        pytest.fail('selection read an array or target')
    monkeypatch.setattr(nib.arrayproxy.ArrayProxy, '__array__', forbid)
    plan = ue.freeze_observations(path)
    assert [(o['input_mode'], o['plane_index'], o['shape']) for o in plan['observations']] == [
        ('2d', 2, [1, 16, 16]), ('3d', None, [4, 16, 16])]
    assert plan['rows'] == [row]
    assert plan['selection'] == 'complete_canonical_central_plane'
    assert plan['evidence'] == 'NOT_EVIDENCE'
    # A role byte change must fail verification, before its image can be opened.
    (path/'validation.json').write_text(json.dumps([{**row, 'split': 'test'}]))
    with pytest.raises(ValueError, match='integrity'):
        ue.freeze_observations(path)


@pytest.mark.parametrize('field,value', [('role', 'test'), ('split', 'train')])
def test_committed_non_development_member_cannot_be_relabelled(qualified, field, value):
    from anatobind.aur import data_contract as dc
    ue = api(); _, path, row = qualified
    member = path/'validation.json'; member.write_text(json.dumps([{**row, field: value}]))
    c = json.loads((path/'contract.json').read_text()); c['outputs']['validation.json'] = dc.file_sha256(member)
    (path/'contract.json').write_text(json.dumps(c))
    (path/'complete.json').write_text(json.dumps(dict(version=dc.VERSION,
        contract_sha256=dc.file_sha256(path/'contract.json'), outputs=c['outputs'])))
    with pytest.raises(ValueError, match='development'):
        ue.freeze_observations(path)


def test_actual_both_mode_observations_have_independent_units_and_r_denominators(qualified):
    ue = api(); _, path, row = qualified
    plan = ue.freeze_observations(path)
    model = tiny_model().eval()
    records = [ue.evaluate_observation(model, row, o, crop=(1 if o['input_mode']=='2d' else 4, 16, 16))
               for o in plan['observations']]
    plane, volume = records
    assert plane['u']['n_gt'] == 2 and plane['u']['n_ignored'] == 1
    assert volume['u']['n_gt'] == 1 and volume['u']['n_ignored'] == 2
    assert plane['r']['eligible'] == 1 and plane['r']['unavailable_instances'] == 1
    assert volume['r']['eligible'] == 1
    assert plane['measurement_unit'] == 'mm2' and volume['measurement_unit'] == 'mm3'
    assert all('area_mm2' in r and 'volume_mm3' not in r for r in plane['u']['gt'])
    assert all('volume_mm3' in r and 'area_mm2' not in r for r in volume['u']['gt'])
    assert plane['r']['instances'][0]['truth'] == 0
    summary = ue.aggregate_observations(records)
    assert summary['n_patients'] == 1 and summary['n_sources'] == 1 and summary['n_observations'] == 2
    assert summary['modes']['2d']['u']['fp_per_image'] is not None
    assert 'fp_per_scan' not in summary['modes']['2d']['u']
    assert summary['modes']['3d']['u']['fp_per_scan'] is not None
    assert summary['modes']['2d']['u']['area_mm2']['truth'] == 10.
    assert summary['modes']['2d']['u']['area_mm2']['ignored'] == .25
    assert 'volume_mm3' not in summary['modes']['2d']['u']
    assert summary['modes']['3d']['u']['volume_mm3']['truth'] == 36.
    assert summary['quality_certified'] is False and summary['evidence'] == 'NOT_EVIDENCE'
    text = json.dumps(summary, allow_nan=False)
    assert 'synthetic_private' not in text


def test_plane_forward_normalizes_raw_canonical_plane_once(qualified):
    ue = api(); _,path,row = qualified
    observation = ue.freeze_observations(path,modes=('2d',))['observations'][0]
    model = tiny_model().eval(); seen = []
    hook = model.backbone.patch_embed_2d.register_forward_pre_hook(lambda module,args:seen.append(args[0].detach().clone()))
    try:
        ue.evaluate_observation(model,row,observation,crop=(1,16,16))
    finally:
        hook.remove()
    raw = np.asarray(nib.load(row['image']).dataobj).transpose(2,1,0)[2:3]
    lo,hi = np.percentile(raw[raw!=0],[.5,99.5])
    expected = ((np.clip(raw,lo,hi)-lo)/(hi-lo)*2-1).astype(np.float32)
    np.testing.assert_allclose(seen[0][0,0].numpy(),expected,atol=1e-7,rtol=0)


def test_anatomy_supports_include_hallucinated_classes_and_exclude_a_ignore():
    ue = api()
    # entity1=label2; entity2=label3. Ignored third voxel must not enter any support.
    pred = np.array([[[1, 2, 2]]]); true = np.array([[[1, 1, 2]]])
    a = ue.anatomy_metrics(pred, true, np.array([[[False, False, True]]]))
    assert a['entities']['truth'][:2] == [2, 0]
    assert a['entities']['predicted'][:2] == [1, 1]
    assert a['entities']['dice'][:2] == [pytest.approx(2/3), 0.]
    assert len(a['entities']['truth']) == 32 and len(a['hosts']['truth']) == 13
    assert a['n_valid'] == 2


def test_unavailable_supervision_stays_null_in_records_and_coverage(qualified):
    ue = api(); _, path, row = qualified
    observation = ue.freeze_observations(path, modes=('2d',))['observations'][0]
    row = dict(row, a_supervised=False, u_supervised=False, r_supervised=False)
    from anatobind.aur.data_contract import json_digest
    observation['row_sha256'] = json_digest(row)
    record = ue.evaluate_observation(tiny_model().eval(), row, observation, crop=(1, 16, 16))
    assert all(record[k] is None for k in ('a','u','r'))
    summary = ue.aggregate_observations([record])['modes']['2d']
    assert summary['coverage'] == dict(a=0,u=0,r=0)
    assert all(summary[k] is None for k in ('a','u','r'))


def test_frozen_member_cannot_be_substituted_before_targets_are_read(qualified, monkeypatch):
    ue = api(); _, path, row = qualified
    observation = ue.freeze_observations(path, modes=('2d',))['observations'][0]
    def forbid(*args, **kwargs):
        pytest.fail('substituted membership reached target loading')
    monkeypatch.setattr(ue, 'load_unified_volume', forbid)
    with pytest.raises(ValueError, match='observation'):
        ue.evaluate_observation(tiny_model(),dict(row,patient='different'),observation,crop=(1,16,16))


def test_matching_filters_before_assignment_and_excuses_ignored():
    ue = api()
    gt = [dict(instance=1, box=[0,0,0,1,2,2]), dict(instance=None, box=[0,4,4,1,5,5], ignore=True)]
    dets = [dict(instance=1,box=[0,0,0,1,2,2],score=.1),
            dict(instance=2,box=[0,0,0,1,2,3],score=.9),
            dict(instance=3,box=[0,4,4,1,5,5],score=.8)]
    counts = ue.detection_metrics(gt, dets, score_threshold=.5)
    assert counts['n_hit'] == 1 and counts['n_fp'] == 0
    assert counts['matches'] == {1: 2} and counts['n_ignored'] == 1


def test_relation_tracks_keep_full_eligible_denominator_and_same_anatomy_b0():
    ue = api()
    # Unknown -1 is excluded; host0 and explicit NO_HOST13 both stay eligible.
    r = ue._binding_metrics(np.array([0,13,-1]),
        [dict(instance=1,host=0),dict(instance=2,host=0)], np.array([1,13,13]),
        {1:2}, [dict(instance=2,host=0)], np.array([13,0]), ['','','unknown'])
    assert r['eligible']==2 and r['unavailable_instances']==1
    assert r['controlled_hits']==1 and r['controlled_b0_hits']==1
    assert r['end_to_end_hits']==1 and r['end_to_end_b0_hits']==1
    assert r['matched_eligible']==1 and r['instances'][1]['end_to_end'] is None


def test_certified_export_real_both_mode_run_freezes_configuration_and_private_outputs(
        qualified, monkeypatch, tmp_path, supervised_boundaries):
    from anatobind.aur import data_contract as dc
    ue = api(); home, contract_path, _ = qualified
    identity = dict(supervised_boundaries[0]['data_contract'], contract_sha256=dc.file_sha256(contract_path/'contract.json'))
    def align(selected, final):
        for state in (selected, final): state['data_contract'] = identity
    exported = certified_export(tmp_path, monkeypatch, supervised_boundaries, align)
    path = home/'certified.pt'; torch.save(exported, path)
    monkeypatch.setattr(ue, 'validate_private_output', lambda p: dc.validate_private_output(p, private_root=home))
    # Observe real forwards: every forward must see an already-persisted plan;
    # hooks do not replace loading, model dispatch, decoding, or binding.
    original = ue.model_from_unified_export
    modes = []
    def observed_loader(*args, **kwargs):
        assert kwargs['data_contract'] == identity
        model, metadata = original(*args, **kwargs)
        def observe(module, args, kwargs):
            assert (home/'normalized/plan_private.json').is_file()
            modes.append(kwargs.get('input_mode'))
        model.register_forward_pre_hook(observe, with_kwargs=True)
        return model, metadata
    monkeypatch.setattr(ue, 'model_from_unified_export', observed_loader)
    summary = ue.run_evaluation(contract_path, path, home/'raw/../normalized',
        crops={'2d': (1,16,16), '3d': (4,16,16)})
    assert {'2d', '3d'} <= set(modes)
    assert summary['n_observations'] == 2
    out = home/'normalized'
    for name in ('plan_private.json','records_private.json','anonymous_summary.json','complete.json'):
        assert (out/name).is_file() and (out/name).stat().st_mode & 0o777 == 0o600
    assert out.stat().st_mode & 0o777 == 0o700
    plan = json.loads((out/'plan_private.json').read_text())
    assert plan['configuration']['score_threshold'] == .5
    assert plan['artifact_identity'] == identity
    assert 'evaluator_source_sha256' in plan
    assert plan['artifact_identity']['runtime_sha256'] != plan['evaluator_source_sha256']
    assert summary['artifact']['phase_completed'] and not summary['quality_certified']
    # Cross-contract artifact must fail before any forward.
    changed = dict(exported, data_contract={**identity, 'contract_sha256': '0'*64})
    torch.save(changed, home/'wrong.pt')
    with pytest.raises(ValueError, match='evaluation'):
        ue.run_evaluation(contract_path, home/'wrong.pt', home/'wrong-out', crops={'2d':(1,16,16),'3d':(4,16,16)})


@pytest.mark.parametrize('arguments', [[], ['--contract','PRIVATE_PATIENT','--export','x','--out','x','--modes','PRIVATE_PATIENT'],
    ['--contract','x','--export','x','--out','x','--score-threshold','PRIVATE_PATIENT']])
def test_cli_argument_errors_are_private(arguments, capsys):
    api()
    main = runpy.run_path(str(Path(__file__).resolve().parents[1]/'scripts/aur_unified_eval.py'))['main']
    with pytest.raises(SystemExit): main(arguments)
    assert 'PRIVATE_PATIENT' not in capsys.readouterr().err


def test_path_failures_are_private_and_cli_does_not_echo_private_values(tmp_path, capsys):
    ue = api()
    with pytest.raises(ValueError) as error:
        ue.run_evaluation(tmp_path/'PRIVATE_PATIENT', 'PRIVATE_CHECKPOINT', tmp_path/'PRIVATE_OUTPUT')
    assert 'PRIVATE_' not in str(error.value)
    main = runpy.run_path(str(Path(__file__).resolve().parents[1]/'scripts/aur_unified_eval.py'))['main']
    assert main(['--contract','PRIVATE_PATIENT','--export','PRIVATE_CHECKPOINT','--out','/PRIVATE_OUTPUT']) == 1
    assert 'PRIVATE_' not in capsys.readouterr().err

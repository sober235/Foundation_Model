"""Acceptance gates are fail-closed and use the registered observation units."""
from copy import deepcopy
import importlib
import json
from pathlib import Path
import runpy

import pytest

from anatobind.aur.data_contract import json_digest
from anatobind.aur.labels import ENTITY_LABELS
from anatobind.anatomy.labels import HOST_IDS, NAMES

SOURCES = ('pdgm', 'bmsr', 'isles')


def api():
    try: return importlib.import_module('anatobind.aur.gate')
    except ModuleNotFoundError: pytest.fail('The registered gate implementation is missing')


def registration(implementation='head'):
    return dict(gate=deepcopy(api().GATE_V1), relation_implementation=implementation,
                validation_selection_sha256='a' * 64, score_thresholds={s: .5 for s in SOURCES},
                frozen_before_evaluation=True)


def rec(mode='3d', source='pdgm', row='r1', plane=0, **kw):
    obs = dict(row_sha256=row, input_mode=mode, plane_index=None if mode == '3d' else plane,
               plane_kinds=['volume'] if mode == '3d' else ['lesion_centroid'],
               target_instances=[] if mode == '3d' else [1], pixel_area_mm2=1.)
    obs.update(kw.pop('observation', {}))
    return dict(input_mode=mode, source=source, observation=obs, **kw)


def supports(labels, n=100, dice=.9):
    return dict(labels=list(labels), truth=[n] * len(labels), predicted=[n] * len(labels),
                intersection=[int(n * dice)] * len(labels), dice=[dice] * len(labels))


def anatomy(n=100, dice=.9):
    return dict(entities=supports(ENTITY_LABELS, n, dice),
                hosts=supports([NAMES[h] for h in HOST_IDS], n, dice))


def u(hits=(1,), n_gt=1, dice=.7, fp=0, source_ids=None):
    out = dict(n_gt=n_gt, n_hit=len(hits), n_fp=fp,
               matches={str(i): i + 10 for i in hits}, matched_dice={str(i): dice for i in hits})
    if source_ids is not None: out['source_ids'] = source_ids
    return out


def references():
    return {s: dict(source=s, grid='aur_1mm_canonical', sensitivity=1., fp_per_scan=1.,
                    n_scans=1, n_gt=1, n_hit=1, n_fp=1, included_row_sha256=[s]) for s in SOURCES}


def relation(instance=1, truth=0, share=.9, detected=11, head=0, lookup=1):
    return dict(instance=instance, truth=truth, top_share=share, detected_instance=detected,
                end_to_end=head, end_to_end_b0=lookup)


def test_a_averages_observation_macros_not_pooled_counts():
    out = api().judge_a([rec(a=anatomy(10000, .99)), rec(row='r2', a=anatomy(100, .5))], '3d')
    assert out['host_macro'] == pytest.approx(.745)
    assert out['status'] == 'fail'


def test_a_requires_all_32_presence_and_all_13_host_support():
    a = anatomy(); a['entities']['predicted'][-1] = 0; a['entities']['intersection'][-1] = 0
    a['entities']['dice'][-1] = 0.
    assert api().judge_a([rec(a=a)], '3d')['status'] == 'fail'
    a = anatomy(); a['hosts']['truth'][-1] = 0; a['hosts']['intersection'][-1] = 0
    assert api().judge_a([rec(a=a)], '3d')['status'] == 'unavailable'


def test_planar_a_uses_visible_area_for_hosts_and_entities():
    assert api().judge_a([rec('2d', a=anatomy(19))], '2d')['status'] == 'unavailable'
    assert api().judge_a([rec('2d', a=anatomy(20))], '2d')['status'] == 'pass'


@pytest.mark.parametrize('value', [None, float('nan'), float('inf')])
def test_a_nonfinite_host_scores_cannot_pass(value):
    a = anatomy(); a['hosts']['dice'][0] = value
    assert api().judge_a([rec(a=a)], '3d')['status'] == 'unavailable'


def test_a_rejects_forged_dice_that_disagrees_with_counts():
    a = anatomy(100, .1); a['hosts']['dice'] = [.99] * 13
    assert api().judge_a([rec(a=a)], '3d')['status'] == 'unavailable'


def test_u_requires_complete_reference_source_set_and_matching_cohort():
    refs = references(); records = [rec(source=s, row=s, u=u()) for s in SOURCES]
    assert api().judge_u_3d(records, refs)['status'] == 'pass'
    refs['isles']['included_row_sha256'] = ['other']
    assert api().judge_u_3d(records, refs)['status'] == 'unavailable'
    del refs['isles']
    assert api().judge_u_3d(records, refs)['status'] == 'unavailable'


def test_u_cohort_intersection_reports_exclusions_and_does_not_pool_different_patients():
    records = [rec(source=s, row=s, u=u()) for s in SOURCES]
    records.append(rec(row='outside', u=u(hits=(), fp=99)))
    out = api().judge_u_3d(records, references())
    assert out['status'] == 'pass'
    assert out['sources']['pdgm']['excluded_records'] == 1


def test_u_budget_and_matched_mask_quality_both_gate():
    records = [rec(source=s, row=s, u=u(dice=.2)) for s in SOURCES]
    assert api().judge_u_3d(records, references())['status'] == 'fail'
    for r in records: r['u'] = u(fp=2)
    assert api().judge_u_3d(records, references())['status'] == 'fail'


def test_u_2d_pairs_lesions_with_their_own_volume_and_ignores_stride():
    vol = rec(u=u((1, 2), 2))
    p1 = rec('2d', u=u(source_ids=[1]))
    p2 = rec('2d', plane=1, observation={'target_instances': [2]}, u=u((), source_ids=[2], fp=2))
    stride = rec('2d', plane=2, observation={'plane_kinds': ['stride'], 'target_instances': []}, u=u(fp=100))
    out = api().judge_u_2d([p1, p2, stride], [vol])
    assert out['sources']['pdgm']['sensitivity_2d'] == .5
    assert out['sources']['pdgm']['sensitivity_3d'] == 1.
    assert out['sources']['pdgm']['status'] == 'fail'
    assert out['fp_per_image'] == 1.


def test_u_2d_cannot_omit_an_undetected_volume_lesion_or_repeat_a_target():
    vol = rec(u=u((1, 2), 2)); plane = rec('2d', u=u(source_ids=[1]))
    assert api().judge_u_2d([plane], [vol])['sources']['pdgm']['status'] == 'unavailable'
    assert api().judge_u_2d([plane, plane], [vol])['status'] == 'unavailable'


def test_r_excludes_ambiguous_unmatched_and_uses_frozen_implementation():
    rows = [relation(), relation(2, 3, .5, 12, 4, 4), relation(3, 5, .9, None, None, None)]
    record = rec(r={'instances': rows}, u=u(n_gt=3))
    out = api().judge_r([record], '3d', registration())
    assert out['n'] == 1 and out['agreement'] == 1. and out['ambiguous'] == 1
    assert out['b0_agreement'] == 0.
    assert api().judge_r([record], '3d', registration('lookup'))['status'] == 'fail'


def test_planar_r_counts_only_target_source_instances_on_centroid_planes():
    record = rec('2d', u=u((1, 2), 2, source_ids=[1, 2]),
                 r_inplane={'instances': [relation(), relation(2, 0, 1., 12, 1, 1)]})
    out = api().judge_r([record], '2d', registration())
    assert out['n'] == 1 and out['status'] == 'pass'


@pytest.mark.parametrize('problem', ['missing_instance', 'hidden_match', 'nonfinite_share'])
def test_r_incomplete_or_inconsistent_relation_records_cannot_pass(problem):
    records = [relation(), relation(2, 0, .9, 12, 0, 0)]
    if problem == 'missing_instance': records.pop()
    elif problem == 'hidden_match': records[-1]['detected_instance'] = None
    else: records[-1]['top_share'] = float('nan')
    out = api().judge_r([rec(u=u((1, 2), 2), r={'instances': records})], '3d', registration())
    assert out['status'] == 'unavailable'


def test_all_six_gates_pass_only_with_every_source_and_valid_registered_evidence():
    records = []
    for source in SOURCES:
        records.append(rec(source=source, row=source, a=anatomy(), u=u(), r={'instances': [relation()]}))
        records.append(rec('2d', source=source, row=source, a=anatomy(), u=u(source_ids=[1]),
                           r_inplane={'instances': [relation()]}))
    out = api().verdict(records, references(), registration=registration())
    assert out['overall'] == 'pass'
    assert all(v['status'] == 'pass' for m in out['modes'].values() for v in m.values())


def test_u_rejects_extra_reference_source_and_nonfinite_reference_scores():
    records = [rec(source=s, row=s, u=u()) for s in SOURCES]
    ref = references(); ref['unexpected'] = ref['pdgm']
    assert api().judge_u_3d(records, ref)['status'] == 'unavailable'


def test_reference_source_metadata_must_match_its_named_bucket():
    records = [rec(source=s, row=s, u=u()) for s in SOURCES]
    ref = references(); ref['isles']['source'] = 'pdgm'
    assert api().judge_u_3d(records, ref)['sources']['isles']['status'] == 'unavailable'
    ref = references(); ref['isles']['sensitivity'] = float('nan')
    assert api().judge_u_3d(records, ref)['status'] == 'unavailable'


def test_random_plane_cannot_enter_the_registered_a_observation_denominator():
    record = rec('2d', a=anatomy(), observation={'plane_kinds': ['random'], 'target_instances': []})
    assert api().judge_a([record], '2d')['status'] == 'unavailable'


def test_verdict_empty_or_unsupervised_denominators_never_pass():
    out = api().verdict([rec(), rec('2d')], {}, registration=registration())
    assert out['overall'] == 'unavailable'
    assert all(v['status'] == 'unavailable' for m in out['modes'].values() for v in m.values())
    assert out['evidence'] == 'NOT_EVIDENCE'


@pytest.mark.parametrize('change', ['constant', 'source', 'implementation', 'frozen', 'selection'])
def test_registration_cannot_silently_modify_protocol_or_pick_post_test(change):
    reg = registration()
    if change == 'constant': reg['gate']['a']['host_macro_min'] = .1
    elif change == 'source': del reg['score_thresholds']['isles']
    elif change == 'implementation': reg['relation_implementation'] = 'best_test_arm'
    elif change == 'frozen': reg['frozen_before_evaluation'] = False
    else: reg['validation_selection_sha256'] = ''
    with pytest.raises(ValueError): api().verdict([], {}, registration=reg)


def test_cli_requires_registration_commitment_and_redacts_arguments(tmp_path, monkeypatch, capsys):
    try: cli = runpy.run_path(str(Path(__file__).resolve().parents[1] / 'scripts/aur_gate.py'))
    except FileNotFoundError: pytest.fail('The gate CLI is missing')
    with pytest.raises(SystemExit): cli['main'](['--synthetic_private', 'secret'])
    assert 'synthetic_private' not in capsys.readouterr().err


def frozen_cli_inputs(tmp_path):
    from anatobind.aur.data_contract import file_sha256, write_private_json
    reg = registration(); regsha = json_digest(reg)
    regpath = tmp_path/'registration.json'; write_private_json(regpath, reg)
    ev = tmp_path/'evaluation'; ev.mkdir(mode=0o700)
    plan = dict(registration=reg, registration_sha256=regsha, role_contract_sha256='b'*64,
                export_sha256='c'*64, observations=[], configuration=dict(
                    score_thresholds=reg['score_thresholds'], registration_sha256=regsha,
                    relation_implementation='head'))
    for name, value in [('plan_private.json', plan), ('records_private.json', []), ('anonymous_summary.json', {})]:
        write_private_json(ev/name, value)
    outputs = {n: file_sha256(ev/n) for n in ['plan_private.json','records_private.json','anonymous_summary.json']}
    write_private_json(ev/'complete.json', dict(version='aur_gate_evaluation_v1', outputs=outputs,
                       registration_sha256=regsha, role_contract_sha256='b'*64, export_sha256='c'*64))
    ref = tmp_path/'reference'; ref.mkdir(mode=0o700)
    for name, value in [('reference_private.json', {}), ('reference_anonymous.json',
                         dict(version='aur_reference_1mm_v1', contract_sha256='b'*64, sources={}))]:
        write_private_json(ref/name, value)
    outputs = {n: file_sha256(ref/n) for n in ['reference_private.json', 'reference_anonymous.json']}
    write_private_json(ref/'contract.json', dict(version='aur_reference_1mm_v1', outputs=outputs, data_contract_sha256='b'*64))
    write_private_json(ref/'complete.json', dict(version='aur_reference_1mm_v1', outputs=outputs,
                       contract_sha256=file_sha256(ref/'contract.json')))
    return ['--records', str(ev/'records_private.json'), '--reference', str(ref/'reference_anonymous.json'),
            '--registration', str(regpath), '--registration-sha256', regsha, '--out', str(tmp_path/'result')]


def test_cli_uses_complete_commitments_private_paths_and_anonymous_outputs(tmp_path, monkeypatch, capsys):
    from anatobind.aur.data_contract import validate_private_output
    args = frozen_cli_inputs(tmp_path)
    try: cli = runpy.run_path(str(Path(__file__).resolve().parents[1] / 'scripts/aur_gate.py'))
    except FileNotFoundError: pytest.fail('The gate CLI is missing')
    monkeypatch.setitem(cli['main'].__globals__, 'validate_private_output', lambda p: validate_private_output(p, private_root=tmp_path))
    assert cli['main'](args) == 0
    text = capsys.readouterr().out
    assert json.loads(text)['overall'] == 'unavailable'
    assert 'synthetic_private' not in text and str(tmp_path) not in text
    assert (tmp_path/'result/verdict_anonymous.json').is_file()
    assert (tmp_path/'result/verdict.md').stat().st_mode & 0o777 == 0o600
    assert (tmp_path/'result').stat().st_mode & 0o777 == 0o700


def test_cli_rejects_mutated_records_before_writing_verdict(tmp_path, monkeypatch, capsys):
    from anatobind.aur.data_contract import validate_private_output
    args = frozen_cli_inputs(tmp_path)
    (tmp_path/'evaluation/records_private.json').write_text('[{"patient":"synthetic_private"}]')
    try: cli = runpy.run_path(str(Path(__file__).resolve().parents[1] / 'scripts/aur_gate.py'))
    except FileNotFoundError: pytest.fail('The gate CLI is missing')
    monkeypatch.setitem(cli['main'].__globals__, 'validate_private_output', lambda p: validate_private_output(p, private_root=tmp_path))
    assert cli['main'](args) == 1
    assert not (tmp_path/'result').exists()
    emitted = capsys.readouterr()
    assert 'synthetic_private' not in emitted.out + emitted.err


@pytest.mark.parametrize('change', ['registration', 'role', 'threshold', 'observation', 'reference'])
def test_cli_rejects_inconsistent_frozen_artifact_links(tmp_path, monkeypatch, capsys, change):
    from anatobind.aur.data_contract import file_sha256, validate_private_output
    args = frozen_cli_inputs(tmp_path)
    planpath = tmp_path/'evaluation/plan_private.json'
    plan = json.loads(planpath.read_text())
    if change == 'registration': plan['registration_sha256'] = 'f'*64
    elif change == 'role': plan['role_contract_sha256'] = 'f'*64
    elif change == 'threshold': plan['configuration']['score_thresholds']['pdgm'] = .1
    elif change == 'observation': plan['observations'] = [{'row_sha256': 'synthetic_private'}]
    else: (tmp_path/'reference/reference_anonymous.json').write_text('{}')
    planpath.write_text(json.dumps(plan))
    completepath = tmp_path/'evaluation/complete.json'
    complete = json.loads(completepath.read_text())
    complete['outputs']['plan_private.json'] = file_sha256(planpath)
    completepath.write_text(json.dumps(complete))
    cli = runpy.run_path(str(Path(__file__).resolve().parents[1] / 'scripts/aur_gate.py'))
    monkeypatch.setitem(cli['main'].__globals__, 'validate_private_output', lambda p: validate_private_output(p, private_root=tmp_path))
    assert cli['main'](args) == 1
    assert not (tmp_path/'result').exists()
    text = capsys.readouterr()
    assert 'synthetic_private' not in text.out + text.err

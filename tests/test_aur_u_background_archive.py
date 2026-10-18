"""Synthetic checks for the complete-only background-allocation archive."""
import copy
import importlib
import json
import sys
from types import ModuleType

import numpy as np
import pytest


def archive_module():
    try:
        return importlib.import_module('scripts.aur_u_background_archive')
    except ModuleNotFoundError:
        pytest.fail('Private background archive API is missing')


@pytest.fixture
def synthetic(monkeypatch):
    archive = archive_module()
    state = '5fbb0d6f9513fdebf66831033c6a8d32710a05ce2ec900656c062a61546e482f'
    recipe = dict(n_rows=1, n_crops=2, crops_per_volume=2, pairs=[['bmsr', 'T1c']],
                  crop=[2, 2, 2], n_queries=3, points=4)
    contract = dict(recipe=recipe, model_state_sha256=state,
                    source_sha256={'scripts/synthetic.py': 'a' * 64})
    arrays = {
        'head_0_weight': np.ones((3, 2), np.float32),
        'head_0_bias': np.zeros(3, np.float32),
        'head_2_weight': np.ones((2, 3), np.float32),
        'head_2_bias': np.zeros(2, np.float32),
        'event_tokens': np.ones((3, 2), np.float32),
        'entity_tokens': np.ones((2, 2), np.float32),
        'event_presence': np.zeros(3, np.float32),
        'entity_presence': np.zeros(2, np.float32),
        'u_logits': np.zeros((2, 2, 3, 4), np.float32),
        'u_features': np.ones((2, 2, 2, 4), np.float32),
        'u_points': np.tile(np.arange(4, dtype=np.int64), (2, 2, 1)),
        'sampled_instances': np.zeros((2, 2, 4), np.int64),
        'sampled_weights': np.ones((2, 2, 4), np.float32),
        'a_logits': np.zeros((2, 2, 4), np.float32),
        'a_entity_points': np.zeros((2, 4), np.int64),
        'a_ignore': np.zeros((2, 4), bool),
        'a_present': np.ones(2, bool),
        'roi_features': np.ones((2, 2), np.float32),
        'roi_reference_responses': np.zeros((2, 2, 2, 2), np.float64),
        'r_logits_before': np.zeros((3, 14), np.float32),
        'r_logits_after': np.zeros((3, 14), np.float32),
    }
    meta = dict(source='bmsr', sequence='T1c', row_ordinal=0, crop_ordinal=0,
                n_instances=1, a_supervised=True, u_supervised=True,
                model_state_before=state, model_state_after=state,
                roi_rows=[dict(draw=0, query=0, panel='fp_all_query', n_voxels=1),
                          dict(draw=1, query=1, panel='target_all', n_voxels=1)])
    tangent = ModuleType('scripts.aur_u_background_tangent')
    measurements = ModuleType('scripts.aur_u_background_measurements')
    # Only the numerical boundary is replaced. Files, hashes, schema, comparison,
    # completion and replay are the real implementation under test.
    tangent.analyze_crop = lambda meta, arrays: {'sum': float(arrays['u_logits'].sum())}
    measurements.response_report = lambda meta, arrays, analysis: dict(
        source=meta['source'], sequence=meta['sequence'],
        row_ordinal=meta['row_ordinal'], crop_ordinal=meta['crop_ordinal'],
        total=analysis['sum'], supported=2)
    measurements.aggregate_reports = lambda reports: dict(
        crops=len(reports), total=sum(r['total'] for r in reports), supported=4)
    monkeypatch.setitem(sys.modules, tangent.__name__, tangent)
    monkeypatch.setitem(sys.modules, measurements.__name__, measurements)
    return archive, contract, meta, arrays


def prepared(synthetic, tmp_path, *, complete=True):
    archive, contract, meta, arrays = synthetic
    out = tmp_path / 'archive'
    archive.create_archive(out, contract)
    for index in range(2):
        m = copy.deepcopy(meta)
        m['crop_ordinal'] = index
        m['array_sha256'] = {key: archive.array_digest(value) for key, value in arrays.items()}
        m['report'] = archive.reconstruct_crop(m, arrays)
        archive.write_crop(out, index, m, arrays)
    if complete:
        archive.complete_archive(out, contract)
    return archive, out, contract


def rehash(archive, out):
    archive.atomic_json(out / 'COMPLETE.json', dict(schema=1, files=archive.archive_hashes(out)))


def replace_npz(path, arrays):
    with path.open('wb') as stream:
        np.savez_compressed(stream, **arrays)
    path.chmod(0o600)


def test_complete_replay_rebuilds_without_model_and_changes_no_bytes(synthetic, tmp_path, monkeypatch):
    archive, out, contract = prepared(synthetic, tmp_path)
    before = {p.name: p.read_bytes() for p in out.iterdir()}
    monkeypatch.setattr('torch.load', lambda *a, **k: pytest.fail('Checkpoint opened'))
    assert archive.replay_archive(out, contract) == {'crops': 2, 'total': 0.0, 'supported': 4}
    assert before == {p.name: p.read_bytes() for p in out.iterdir()}
    assert set(before) == {'contract.json', 'crop_00.json', 'crop_00.npz',
                           'crop_01.json', 'crop_01.npz', 'aggregate.json', 'COMPLETE.json'}
    assert out.stat().st_mode & 0o777 == 0o700
    assert all(p.stat().st_mode & 0o777 == 0o600 for p in out.iterdir())


@pytest.mark.parametrize('change', ['missing', 'extra', 'changed', 'marker', 'directory', 'symlink', 'permissions'])
def test_bad_inventory_or_marker_never_replays(synthetic, tmp_path, change):
    archive, out, contract = prepared(synthetic, tmp_path)
    if change == 'missing': (out / 'crop_01.npz').unlink()
    elif change == 'extra': (out / 'extra').write_bytes(b'')
    elif change == 'changed': (out / 'crop_00.json').write_bytes(b'{}')
    elif change == 'marker': (out / 'COMPLETE.json').unlink()
    elif change == 'directory': (out / 'extra').mkdir()
    elif change == 'symlink':
        (out / 'crop_01.npz').unlink()
        (out / 'crop_01.npz').symlink_to(out / 'crop_00.npz')
    else: (out / 'crop_00.npz').chmod(0o644)
    with pytest.raises(ValueError): archive.replay_archive(out, contract)


@pytest.mark.parametrize('field', ['u_logits', 'sampled_instances', 'sampled_weights', 'a_entity_points',
                                  'u_points', 'head_0_bias', 'roi_features', 'roi_reference_responses',
                                  'r_logits_after', 'a_ignore', 'unknown'])
def test_semantic_array_tampering_is_rejected_even_with_updated_hashes(synthetic, tmp_path, field):
    archive, out, contract = prepared(synthetic, tmp_path)
    _, _, _, source_arrays = synthetic
    arrays = {k: v.copy() for k, v in source_arrays.items()}
    if field == 'u_logits': arrays[field].flat[0] = np.nan
    elif field in ('sampled_instances', 'a_entity_points'): arrays[field].flat[0] = 99
    elif field == 'sampled_weights': arrays[field].flat[0] = -1
    elif field == 'u_points': arrays[field].flat[0] = 8
    elif field == 'head_0_bias': arrays[field] = arrays[field].astype(np.float64)
    elif field == 'roi_features': arrays[field] = arrays[field][:1]
    elif field == 'roi_reference_responses': arrays[field] = arrays[field][..., :1]
    elif field == 'r_logits_after': arrays[field].flat[0] = 1
    elif field == 'a_ignore': arrays[field] = arrays[field].astype(np.int64)
    else: arrays['PRIVATE_UNKNOWN'] = np.zeros(1, np.float32)
    replace_npz(out / 'crop_00.npz', arrays)
    meta = archive.read_json(out / 'crop_00.json')
    meta['array_sha256'] = {k: archive.array_digest(v) for k, v in arrays.items()}
    archive.atomic_json(out / 'crop_00.json', meta)
    rehash(archive, out)
    with pytest.raises(ValueError) as exc: archive.replay_archive(out, contract)
    assert 'PRIVATE_UNKNOWN' not in str(exc.value)


@pytest.mark.parametrize('change', ['source', 'sequence', 'ordinal', 'state', 'report', 'roi', 'empty_roi', 'n_instances'])
def test_metadata_semantics_and_cached_reports_are_recomputed(synthetic, tmp_path, change):
    archive, out, contract = prepared(synthetic, tmp_path)
    meta = archive.read_json(out / 'crop_00.json')
    if change == 'source': meta['source'] = 'PRIVATE_SOURCE'
    elif change == 'sequence': meta['sequence'] = 'FLAIR'
    elif change == 'ordinal': meta['crop_ordinal'] = 1
    elif change == 'state': meta['model_state_before'] = meta['model_state_after'] = 'f' * 64
    elif change == 'report': meta['report']['total'] = 1.0
    elif change == 'roi': meta['roi_rows'][0]['query'] = 3
    elif change == 'empty_roi': meta['roi_rows'][0]['n_voxels'] = 0
    else: meta['n_instances'] = -1
    archive.atomic_json(out / 'crop_00.json', meta)
    rehash(archive, out)
    with pytest.raises(ValueError) as exc: archive.replay_archive(out, contract)
    assert 'PRIVATE_SOURCE' not in str(exc.value)


def test_numeric_commitments_and_aggregate_are_checked(synthetic, tmp_path):
    archive, out, contract = prepared(synthetic, tmp_path)
    meta = archive.read_json(out / 'crop_00.json')
    meta['array_sha256']['u_logits'] = '0' * 64
    archive.atomic_json(out / 'crop_00.json', meta)
    rehash(archive, out)
    with pytest.raises(ValueError): archive.replay_archive(out, contract)
    meta['array_sha256']['u_logits'] = archive.array_digest(synthetic[3]['u_logits'])
    archive.atomic_json(out / 'crop_00.json', meta)
    archive.atomic_json(out / 'aggregate.json', dict(crops=2, total=4.0, supported=4))
    rehash(archive, out)
    with pytest.raises(ValueError): archive.replay_archive(out, contract)


def test_archive_and_crop_overwrite_are_refused(synthetic, tmp_path):
    archive, out, contract = prepared(synthetic, tmp_path)
    with pytest.raises((ValueError, FileExistsError)): archive.create_archive(out, contract)
    before = {p.name: p.read_bytes() for p in out.iterdir()}
    with pytest.raises((ValueError, FileExistsError)):
        archive.write_crop(out, 0, synthetic[2], synthetic[3])
    assert before == {p.name: p.read_bytes() for p in out.iterdir()}
    with pytest.raises(ValueError): archive.complete_archive(out, contract)


@pytest.mark.parametrize('index', [-1, 2, '../escape', True])
def test_invalid_crop_indices_cannot_escape_archive(synthetic, tmp_path, index):
    archive, contract, meta, arrays = synthetic
    out = tmp_path / 'archive'
    archive.create_archive(out, contract)
    with pytest.raises(ValueError): archive.write_crop(out, index, meta, arrays)
    assert {p.name for p in out.iterdir()} == {'contract.json'}


@pytest.mark.parametrize('change', ['missing_recipe', 'wrong_state', 'bad_count', 'bad_pairs', 'foreign_contract'])
def test_invalid_and_changed_external_contracts_fail_closed(synthetic, tmp_path, change):
    archive, out, contract = prepared(synthetic, tmp_path)
    wrong = copy.deepcopy(contract)
    if change == 'missing_recipe': wrong.pop('recipe')
    elif change == 'wrong_state': wrong['model_state_sha256'] = 'f' * 64
    elif change == 'bad_count': wrong['recipe']['n_crops'] = 3
    elif change == 'bad_pairs': wrong['recipe']['pairs'] = [['PRIVATE_SOURCE', 'PRIVATE_SEQUENCE']]
    else: wrong['source_sha256']['scripts/synthetic.py'] = 'b' * 64
    with pytest.raises(ValueError): archive.replay_archive(out, wrong)


@pytest.mark.parametrize('phase', ['initial', 'hashing', 'marker'])
def test_expiry_never_leaves_complete_marker(synthetic, tmp_path, monkeypatch, phase):
    archive, out, contract = prepared(synthetic, tmp_path, complete=False)
    clock = {'value': 0}
    monkeypatch.setattr(archive.time, 'monotonic', lambda: clock['value'])
    if phase == 'initial': clock['value'] = 2
    elif phase == 'hashing':
        original = archive.archive_hashes
        def hashes(*args, **kwargs):
            result = original(*args, **kwargs)
            clock['value'] = 2
            return result
        monkeypatch.setattr(archive, 'archive_hashes', hashes)
    else:
        original = archive.atomic_json
        def atomic(path, value):
            original(path, value)
            if path.name == 'COMPLETE.json': clock['value'] = 2
        monkeypatch.setattr(archive, 'atomic_json', atomic)
    with pytest.raises(TimeoutError): archive.complete_archive(out, contract, deadline=1)
    assert not (out / 'COMPLETE.json').exists()


def test_compare_values_checks_nested_schema_and_finiteness_without_echoing_keys():
    archive = archive_module()
    archive.compare_values({'x': [None, True, 1, 1.0]}, {'x': [None, True, 1, 1.000001]})
    for a, b in [({'PRIVATE_KEY': 1}, {}), ([float('nan')], [0.0]), ([True], [1]), ([1], [1.0])]:
        with pytest.raises(ValueError) as exc: archive.compare_values(a, b)
        assert 'PRIVATE_KEY' not in str(exc.value)


def test_atomic_json_is_private_and_rejects_nonfinite(tmp_path):
    archive = archive_module()
    p = tmp_path / 'record.json'
    archive.atomic_json(p, {'value': 1})
    assert p.stat().st_mode & 0o777 == 0o600
    with pytest.raises(ValueError): archive.atomic_json(p, {'value': float('inf')})
    assert archive.read_json(p) == {'value': 1}


def test_array_digest_preserves_original_archive_commitment_format():
    archive = archive_module()
    array = np.array([[1., 2.]], np.float32)
    assert archive.array_digest(array) == '38923130b143b5e789ac7b9156188e98284936534ba113f676a6ad82a14bda90'


def test_completion_schema_requires_integer_version(synthetic, tmp_path):
    archive, out, contract = prepared(synthetic, tmp_path)
    marker = archive.read_json(out / 'COMPLETE.json')
    marker['schema'] = True
    archive.atomic_json(out / 'COMPLETE.json', marker)
    with pytest.raises(ValueError): archive.replay_archive(out, contract)


def test_contract_count_does_not_accept_boolean_alias(synthetic, tmp_path):
    archive, out, contract = prepared(synthetic, tmp_path)
    changed = copy.deepcopy(contract)
    changed['recipe']['n_rows'] = True
    archive.atomic_json(out / 'contract.json', changed)
    rehash(archive, out)
    with pytest.raises(ValueError): archive.replay_archive(out, contract)


@pytest.mark.parametrize('payload', [b'{"x":1,"x":2}', b'{"x":NaN}', b'{"x":Infinity}'])
def test_json_reader_rejects_ambiguous_or_nonfinite_values(tmp_path, payload):
    archive = archive_module()
    path = tmp_path / 'bad.json'
    path.write_bytes(payload)
    path.chmod(0o600)
    with pytest.raises(ValueError): archive.read_json(path)


def test_symlinked_output_parent_is_refused(synthetic, tmp_path):
    archive, contract, _, _ = synthetic
    destination = tmp_path / 'destination'
    destination.mkdir()
    indirect = tmp_path / 'indirect'
    indirect.symlink_to(destination, target_is_directory=True)
    with pytest.raises(ValueError): archive.create_archive(indirect / 'archive', contract)
    assert not (destination / 'archive').exists()

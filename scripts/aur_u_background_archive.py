"""Complete-only private numeric evidence for the frozen head-tangent diagnostic.

Only numeric NPZ files and private JSON records are read during replay. Imports
of the tangent and measurement helpers are deliberately deferred until numeric
reconstruction, so this module never imports a model, image loader or checkpoint.
All errors use fixed text because even metadata keys may identify a patient.
"""
from functools import wraps
import hashlib
import json
import os
from pathlib import Path
import stat
import tempfile
import time
import zipfile

import numpy as np


CHECKPOINT_STATE_SHA = '5fbb0d6f9513fdebf66831033c6a8d32710a05ce2ec900656c062a61546e482f'
PAIRS = (('bmsr', 'T1c'), ('isles', 'DWI'), ('pdgm', 'FLAIR'))
PANELS = frozenset(('fp_unsampled_query', 'fp_all_query', 'fp_unsampled_component',
                    'fp_all_component', 'target_all', 'target_fp_query', 'target_tp_query'))
ARRAY_DTYPES = {name: np.dtype(dtype) for dtype, names in (
    ('float32', ('head_0_weight', 'head_0_bias', 'head_2_weight', 'head_2_bias',
                 'event_tokens', 'entity_tokens', 'event_presence', 'entity_presence',
                 'u_logits', 'u_features', 'sampled_weights', 'a_logits', 'roi_features',
                 'r_logits_before', 'r_logits_after')),
    ('int64', ('u_points', 'sampled_instances', 'a_entity_points')),
    ('bool', ('a_ignore', 'a_present')),
    ('float64', ('roi_reference_responses',)),
) for name in names}
RTOL = 1e-5
ATOL = 1e-5


def _safe(function):
    @wraps(function)
    def guarded(*args, **kwargs):
        try:
            return function(*args, **kwargs)
        except TimeoutError:
            raise TimeoutError('Diagnostic time budget exhausted') from None
        except Exception:
            raise ValueError('Invalid or incomplete diagnostic archive') from None
    return guarded


def _require(condition):
    if not condition:
        raise ValueError('Invalid or incomplete diagnostic archive')


def _deadline(deadline):
    if deadline is not None:
        _require(type(deadline) in (int, float) and np.isfinite(deadline))
        if time.monotonic() >= deadline:
            raise TimeoutError('Diagnostic time budget exhausted')


def _path(path):
    path = Path(path).absolute()
    _require('..' not in path.parts)
    _require(not any(p.is_symlink() for p in (path, *path.parents)))
    return path


def _regular(path, *, private=True):
    mode = path.lstat().st_mode
    _require(stat.S_ISREG(mode) and (not private or stat.S_IMODE(mode) == 0o600))


def _directory(out):
    out = _path(out)
    mode = out.lstat().st_mode
    _require(stat.S_ISDIR(mode) and stat.S_IMODE(mode) == 0o700)
    return out


def _json_pairs(pairs):
    value = {}
    for key, item in pairs:
        _require(key not in value)
        value[key] = item
    return value


def _reject_constant(_):
    raise ValueError('Invalid JSON numeric value')


@_safe
def read_json(path):
    path = _path(path)
    _regular(path)
    return json.loads(path.read_bytes(), object_pairs_hook=_json_pairs,
                      parse_constant=_reject_constant)


def _atomic_bytes(path, payload, *, overwrite):
    path = _path(path)
    _require(path.parent.is_dir())
    if path.exists():
        _regular(path)
        _require(overwrite)
    fd, temporary = tempfile.mkstemp(prefix='.' + path.name + '.', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as stream:
            os.fchmod(stream.fileno(), 0o600)
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        if overwrite:
            os.replace(temporary, path)
        else:
            # Atomic publication without silently overwriting a concurrent file.
            os.link(temporary, path)
            os.unlink(temporary)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _json_bytes(value):
    return json.dumps(value, sort_keys=True, indent=2, allow_nan=False).encode() + b'\n'


@_safe
def atomic_json(path, value):
    _atomic_bytes(path, _json_bytes(value), overwrite=True)


def array_digest(array):
    """Same dtype/shape/bytes commitment as the original numeric diagnostic."""
    array = np.asarray(array)
    digest = hashlib.sha256(str((array.shape, array.dtype.str)).encode())
    digest.update(array.tobytes(order='C'))
    return digest.hexdigest()


def _sha(value):
    return type(value) is str and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


def _contract(contract):
    _require(type(contract) is dict)
    # Validate JSON serializability and finite numbers before creating any output.
    _json_bytes(contract)
    recipe = contract['recipe']
    _require(type(recipe) is dict)
    for key in ('n_rows', 'n_crops', 'crops_per_volume'):
        _require(type(recipe[key]) is int and recipe[key] > 0)
    _require(recipe['n_rows'] * recipe['crops_per_volume'] == recipe['n_crops'])
    pairs = recipe['pairs']
    _require(type(pairs) is list and bool(pairs))
    _require(all(type(p) is list and tuple(p) in PAIRS for p in pairs))
    _require(len({tuple(p) for p in pairs}) == len(pairs))
    _require(recipe['n_rows'] % len(pairs) == 0)
    if 'n_crops' in contract:
        _require(type(contract['n_crops']) is int and contract['n_crops'] == recipe['n_crops'])
    _require(contract['model_state_sha256'] == CHECKPOINT_STATE_SHA)
    sources = contract['source_sha256']
    _require(type(sources) is dict and all(type(k) is str and _sha(v) for k, v in sources.items()))
    for key in ('n_queries', 'points', 'n_entities', 'n_phi'):
        if key in recipe:
            _require(type(recipe[key]) is int and recipe[key] > 0)
    if 'crop' in recipe:
        _require(type(recipe['crop']) is list and len(recipe['crop']) == 3)
        _require(all(type(v) is int and v > 0 for v in recipe['crop']))
    expected = [pairs[i // (recipe['n_rows'] // len(pairs))] for i in range(recipe['n_rows'])]
    if 'selection' in contract:
        selection = contract['selection']
        _require(type(selection) is list and len(selection) == recipe['n_rows'])
        _require([[m['row']['source'], m['row']['sequence']] for m in selection] == expected)
    return recipe, expected


@_safe
def compare_values(actual, expected):
    """Exact recursive schemas and IDs; declared tolerance for float scalars."""
    if type(actual) is dict and type(expected) is dict:
        _require(actual.keys() == expected.keys())
        for key in actual:
            compare_values(actual[key], expected[key])
    elif type(actual) is list and type(expected) is list:
        _require(len(actual) == len(expected))
        for a, b in zip(actual, expected):
            compare_values(a, b)
    elif type(actual) is float and type(expected) is float:
        _require(np.isfinite(actual) and np.isfinite(expected))
        _require(np.isclose(actual, expected, rtol=RTOL, atol=ATOL))
    else:
        _require(type(actual) is type(expected) and actual == expected)


def _metadata(meta, contract=None, index=None):
    _require(type(meta) is dict)
    _json_bytes(meta)
    _require((meta['source'], meta['sequence']) in PAIRS)
    for key in ('row_ordinal', 'crop_ordinal', 'n_instances'):
        _require(type(meta[key]) is int and meta[key] >= 0)
    _require(type(meta['a_supervised']) is bool and type(meta['u_supervised']) is bool)
    _require(meta['model_state_before'] == meta['model_state_after'] == CHECKPOINT_STATE_SHA)
    if contract is not None:
        recipe, pairs = _contract(contract)
        _require(type(index) is int and 0 <= index < recipe['n_crops'])
        row, crop = divmod(index, recipe['crops_per_volume'])
        _require((meta['row_ordinal'], meta['crop_ordinal']) == (row, crop))
        _require([meta['source'], meta['sequence']] == pairs[row])


@_safe
def validate_arrays(meta, arrays, contract=None, index=None):
    _metadata(meta, contract, index)
    _require(type(arrays) is dict and arrays.keys() == ARRAY_DTYPES.keys())
    for key, value in arrays.items():
        _require(isinstance(value, np.ndarray) and value.dtype == ARRAY_DTYPES[key])
        _require(np.isfinite(value).all())
    _require(arrays['head_0_weight'].ndim == arrays['head_2_weight'].ndim == 2)
    hidden, tokens = arrays['head_0_weight'].shape
    channels, hidden_out = arrays['head_2_weight'].shape
    _require(min(hidden, tokens, channels) > 0 and hidden_out == hidden)
    _require(arrays['event_tokens'].ndim == arrays['entity_tokens'].ndim == 2)
    queries, width = arrays['event_tokens'].shape
    entities, entity_width = arrays['entity_tokens'].shape
    _require(min(queries, entities) > 0 and width == entity_width == tokens)
    _require(arrays['u_points'].ndim == 3)
    points = arrays['u_points'].shape[-1]
    _require(points > 0)
    rows = meta['roi_rows']
    _require(type(rows) is list)
    shapes = dict(head_0_weight=(hidden, tokens), head_0_bias=(hidden,),
                  head_2_weight=(channels, hidden), head_2_bias=(channels,),
                  event_tokens=(queries, tokens), entity_tokens=(entities, tokens),
                  event_presence=(queries,), entity_presence=(entities,),
                  u_logits=(2, 2, queries, points), u_features=(2, 2, channels, points),
                  u_points=(2, 2, points), sampled_instances=(2, 2, points),
                  sampled_weights=(2, 2, points), a_logits=(2, entities, points),
                  a_entity_points=(2, points), a_ignore=(2, points), a_present=(entities,),
                  roi_features=(len(rows), channels), roi_reference_responses=(len(rows), 2, 2, 2),
                  r_logits_before=(queries, 14), r_logits_after=(queries, 14))
    _require(all(value.shape == shapes[key] for key, value in arrays.items()))
    _require((arrays['u_points'] >= 0).all())
    _require(((arrays['sampled_instances'] >= 0) & (arrays['sampled_instances'] <= meta['n_instances'])).all())
    _require((arrays['sampled_weights'] >= 0).all())
    _require(((arrays['a_entity_points'] >= 0) & (arrays['a_entity_points'] <= entities)).all())
    _require(np.array_equal(arrays['r_logits_before'], arrays['r_logits_after']))
    if contract is not None:
        recipe = contract['recipe']
        for key, actual in (('n_queries', queries), ('points', points), ('n_entities', entities),
                            ('n_phi', sum(arrays[k].size for k in
                                          ('head_0_weight', 'head_0_bias', 'head_2_weight', 'head_2_bias')))):
            if key in recipe:
                _require(recipe[key] == actual)
        if 'crop' in recipe:
            _require((arrays['u_points'] < np.prod(recipe['crop'])).all())
    for i, row in enumerate(rows):
        _require(type(row) is dict and row['panel'] in PANELS)
        _require(type(row['draw']) is int and 0 <= row['draw'] < 2)
        _require(type(row['query']) is int and 0 <= row['query'] < queries)
        _require(type(row['n_voxels']) is int and row['n_voxels'] >= 0)
        if row['n_voxels'] == 0:
            _require(not arrays['roi_features'][i].any() and not arrays['roi_reference_responses'][i].any())
    _require(meta.get('array_sha256') == {k: array_digest(v) for k, v in arrays.items()})


@_safe
def reconstruct_crop(meta, arrays):
    """Recompute the declared float32 surrogate and its response report on CPU."""
    validate_arrays(meta, arrays)
    from scripts.aur_u_background_tangent import analyze_crop
    from scripts.aur_u_background_measurements import response_report
    analysis = analyze_crop(meta, arrays)
    result = response_report(meta, arrays, analysis)
    _require(type(result) is dict)
    _json_bytes(result)
    return result


def _names(contract, *, complete=False):
    names = {'contract.json'} | {f'crop_{i:02d}.{suffix}'
                                for i in range(contract['recipe']['n_crops'])
                                for suffix in ('json', 'npz')}
    if complete:
        names |= {'aggregate.json', 'COMPLETE.json'}
    return names


def _inventory(out, required=None):
    out = _directory(out)
    paths = list(out.iterdir())
    for path in paths:
        _regular(path)
    if required is not None:
        _require({p.name for p in paths} == required)
    return out


@_safe
def archive_hashes(out, *, deadline=None):
    out = _inventory(out)
    hashes = {}
    for path in sorted(out.iterdir()):
        _deadline(deadline)
        if path.name == 'COMPLETE.json':
            continue
        digest = hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b''):
                _deadline(deadline)
                digest.update(block)
        hashes[path.name] = digest.hexdigest()
    _deadline(deadline)
    return hashes


@_safe
def create_archive(out, contract):
    _contract(contract)
    out = _path(out)
    out.mkdir(mode=0o700, parents=False, exist_ok=False)
    os.chmod(out, 0o700)
    _atomic_bytes(out / 'contract.json', _json_bytes(contract), overwrite=False)


@_safe
def write_crop(out, index, meta, arrays):
    out = _inventory(out)
    contract = read_json(out / 'contract.json')
    recipe, _ = _contract(contract)
    _require(type(index) is int and 0 <= index < recipe['n_crops'])
    _require(not (out / 'COMPLETE.json').exists() and not (out / 'aggregate.json').exists())
    _require(not (out / f'crop_{index:02d}.npz').exists() and
             not (out / f'crop_{index:02d}.json').exists())
    _require({p.name for p in out.iterdir()} <= _names(contract))
    validate_arrays(meta, arrays, contract, index)
    compare_values(reconstruct_crop(meta, arrays), meta['report'])
    # A failed pair remains explicitly incomplete; no partial-cache resume exists.
    fd, temporary = tempfile.mkstemp(prefix='.crop.', dir=out)
    try:
        with os.fdopen(fd, 'wb') as stream:
            os.fchmod(stream.fileno(), 0o600)
            np.savez_compressed(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, out / f'crop_{index:02d}.npz')
    finally:
        os.unlink(temporary)
    _atomic_bytes(out / f'crop_{index:02d}.json', _json_bytes(meta), overwrite=False)


def _load_arrays(path):
    with zipfile.ZipFile(path) as zipped:
        names = zipped.namelist()
        _require(len(names) == len(set(names)) and
                 set(names) == {k + '.npy' for k in ARRAY_DTYPES})
    with np.load(path, allow_pickle=False) as values:
        return {key: values[key] for key in values.files}


def _rebuild(out, contract, *, complete, deadline=None):
    _deadline(deadline)
    recipe, _ = _contract(contract)
    out = _inventory(out, _names(contract, complete=complete))
    stored_contract = read_json(out / 'contract.json')
    _contract(stored_contract)
    _require(_json_bytes(stored_contract) == _json_bytes(contract))
    initial_hashes = None
    if complete:
        initial_hashes = archive_hashes(out, deadline=deadline)
        _require(_json_bytes(read_json(out / 'COMPLETE.json')) ==
                 _json_bytes({'schema': 1, 'files': initial_hashes}))
    reports = []
    for index in range(recipe['n_crops']):
        _deadline(deadline)
        meta = read_json(out / f'crop_{index:02d}.json')
        arrays = _load_arrays(out / f'crop_{index:02d}.npz')
        validate_arrays(meta, arrays, contract, index)
        actual = reconstruct_crop(meta, arrays)
        compare_values(actual, meta['report'])
        reports.append(actual)
        _deadline(deadline)
    from scripts.aur_u_background_measurements import aggregate_reports
    aggregate = aggregate_reports(reports)
    _require(type(aggregate) is dict)
    _json_bytes(aggregate)
    if complete:
        compare_values(aggregate, read_json(out / 'aggregate.json'))
        _inventory(out, _names(contract, complete=True))
        _require(archive_hashes(out, deadline=deadline) == initial_hashes)
    _deadline(deadline)
    return aggregate


@_safe
def complete_archive(out, contract, deadline=None):
    out = _directory(out)
    result = _rebuild(out, contract, complete=False, deadline=deadline)
    _deadline(deadline)
    atomic_json(out / 'aggregate.json', result)
    files = archive_hashes(out, deadline=deadline)
    _deadline(deadline)
    marker = out / 'COMPLETE.json'
    try:
        atomic_json(marker, {'schema': 1, 'files': files})
        _deadline(deadline)
        _inventory(out, _names(contract, complete=True))
    except Exception:
        marker.unlink(missing_ok=True)
        raise
    return result


@_safe
def replay_archive(out, expected_contract):
    return _rebuild(out, expected_contract, complete=True)

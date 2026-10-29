"""Development-only operating point selection; never access historical test."""
from copy import deepcopy
import math
from pathlib import Path

from anatobind.aur.data_contract import file_sha256, json_digest, read_json
from anatobind.aur.eval import REFERENCE, scan, sensitivity_at_budget
from anatobind.aur.gate import SOURCES, _reference


def read_reference(directory, contract_sha):
    """Read completed cached S7 results without repeating reference inference."""
    root = Path(directory).resolve()
    complete, contract = read_json(root/'complete.json'), read_json(root/'contract.json')
    mandatory = {'reference_anonymous.json', 'reference_private.json'}
    if (complete['version'] != 'aur_reference_1mm_v1'
            or not mandatory <= set(complete['outputs'])
            or file_sha256(root/'contract.json') != complete['contract_sha256']
            or contract['version'] != complete['version']
            or contract['outputs'] != complete['outputs']
            or contract['data_contract_sha256'] != contract_sha
            or contract.get('parent_contract_sha256') is not None):
        raise ValueError('Reference role or completion differs')
    for name, digest in complete['outputs'].items():
        path = (root/name).resolve()
        if not path.is_relative_to(root) or file_sha256(path) != digest:
            raise ValueError('Reference output differs')
    value = read_json(root/'reference_anonymous.json')
    if value['version'] != complete['version'] or value['contract_sha256'] != contract_sha:
        raise ValueError('Reference summary role differs')
    return value['sources']


def selection_rows(rows, reference):
    """Require the exact previously committed S7-covered development intersection."""
    if any(r.get('role') != 'development_validation' or r.get('split') != 'val' for r in rows):
        raise ValueError('Selection requires development membership')
    selected = []
    for source in SOURCES:
        ids, _, _, _ = _reference(reference[source])
        candidates = [r for r in rows if r.get('source') == source
                      and r.get('sequence') == REFERENCE[source]['sequence']
                      and r.get('u_supervised') is True and json_digest(r) in ids]
        hashes = [json_digest(r) for r in candidates]
        if set(hashes) != ids or len(hashes) != len(ids):
            raise ValueError('Development reference intersection differs')
        selected.extend(candidates)
    return selected


def choose_thresholds(records, reference):
    """Re-match complete score0 candidates at every registered grid cutoff."""
    result = {}
    for source in SOURCES:
        ids, gt, _, budget = _reference(reference[source])
        selected = [r for r in records if r['source'] == source]
        hashes = [r['row_sha256'] for r in selected]
        if (set(hashes) != ids or len(hashes) != len(ids)
                or any(r['input_mode'] != '3d' for r in selected)):
            raise ValueError('Incomplete development threshold cohort')
        scans = [scan('', r['u']['gt'], r['u']['detections']) for r in selected]
        if sum(not g.get('ignore', False) for s in scans for g in s['gt']) != gt:
            raise ValueError('Development reference ground truth differs')
        result[source] = sensitivity_at_budget(scans, budget)
    return result


def choose_relation(comparison):
    """Protect the weaker mode; use lookup for exact ties. Empty is not success."""
    scores = {}
    for name in ('head', 'lookup'):
        values = []
        for mode in ('2d', '3d'):
            value = comparison.get(name, {}).get(mode, {})
            if not isinstance(value, dict):
                raise ValueError('Both relation modes require matched support')
            n, agreement = value.get('n'), value.get('agreement')
            if (type(n) is not int or n <= 0 or isinstance(agreement, bool)
                    or not isinstance(agreement, (int, float))
                    or not math.isfinite(agreement) or not 0 <= agreement <= 1):
                raise ValueError('Both relation modes require matched support')
            values.append(agreement)
        scores[name] = (min(values), sum(values) / 2, name == 'lookup')
    return max(scores, key=scores.get)

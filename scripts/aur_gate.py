#!/usr/bin/env python
"""Judge frozen AUR evaluation/reference artifacts under a registered protocol."""
import argparse
import json
import os
from pathlib import Path
import sys

from anatobind.aur.data_contract import (file_sha256, json_digest, read_json,
                                        validate_private_output, write_private_json)
from anatobind.aur.gate import validate_registration, verdict


class _Parser(argparse.ArgumentParser):
    def error(self, message):
        self.exit(2, 'Invalid gate arguments\n')


def _committed(directory, version, mandatory):
    complete = read_json(directory/'complete.json')
    if complete['version'] != version or not set(mandatory) <= set(complete['outputs']):
        raise ValueError('Incomplete gate evidence')
    hashes = {str(directory/'complete.json'): file_sha256(directory/'complete.json')}
    for name, digest in complete['outputs'].items():
        path = (directory/name).resolve()
        if not path.is_relative_to(directory) or file_sha256(path) != digest:
            raise ValueError('Changed gate evidence')
        hashes[str(path)] = digest
    return complete, hashes


def _inputs(records_path, reference_path, registration_path, expected_sha):
    registration = validate_registration(read_json(registration_path))
    if json_digest(registration) != expected_sha:
        raise ValueError('Gate registration commitment differs')
    if records_path.name != 'records_private.json' or reference_path.name != 'reference_anonymous.json':
        raise ValueError('Invalid gate artifact')
    complete, hashes = _committed(records_path.parent, 'aur_gate_evaluation_v1',
        ('plan_private.json', 'records_private.json', 'anonymous_summary.json'))
    plan = read_json(records_path.parent/'plan_private.json')
    records = read_json(records_path)
    config = plan['configuration']
    if (complete['registration_sha256'] != expected_sha or plan['registration_sha256'] != expected_sha
            or json_digest(plan['registration']) != expected_sha
            or config['registration_sha256'] != expected_sha
            or config['relation_implementation'] != registration['relation_implementation']
            or complete['role_contract_sha256'] != plan['role_contract_sha256']
            or complete['export_sha256'] != plan['export_sha256']
            or any(registration['score_thresholds'].get(s) != value for s, value in config['score_thresholds'].items())
            or any(r['registration_sha256'] != expected_sha for r in records)
            or len(records) != len(plan['observations'])
            or [json_digest(r['observation']) for r in records] != [json_digest(o) for o in plan['observations']]):
        raise ValueError('Evaluation registration differs')
    reference_complete, reference_hashes = _committed(reference_path.parent, 'aur_reference_1mm_v1',
        ('reference_private.json', 'reference_anonymous.json'))
    reference_contract_path = reference_path.parent/'contract.json'
    if file_sha256(reference_contract_path) != reference_complete['contract_sha256']:
        raise ValueError('Reference completion differs')
    reference_contract = read_json(reference_contract_path)
    reference = read_json(reference_path)
    if (reference_contract['version'] != 'aur_reference_1mm_v1'
            or reference_contract['outputs'] != reference_complete['outputs']
            or reference_contract['data_contract_sha256'] != plan['role_contract_sha256']
            or reference['contract_sha256'] != plan['role_contract_sha256']
            or reference['version'] != 'aur_reference_1mm_v1'):
        raise ValueError('Reference data role differs')
    hashes.update(reference_hashes)
    hashes[str(reference_contract_path)] = file_sha256(reference_contract_path)
    hashes[str(registration_path)] = file_sha256(registration_path)
    return records, reference['sources'], registration, hashes


def _markdown(result):
    lines = ['# AUR acceptance verdict', '', 'Evidence: NOT_EVIDENCE', '',
             'Registration SHA256: ' + result['registration_sha256'], '',
             '| Mode | Task | Status | Supported observations/instances |',
             '|---|---|---|---|']
    for mode, tasks in result['modes'].items():
        for task, item in tasks.items():
            count = item.get('n_observations', item.get('n'))
            if count is None and 'sources' in item:
                count = sum(v.get('n_gt', 0) for v in item['sources'].values())
            lines.append(f"| {mode} | {task.upper()} | {item['status']} | {count if count is not None else 'unavailable'} |")
    lines.extend(['', 'Overall: ' + result['overall'], '',
                  'Thresholds are fixed by the registered protocol. Missing support is unavailable.', '',
                  '```json', json.dumps(result['thresholds'], indent=2, sort_keys=True), '```', ''])
    return '\n'.join(lines)


def main(argv=None):
    parser = _Parser(description=__doc__)
    for name in ('records', 'reference', 'registration', 'registration-sha256', 'out'):
        parser.add_argument('--'+name, required=True)
    args = parser.parse_args(argv)
    try:
        records_path, reference_path, registration_path, out = [validate_private_output(p) for p in
            (args.records, args.reference, args.registration, args.out)]
        if out.exists(): raise ValueError('Output already exists')
        records, reference, registration, hashes = _inputs(records_path, reference_path, registration_path,
                                                          args.registration_sha256)
        result = verdict(records, reference, registration=registration)
        if any(file_sha256(path) != digest for path, digest in hashes.items()):
            raise ValueError('Gate inputs changed')
        write_private_json(out/'verdict_anonymous.json', result)
        with os.fdopen(os.open(out/'verdict.md', os.O_WRONLY|os.O_CREAT|os.O_EXCL, 0o600), 'w') as stream:
            stream.write(_markdown(result))
        write_private_json(out/'complete.json', dict(version='aur_gate_v1', evidence='NOT_EVIDENCE',
            registration_sha256=json_digest(registration), inputs=hashes,
            outputs={n: file_sha256(out/n) for n in ('verdict_anonymous.json', 'verdict.md')}))
        print(json.dumps(dict(overall=result['overall'], modes={mode: {task: item['status'] for task, item in tasks.items()}
                                                              for mode, tasks in result['modes'].items()}), sort_keys=True))
        return 0
    except Exception:
        print('Gate verification failed; private evidence preserved', file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())

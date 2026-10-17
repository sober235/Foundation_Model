"""Prepare exact frozen sub-shards and import validated atomic records; no model execution."""
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
import hashlib
import json
import os
import runpy
import shutil
import time

REPO = Path('/home/congcongliu/src/Foundation_Model-aur-diagnostic')
BASE = Path('/home/congcongliu/outputs/aur_query_cc_20261017')
SOURCE = BASE / 'val_1874af2_s0of2'
DONE = BASE / 'val_1874af2_s1of2'


class PreparedOnly(Exception):
    pass


def validate_manifest(source, source_contract, target):
    digest = hashlib.sha256(json.dumps(source, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    if digest != source_contract['raw_inputs_sha256']:
        raise ValueError('Original frozen input manifest digest differs')
    if any(source.get(token) != hashes for token, hashes in target.items()):
        raise ValueError('Repartition would change an originally frozen raw input')


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    os.chdir(REPO)
    m = runpy.run_path(str(REPO / 'scripts/aur_compare_decoders.py'), run_name='repartition_helpers')
    m['torch'].set_num_threads(1)
    baseline = json.loads((m['SAVED'] / 'cases.json').read_text())
    old = {m['identity'](r): r for r in baseline}
    source_contract = json.loads((SOURCE / 'contract.json').read_text())
    source_manifest = json.loads((SOURCE / 'input_manifest_private.json').read_text())
    assert source_contract['code_sha'] == m['code_sha']()
    assert json.loads((DONE / 'COMPLETE.json').read_text())['all_reproduction_checks_passed']
    copied = []
    selections = [json.loads((DONE / 'selection_private.json').read_text())]
    original_loader = m['I'].model_from_export
    def stop_before_model(*args, **kwargs):
        raise PreparedOnly
    m['I'].model_from_export = stop_before_model
    try:
        for k in (0, 2):
            target = BASE / ('val_1874af2_s' + str(k) + 'of4')
            args = SimpleNamespace(saved_eval=m['SAVED'], samples=m['DATA'] / 'samples_1mm_v1.json',
                val_patients=m['DATA'] / 'ssl_manifest_v1/val_patients.json',
                checkpoint=m['DATA'] / 'ssl_runs/c2_stage3_80k_20261014_1344/aur_stage3_best.pt',
                mode='val', shard=str(k) + '/4', out=target, resume=False, gpu=0,
                started=time.monotonic(), max_seconds=28800)
            try:
                m['run'](args)
            except PreparedOnly:
                pass
            else:
                raise AssertionError('Model-load interception was not reached')
            contract = json.loads((target / 'contract.json').read_text())
            for key in ('code_sha', 'checkpoint_sha256', 'samples_sha256', 'val_patients_sha256',
                        'saved_cases_sha256', 'reference_sha256', 'saved_reference_sha256',
                        'saved_aggregate_sha256', 'decoder_rules', 'score_thresholds', 'crop', 'batch_size', 'mode'):
                assert contract[key] == source_contract[key], key
            selection = json.loads((target / 'selection_private.json').read_text())
            selections.append(selection)
            manifest = json.loads((target / 'input_manifest_private.json').read_text())
            # Check every row, including those without a completed source record.
            validate_manifest(source_manifest, source_contract, manifest)
            n = 0
            for row in selection:
                ident = m['identity'](row)
                token = hashlib.sha256(json.dumps(ident, separators=(',', ':')).encode()).hexdigest()
                path = SOURCE / 'row_records' / (token + '.json')
                if not path.exists():
                    continue
                raw = path.read_bytes()
                record = json.loads(raw)
                assert record['code_sha'] == contract['code_sha']
                assert record['input_sha256'] == manifest[token]
                assert m['identity'](record['result']) == ident
                m['verify_reference'](record['result'], old[ident])
                destination = target / 'row_records' / path.name
                with destination.open('xb') as stream:
                    stream.write(raw)
                assert destination.read_bytes() == raw
                n += 1
            copied.append(dict(shard=[k, 4], n_rows=len(selection), imported_validated_records=n,
                               contract_sha256=m['file_sha256'](target / 'contract.json')))
    finally:
        m['I'].model_from_export = original_loader
    keys = [m['identity'](r) for selection in selections for r in selection]
    assert len(keys) == len(set(keys)) == len(old) == 405
    assert set(keys) == set(old)
    report = dict(reason='Scheduling-only rebalance: alternating sequence order put all ISLES DWI on even shard.',
        method='Official frozen runner intercepted before model loading; imported records verified against raw manifest and saved reference.',
        code_sha=m['code_sha'](), completed_odd_rows=len(selections[0]), final_rows=len(keys),
        final_membership_exact=True, new_shards=copied,
        U_rows_per_new_shard=[dict(Counter(r['source'] + '/' + r['sequence'] for r in s if r.get('u_supervised'))) for s in selections[1:]],
        weights_rules_scores_and_inputs_unchanged=True, original_raw_input_hashes_preserved=True,
        model_forward_calls=0)
    (BASE / 'repartition_preparation.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report))


if __name__ == '__main__':
    main()

"""Data contract isolation, provenance, geometry and replay checks."""
import importlib
import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest


def api():
    try:
        return importlib.import_module('anatobind.aur.data_contract')
    except ModuleNotFoundError:
        pytest.fail('The formal data contract API is missing')


def row(patient='p', split='train', **extra):
    return dict(source='isles', patient=patient, case=patient, sequence='DWI', source_sequence='DWI',
                split=split, image='image.nii.gz', anatomy='anatomy.nii.gz', lesion='lesion.nii.gz',
                a_supervised=True, u_supervised=True, r_supervised=True, u_values=[1], a_ignore_values=[1], **extra)


def test_patient_group_rejects_cross_split_and_same_patient_sequences():
    dc = api()
    with pytest.raises(ValueError, match='split'):
        dc.validate_patient_roles([row(), row(split='test')], [], {})


def test_validation_patient_cannot_be_supervised_train():
    dc = api()
    with pytest.raises(ValueError, match='validation'):
        dc.validate_patient_roles([row()], [], {'isles': ['p']}, repaired_train=[row()])


def test_ssl_split_is_verified_against_patient_membership():
    dc = api()
    with pytest.raises(ValueError, match='split'):
        dc.validate_patient_roles([row()], [dict(row(), ssl_split='train')], {'isles': ['p']})


def test_unknown_u_and_r_do_not_become_negative_targets():
    dc = api()
    qualified = dc.supervised_row(row(), split='val', native_status='source_empty')
    assert qualified['u_supervised'] is False
    assert qualified['r_supervised'] is False
    assert qualified['qualification_reason']['u'] == 'source_empty_unknown'
    assert qualified['label_provenance']['negative_truth'] is False


def test_pdgm_sequence_empty_does_not_make_disease_free_claim():
    dc = api()
    sample = row(); sample.update(source='pdgm', sequence='T1c', source_sequence='T1c', u_values=[1,4])
    qualified = dc.supervised_row(sample, split='val', native_status='sequence_empty')
    assert qualified['u_supervised'] is True
    assert qualified['label_provenance']['negative_truth'] is False
    assert qualified['label_provenance']['lesion_state'] == 'sequence_empty'


def test_image_only_ssl_export_has_no_label_or_target_fields():
    dc = api()
    qualified = dc.image_only_row(row(), split='train')
    assert not any(k in qualified for k in ('anatomy','lesion','target','u_values','u_supervised','a_ignore_values'))
    assert qualified['patient_group'] == 'isles:p'
    assert qualified['input_modes'] == ['2d', '3d']
    assert qualified['role'] == 'train_ssl'


def test_normalized_output_is_the_same_private_path_used_by_writer(tmp_path, monkeypatch):
    dc = api()
    private = tmp_path / 'home'; private.mkdir()
    monkeypatch.setenv('HOME', str(private))
    target = dc.private_output('~/runs/../contract', private_root=private)
    assert target == (private / 'contract').resolve()
    dc.write_private_json(target / 'test.json', {'ok': True})
    assert (private / 'contract/test.json').exists()
    assert (target.stat().st_mode & 0o777) == 0o700
    assert ((target/'test.json').stat().st_mode & 0o777) == 0o600


def test_output_rejects_any_nested_git_and_external_source(tmp_path):
    dc = api()
    repo = tmp_path / 'other_repo'; repo.mkdir(); (repo/'.git').write_text('gitdir: elsewhere')
    with pytest.raises(ValueError, match='private'):
        dc.private_output(repo/'out', private_root=tmp_path)
    with pytest.raises(ValueError, match='source'):
        dc.source_path(tmp_path/'outside.nii', data_root=tmp_path/'data', repair_root=tmp_path/'repair')


def image(path, values=None, unit='mm', affine=None):
    obj=nib.Nifti1Image(np.zeros((3,4,5), np.float32) if values is None else values,
                       np.eye(4) if affine is None else affine)
    obj.header.set_xyzt_units(unit)
    nib.save(obj, path)
    return obj


def committed_empty_roles(dc,out,extra=None):
    outputs={} if extra is None else dict(extra)
    for name in (*dc.ROLE_FILES.values(),'anonymous_summary.json'):
        dc.write_private_json(out/name,{} if name=='anonymous_summary.json' else [])
        outputs[name]=dc.file_sha256(out/name)
    return outputs


def test_corrupt_labels_and_non_mm_units_are_refused(tmp_path):
    dc = api()
    good=image(tmp_path/'good.nii')
    image(tmp_path/'bad.nii', np.full((3,4,5), .4, np.float32))
    with pytest.raises(ValueError, match='label'):
        dc.inspect_label(tmp_path/'bad.nii', good)
    metres=image(tmp_path/'metres.nii', unit='meter')
    with pytest.raises(ValueError, match='geometry'):
        dc.inspect_geometry(metres)


def test_replay_rejects_source_and_output_changes(tmp_path):
    dc = api()
    source=tmp_path/'source.json'; source.write_text('{}')
    out=tmp_path/'out'; out.mkdir(mode=0o700)
    outputs=committed_empty_roles(dc,out)
    contract={'version': dc.VERSION, 'inputs': {str(source):dc.file_sha256(source)},
              'source_files': {}, 'source_snapshot': {}, 'outputs': outputs,
              'summary': {}}
    dc.write_private_json(out/'contract.json', contract)
    dc.write_private_json(out/'complete.json', {'version':dc.VERSION,'contract_sha256':dc.file_sha256(out/'contract.json'),
                                               'outputs':contract['outputs']})
    assert dc.verify_contract(out, check_paths=False)['version'] == dc.VERSION
    source.write_text('{"changed": true}')
    with pytest.raises(ValueError, match='integrity'):
        dc.verify_contract(out, check_paths=False)
    source.write_text('{}'); (out/'train_ssl.json').write_text('[1]')
    with pytest.raises(ValueError, match='integrity'):
        dc.verify_contract(out, check_paths=False)


def test_frozen_val_policy_precedes_native_label_read(tmp_path):
    dc = api()
    dc.freeze_policy(tmp_path, [row(patient='a'), row(patient='b')])
    frozen=json.loads((tmp_path/'policy.json').read_text())
    assert frozen['discrete_atol'] == 1e-5
    assert frozen['isles_selection'] == 'all_development_validation_patients'
    assert frozen['isles_patient_count'] == 2
    assert 'a' not in frozen and 'b' not in frozen
    assert (tmp_path/'validation_membership_private.json').exists()


def test_private_output_rejects_bare_repo_and_symlink_escape(tmp_path):
    dc=api()
    bare=tmp_path/'bare'; bare.mkdir(); (bare/'HEAD').write_text('ref: refs/heads/main')
    (bare/'objects').mkdir(); (bare/'config').write_text('[core]\n bare = true\n')
    with pytest.raises(ValueError, match='private'):
        dc.validate_private_output(bare/'output', private_root=tmp_path)
    home=tmp_path/'home'; home.mkdir(); (home/'escape').symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(ValueError, match='private'):
        dc.validate_private_output(home/'escape/output', private_root=home)


def test_cli_uses_normalized_validated_output_and_redacts_invalid_arguments(tmp_path, monkeypatch, capsys):
    import runpy
    module=runpy.run_path(str(Path(__file__).resolve().parents[1]/'scripts/aur_qualify_training_data.py'))
    main=module['main']; globals_=main.__globals__
    seen=[]
    monkeypatch.setitem(globals_,'private_output',lambda path:(tmp_path/'normalized').resolve())
    monkeypatch.setitem(globals_,'qualify',lambda path,workers:seen.append(path) or {'ok':True})
    assert main(['--out','~/raw/../normalized']) == 0
    assert seen == [(tmp_path/'normalized').resolve()]
    with pytest.raises(SystemExit): main(['--out','private-patient-value','--workers','private-patient-value'])
    assert 'private-patient-value' not in capsys.readouterr().err


def test_source_symlink_cannot_escape_data_root(tmp_path):
    dc=api()
    data=tmp_path/'data'; data.mkdir(); outside=tmp_path/'external.nii'; outside.write_bytes(b'x')
    (data/'image.nii').symlink_to(outside)
    with pytest.raises(ValueError, match='source'):
        dc.source_path(data/'image.nii',data_root=data,repair_root=tmp_path/'repair')


def test_prior_qc_reuse_requires_matching_anatomy_and_lesion_bytes():
    dc=api()
    assert hasattr(dc,'verify_training_qc'), 'Frozen TRAIN array QC must verify its per-file hashes'
    sample=row(); sample.update(anatomy='/data/anatomy.nii',lesion='/data/lesion.nii')
    records=[{'source':'isles','anatomy_path':'/data/anatomy.nii','sha256':'a',
              'lesion_path':'/data/lesion.nii','lesion_sha256':'b'}]
    supplement=[{'path':'/data/lesion.nii','sha256':'b','qualified':True}]
    hashes={'/data/anatomy.nii':'a','/data/lesion.nii':'b'}
    assert dc.verify_training_qc([sample],hashes,records,[],supplement=supplement)['anatomy_rows'] == 1
    with pytest.raises(ValueError, match='QC'):
        dc.verify_training_qc([sample],{**hashes,'/data/lesion.nii':'changed'},records,[],supplement=supplement)
    with pytest.raises(ValueError, match='QC'):
        dc.verify_training_qc([sample],hashes,records,[])
    repair=[{'new_path':'/data/lesion.nii','new_sha256':'b'}]
    assert dc.verify_training_qc([sample],hashes,records,repair)['repaired_lesion_rows'] == 1


def test_replay_rejects_decoder_environment_change(tmp_path):
    dc=api()
    out=tmp_path/'out'; out.mkdir(mode=0o700)
    contract={'version':dc.VERSION,'inputs':{},'source_files':{},'source_snapshot':{},'outputs':{},'summary':{},
              'environment':{'numpy':'incompatible-decoder-environment'}}
    dc.write_private_json(out/'contract.json',contract)
    dc.write_private_json(out/'complete.json',{'version':dc.VERSION,'contract_sha256':dc.file_sha256(out/'contract.json'),'outputs':{}})
    with pytest.raises(ValueError, match='integrity'):
        dc.verify_contract(out,check_paths=False)


def test_metadata_finalization_preserves_and_verifies_inherited_output_chain(tmp_path, monkeypatch):
    dc=api()
    repo=tmp_path/'repo'; repo.mkdir(); code=repo/'module.py'; code.write_text('new execution')
    monkeypatch.setattr(dc,'REPO',repo)
    initial=tmp_path/'initial'; initial.mkdir(mode=0o700)
    frozen=initial/'execution_source/module.py'; frozen.parent.mkdir(mode=0o700);frozen.write_text('original execution')
    mask=initial/'mask.nii';mask.write_bytes(b'frozen label')
    source=tmp_path/'source';source.write_bytes(b'unchanged input')
    inherited_outputs={'mask.nii':dc.file_sha256(mask),'execution_source/module.py':dc.file_sha256(frozen)}
    inherited_outputs=committed_empty_roles(dc,initial,inherited_outputs)
    original={'version':dc.VERSION,'inputs':{str(source):dc.file_sha256(source)},'source_files':{},
              'source_snapshot':{'module.py':dc.file_sha256(frozen)},'outputs':inherited_outputs,'summary':{}}
    dc.write_private_json(initial/'contract.json',original)
    dc.write_private_json(initial/'complete.json',{'version':dc.VERSION,'contract_sha256':dc.file_sha256(initial/'contract.json'),'outputs':inherited_outputs})
    final=initial/'final_v1';final.mkdir(mode=0o700)
    final_outputs=committed_empty_roles(dc,final)
    contract={'version':dc.VERSION,'inputs':{str(source):dc.file_sha256(source)},'source_files':{},
              'source_snapshot':{'module.py':dc.file_sha256(code)},'outputs':final_outputs,'summary':{},
              'inherited_contract':{'path':str(initial/'contract.json'),'sha256':dc.file_sha256(initial/'contract.json'),
                                    'complete_sha256':dc.file_sha256(initial/'complete.json')}}
    dc.write_private_json(final/'contract.json',contract)
    dc.write_private_json(final/'complete.json',{'version':dc.VERSION,'contract_sha256':dc.file_sha256(final/'contract.json'),'outputs':final_outputs})
    assert dc.verify_contract(final,check_paths=False)['inherited_contract']['sha256'] == dc.file_sha256(initial/'contract.json')
    mask.write_bytes(b'changed label')
    with pytest.raises(ValueError,match='integrity'):
        dc.verify_contract(final,check_paths=False)


def test_legacy_absent_r_flag_retains_qualified_3d_pseudo_supervision():
    dc=api()
    sample=row();sample.pop('r_supervised')
    qualified=dc.supervised_row(sample,split='train')
    assert qualified['r_supervised'] is True
    assert qualified['label_provenance']['planar_relation_qualified'] is False


def test_prior_qc_does_not_reuse_anatomy_digest_for_lesion():
    dc=api()
    sample=row();sample.update(anatomy='/data/anatomy.nii',lesion='/data/lesion.nii')
    records=[{'source_root':'/data','anatomy_path':'/data/anatomy.nii','sha256':'a',
              'lesion_path':'/data/lesion.nii','lesion_sha256':'b'}]
    supplement=[{'path':'/data/lesion.nii','sha256':'b','qualified':True}]
    result=dc.verify_training_qc([sample],{'/data/anatomy.nii':'a','/data/lesion.nii':'b'},records,[],supplement=supplement)
    assert result['lesion_rows'] == 1
    with pytest.raises(ValueError,match='QC'):
        dc.verify_training_qc([sample],{'/data/anatomy.nii':'a','/data/lesion.nii':'b'},records,[])


def test_training_label_qc_rejects_unknown_classes_and_unproven_empty():
    dc=api()
    sample=row()
    with pytest.raises(ValueError,match='label'):
        dc.training_label_state(np.full((2,2,2),2,dtype=np.int16),sample)
    with pytest.raises(ValueError,match='empty'):
        dc.training_label_state(np.zeros((2,2,2),dtype=np.int16),sample)
    pdgm=dict(sample,source='pdgm',source_sequence='T1c',sequence='T1c',u_values=[1,4],a_ignore_values=[1,2,4])
    state=dc.training_label_state(np.full((2,2,2),2,dtype=np.int16),pdgm)
    assert state['lesion_state']=='sequence_empty'
    assert state['negative_truth'] is False
    assert state['sequence_visible_voxels']==0


def test_known_native_empty_reuses_matching_prior_lesion_hash_only():
    dc=api()
    sample=row(); sample.update(anatomy='/data/a.nii',lesion='/data/u.nii',u_supervised=False,r_supervised=False)
    records=[{'anatomy_path':'/data/a.nii','sha256':'a','lesion_path':'/data/u.nii','lesion_sha256':'u'}]
    assert dc.verify_training_qc([sample],{'/data/a.nii':'a','/data/u.nii':'u'},records,[])['known_source_empty_lesion_rows']==1
    records[0]['lesion_sha256']='wrong'
    with pytest.raises(ValueError,match='QC'):
        dc.verify_training_qc([sample],{'/data/a.nii':'a','/data/u.nii':'u'},records,[])


def test_replay_requires_role_files_to_be_committed(tmp_path):
    dc=api();out=tmp_path/'out';out.mkdir()
    committed_empty_roles(dc,out)
    c={'version':dc.VERSION,'inputs':{},'source_files':{},'source_snapshot':{},'outputs':{},'summary':{}}
    dc.write_private_json(out/'contract.json',c)
    dc.write_private_json(out/'complete.json',{'version':dc.VERSION,'contract_sha256':dc.file_sha256(out/'contract.json'),'outputs':{}})
    with pytest.raises(ValueError,match='integrity'):
        dc.verify_contract(out,check_paths=False)


def test_supplement_requires_ledger_and_summary_commitments(tmp_path):
    dc=api()
    dc.write_private_json(tmp_path/'complete.json',{'version':dc.VERSION,'outputs':{},'environment':dc.decoder_environment(),
                         'source_snapshot':{},'initial_contract_sha256':'1'*64})
    dc.write_private_json(tmp_path/'ledger_private.json',[])
    dc.write_private_json(tmp_path/'anonymous_summary.json',{})
    with pytest.raises(ValueError,match='QC'):
        dc.load_training_supplement(tmp_path,'1'*64)


@pytest.mark.parametrize('missing',['all','one'])
def test_supplement_requires_complete_execution_source_closure(tmp_path,missing):
    dc=api();outputs={}
    for name in ('ledger_private.json','anonymous_summary.json','scope_private.json'):
        dc.write_private_json(tmp_path/name,[] if name=='ledger_private.json' else {})
        outputs[name]=dc.file_sha256(tmp_path/name)
    snap=dc.source_snapshot()
    if missing=='all': snap={}
    else: snap.pop(next(iter(snap)))
    for name,digest in snap.items():
        dest=tmp_path/'execution_source'/name;dest.parent.mkdir(parents=True,exist_ok=True)
        dest.write_bytes((dc.REPO/name).read_bytes());outputs['execution_source/'+name]=digest
    dc.write_private_json(tmp_path/'complete.json',{'version':dc.VERSION,'outputs':outputs,'environment':dc.decoder_environment(),
                         'source_snapshot':snap,'initial_contract_sha256':'1'*64})
    with pytest.raises(ValueError,match='QC'):
        dc.load_training_supplement(tmp_path,'1'*64)


def test_role_image_must_resolve_to_its_committed_source(tmp_path):
    dc=api();out=tmp_path/'out';out.mkdir()
    first=tmp_path/'first';first.write_bytes(b'approved')
    other=tmp_path/'other';other.write_bytes(b'changed')
    alias=tmp_path/'alias';alias.symlink_to(first)
    outputs=committed_empty_roles(dc,out)
    member=out/'train_ssl.json'
    member.write_text(json.dumps([{'image':str(alias),'source_hashes':{'image':dc.file_sha256(first)}}]))
    outputs['train_ssl.json']=dc.file_sha256(member)
    c={'version':dc.VERSION,'inputs':{},'source_files':{str(first):dc.file_sha256(first)},'source_snapshot':{},'outputs':outputs,'summary':{}}
    dc.write_private_json(out/'contract.json',c)
    dc.write_private_json(out/'complete.json',{'version':dc.VERSION,'contract_sha256':dc.file_sha256(out/'contract.json'),'outputs':outputs})
    alias.unlink();alias.symlink_to(other)
    with pytest.raises(ValueError,match='integrity'):
        dc.verify_contract(out,check_paths=False)


@pytest.mark.parametrize('damage', [None, 'role_link', 'source', 'snapshot', 'environment','missing_environment'])
def test_evaluation_verifies_historical_source_and_current_role_links(tmp_path, monkeypatch, damage):
    dc = api()
    repo = tmp_path/'repo'; repo.mkdir()
    (repo/'module.py').write_text('new evaluator')
    monkeypatch.setattr(dc, 'REPO', repo)
    out = tmp_path/'qualified'; out.mkdir()
    frozen = out/'execution_source/module.py'; frozen.parent.mkdir()
    frozen.write_text('qualified source')
    source = tmp_path/'source'; source.write_bytes(b'approved source')
    outputs = committed_empty_roles(dc, out, {'execution_source/module.py': dc.file_sha256(frozen)})
    role = [{'image': str(source), 'source_hashes': {'image': dc.file_sha256(source)}}]
    if damage == 'role_link':
        role[0]['source_hashes']['image'] = '0'*64
    (out/'validation.json').write_text(json.dumps(role))
    outputs['validation.json'] = dc.file_sha256(out/'validation.json')
    contract = dict(version=dc.VERSION, inputs={}, source_files={str(source): dc.file_sha256(source)},
                    source_snapshot={'module.py': dc.file_sha256(frozen)}, outputs=outputs, summary={},
                    environment=dc.decoder_environment())
    if damage == 'environment': contract['environment'] = {'numpy': 'changed'}
    if damage == 'missing_environment': contract.pop('environment')
    dc.write_private_json(out/'contract.json', contract)
    dc.write_private_json(out/'complete.json', dict(version=dc.VERSION,
        contract_sha256=dc.file_sha256(out/'contract.json'), outputs=outputs))
    before = (out/'contract.json').read_bytes()
    if damage == 'source': source.write_bytes(b'changed')
    if damage == 'snapshot': frozen.write_text('changed')
    if damage is None:
        assert dc.verify_contract(out, check_paths=False, source_mode='evaluation') == contract
        monkeypatch.setattr(dc, 'private_output', lambda path: Path(path).resolve())
        monkeypatch.setattr(dc, 'source_path', lambda path: Path(path).resolve())
        assert dc.read_contract(out, source_mode='evaluation')['validation'] == role
    else:
        with pytest.raises(ValueError, match='integrity'):
            dc.verify_contract(out, check_paths=False, source_mode='evaluation')
    with pytest.raises(ValueError, match='integrity'):
        dc.verify_contract(out, check_paths=False)
    assert (out/'contract.json').read_bytes() == before


@pytest.mark.parametrize('prefix', ['training_qc/execution_source', None])
def test_evaluation_accepts_exact_committed_or_external_qualified_source(tmp_path, monkeypatch, prefix):
    dc = api(); out = tmp_path/'out'; out.mkdir()
    live = tmp_path/'live'; live.mkdir(); (live/'module.py').write_text('changed live source')
    monkeypatch.setattr(dc,'REPO',live)
    archive = tmp_path/'archive'; archive.mkdir(); (archive/'module.py').write_text('historical source')
    expected = dc.file_sha256(archive/'module.py')
    outputs = committed_empty_roles(dc,out)
    if prefix:
        dest=out/prefix/'module.py'; dest.parent.mkdir(parents=True); dest.write_text('historical source')
        outputs[prefix+'/module.py']=expected
    c = dict(version=dc.VERSION,inputs={},source_files={},source_snapshot={'module.py':expected},
             outputs=outputs,summary={},environment=dc.decoder_environment())
    dc.write_private_json(out/'contract.json',c)
    dc.write_private_json(out/'complete.json',dict(version=dc.VERSION,
        contract_sha256=dc.file_sha256(out/'contract.json'),outputs=outputs))
    if prefix:
        assert dc.verify_contract(out,check_paths=False,source_mode='evaluation') == c
    else:
        with pytest.raises(ValueError,match='integrity'):
            dc.verify_contract(out,check_paths=False,source_mode='evaluation')
    assert dc.verify_contract(out,check_paths=False,source_mode='evaluation',qualified_source_root=archive) == c
    if not prefix:
        (archive/'module.py').write_text('wrong archive revision')
        with pytest.raises(ValueError,match='integrity'):
            dc.verify_contract(out,check_paths=False,source_mode='evaluation',qualified_source_root=archive)

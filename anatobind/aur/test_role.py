"""Immutable evaluation-only historical-test successor; training roles stay sealed.

This builder is deliberately separate from the original data-contract module.
Real historical arrays are opened only by an explicit final-gate build.
"""
from collections import Counter
from pathlib import Path

import nibabel as nib
from nibabel.processing import resample_from_to
import numpy as np

from anatobind.aur import data_contract as dc
from anatobind.aur.labels import discrete_labels
from anatobind.aur.resample import _target_grid

__test__ = False
VERSION = 'aur_historical_test_role_v1'
ROLE = 'historical_test'
ROLE_FILE = 'historical_test.json'
SNAPSHOT_FILES = ('anatobind/aur/test_role.py','anatobind/aur/data_contract.py','anatobind/aur/labels.py',
                  'anatobind/aur/resample.py','anatobind/aur/samples.py')


def _snapshot():
    return {name:dc.file_sha256(dc.REPO/name) for name in SNAPSHOT_FILES}


def test_rows_from_current(current_rows, native_rows):
    if any(row.get('split')=='test' and row.get('source') not in {'pdgm','bmsr','isles','sibbms'}
           for row in [*current_rows,*native_rows]):raise ValueError('Unsupported historical source')
    dc.validate_patient_roles(current_rows,[],{})
    dc.validate_patient_roles(native_rows,[],{})
    native={dc.identity(row):row for row in native_rows}
    if len(native)!=len(native_rows):raise ValueError('Historical native identity conflict')
    result=[]
    for row in current_rows:
        if row['split']!='test':continue
        original=native.get(dc.identity(row))
        if original is None or original['split']!='test':raise ValueError('Historical native split conflict')
        qualified=dc.supervised_row(row,split='test',native_status='historical_metadata_only');qualified['role']=ROLE
        result.append(qualified)
    if not result:raise ValueError('Historical test role is empty')
    return result


def repair_isles_test_label(row, native_row, out_dir, ledger):
    if row['source']!='isles' or not row.get('lesion'):return row
    key=str(dc.source_path(native_row['lesion']))
    if key in ledger['files']:
        known=ledger['files'][key]
        result=dc.supervised_row(row,split='test',native_status=known['state'])
        result.update(role=ROLE,lesion=known['lesion'])
        return result
    image=nib.load(str(dc.source_path(row['image'])));dc.inspect_geometry(image)
    current=dc.inspect_label(dc.source_path(row['lesion']),image)
    native_image=nib.load(key);dc.inspect_geometry(native_image)
    values=dc.inspect_label(key)
    if not set(map(int,np.unique(values)))<={0,1}:raise ValueError('Invalid historical lesion classes')
    anatomy=nib.load(str(dc.source_path(native_row['anatomy'])));dc.inspect_geometry(anatomy)
    grid=_target_grid(anatomy)
    restored=discrete_labels(np.asarray(resample_from_to(nib.Nifti1Image(values,native_image.affine),grid,
                                order=0,mode='constant',cval=0).dataobj))
    if restored.shape!=image.shape or not np.allclose(grid[1],image.affine,atol=1e-3,rtol=1e-5):
        raise ValueError('Historical target grid differs')
    positive=bool(np.any(values))
    if positive and not np.any(restored):
        raise ValueError('Historical native-positive lesion has no foreground on target grid')
    changed=not np.array_equal(current,restored)
    lesion=row['lesion'];state='source_positive' if positive else 'source_empty'
    if changed:
        folder=Path(out_dir)/'test_repair_v1';folder.mkdir(mode=0o700,exist_ok=True)
        destination=folder/f'label_{ledger["counts"]["isles_repaired_cases"]:04d}.nii.gz'
        if destination.exists():raise ValueError('Historical repair already exists')
        result=nib.Nifti1Image(restored,grid[1]);result.header.set_xyzt_units('mm');nib.save(result,str(destination));destination.chmod(0o600)
        dc.inspect_label(destination,image);lesion=str(destination.resolve())
        ledger['counts']['isles_repaired_cases']+=1
        ledger['counts']['isles_restored_voxels']+=int(np.count_nonzero(restored))-int(np.count_nonzero(current))
        if positive:state='native_positive_repaired'
    ledger['counts']['isles_cases']+=1
    ledger['counts']['isles_native_empty_cases']+=int(not positive)
    ledger['files'][key]={'lesion':lesion,'state':state}
    output=dc.supervised_row(row,split='test',native_status=state);output.update(role=ROLE,lesion=lesion)
    return output


def build_historical_test_role(parent_dir,out_dir,*,current=dc.DATA_ROOT/'derived/aur/samples_1mm_v1.json',
                               native=dc.DATA_ROOT/'derived/aur/samples.json',qualified_source_root=None,workers=2):
    out=dc.private_output(out_dir);parent=dc.private_output(parent_dir)
    if parent.is_file():parent=parent.parent
    if out.exists() or out==parent:raise ValueError('Historical output already exists')
    current=dc.source_path(current);native=dc.source_path(native)
    frozen=dc.read_contract(parent,source_mode='evaluation',qualified_source_root=qualified_source_root)
    input_hashes={str(path):dc.file_sha256(path) for path in (current,native)}
    current_rows,native_rows=dc.read_json(current),dc.read_json(native)
    held={dc.patient_group(row) for key in dc.ROLE_FILES for row in frozen[key]}
    if held&{dc.patient_group(row) for row in current_rows if row.get('split')=='test'}:
        raise ValueError('Historical patients intersect parent roles')
    selected=test_rows_from_current(current_rows,native_rows)
    original={dc.identity(row):row for row in native_rows}
    snapshot=_snapshot();parent_hash=dc.file_sha256(parent/'contract.json');parent_complete=dc.file_sha256(parent/'complete.json')
    if parent_hash!=frozen['contract_sha256']:raise ValueError('Historical parent changed')
    policy={'historical_test_dense':'opened_for_final_gate_only',
            'label_arrays_read_for':'categorical_qc_native_lesion_qualification_and_isles_repair',
            'model_evaluations_allowed':1,'prior_test_exposure':'historically_exposed',
            'decoder':'discrete_labels','discrete_atol':1e-5,'unknown_U_R':'not_healthy_or_no_host'}
    dc.write_private_json(out/'policy.json',policy)
    ledger={'files':{},'counts':Counter(isles_cases=0,isles_repaired_cases=0,isles_native_empty_cases=0,isles_restored_voxels=0)}
    paths={str(current),str(native)};rows=[]
    for row in selected:
        for key in ('image','anatomy','lesion'):
            if row.get(key):paths.add(str(dc.source_path(row[key])))
        original_row=original[dc.identity(row)]
        if original_row.get('lesion'):paths.add(str(dc.source_path(original_row['lesion'])))
        if row['source']=='isles':paths.add(str(dc.source_path(original_row['anatomy'])))
    hashes=dc.hash_files(paths,workers=workers)
    if any(hashes[path]!=digest for path,digest in input_hashes.items()):raise ValueError('Historical metadata inputs changed')
    for row in selected:
        image=nib.load(str(dc.source_path(row['image'])));geometry=dc.inspect_geometry(image)
        dc.inspect_label(dc.source_path(row['anatomy']),image)
        original_row=original[dc.identity(row)]
        if row.get('lesion'):
            current_values=dc.inspect_label(dc.source_path(row['lesion']),image)
            native_values=dc.inspect_label(dc.source_path(original_row['lesion']))
            allowed={'pdgm':{0,1,2,4},'bmsr':{0,1},'isles':{0,1}}.get(row['source'])
            if allowed is None or not set(map(int,np.unique(current_values)))<=allowed or not set(map(int,np.unique(native_values)))<=allowed:
                raise ValueError('Invalid historical lesion classes')
            source_positive=bool(np.isin(native_values,row['a_ignore_values']).any())
            visible_native=bool(np.isin(native_values,row['u_values']).any())
            visible_current=bool(np.isin(current_values,row['u_values']).any())
            state='source_empty' if not source_positive else ('sequence_empty' if row['u_supervised'] and not visible_native else 'source_positive')
            if row['source']=='isles':
                row=repair_isles_test_label(row,original_row,out,ledger)
            else:
                if (visible_native and not visible_current) or (not source_positive and np.any(current_values)):
                    raise ValueError('Historical lesion conversion loss requires a new qualified repair')
                row=dc.supervised_row(row,split='test',native_status=state);row['role']=ROLE
        rows.append({**row,'geometry':geometry})
    if dc.hash_files(paths,workers=workers)!=hashes:raise ValueError('Historical source changed during qualification')
    for row in rows:
        row['source_hashes']={}
        for key in ('image','anatomy','lesion'):
            if row.get(key):
                path=Path(row[key]).expanduser().resolve();row[key]=str(path)
                row['source_hashes'][key]=hashes.get(str(path)) or dc.file_sha256(path)
    summary={**dc.coverage(rows),'version':VERSION,'role':ROLE,'repair':dict(ledger['counts']),
             'prior_test_exposure':'historically_exposed','model_evaluations_allowed':1,'evidence':'NOT_EVIDENCE'}
    dc.write_private_json(out/ROLE_FILE,rows);dc.write_private_json(out/'anonymous_summary.json',summary)
    execution={}
    for name,digest in snapshot.items():
        destination=out/'execution_source'/name
        missing=[];folder=destination.parent
        while not folder.exists():missing.append(folder);folder=folder.parent
        for folder in reversed(missing):folder.mkdir(mode=0o700)
        destination.write_bytes((dc.REPO/name).read_bytes());destination.chmod(0o600)
        if dc.file_sha256(destination)!=digest:raise ValueError('Historical execution source changed')
        execution['execution_source/'+name]=digest
    outputs={str(path.relative_to(out)):dc.file_sha256(path) for path in out.rglob('*') if path.is_file()}
    if (any(hashes[path]!=digest for path,digest in input_hashes.items()) or _snapshot()!=snapshot or dc.file_sha256(parent/'contract.json')!=parent_hash
            or dc.file_sha256(parent/'complete.json')!=parent_complete):raise ValueError('Historical inputs changed')
    contract={'version':VERSION,'parent':{'path':str(parent),'sha256':parent_hash,'complete_sha256':parent_complete},
              'source_files':hashes,'outputs':outputs,'source_snapshot':snapshot,'environment':dc.decoder_environment()}
    dc.write_private_json(out/'contract.json',contract)
    dc.write_private_json(out/'complete.json',{'version':VERSION,'contract_sha256':dc.file_sha256(out/'contract.json'),'outputs':outputs})
    return summary


def read_test_role(out_dir,*,qualified_source_root=None):
    try:
        out=dc.private_output(out_dir)
        if out.is_file():out=out.parent
        contract=dc.read_json(out/'contract.json');complete=dc.read_json(out/'complete.json')
        if (contract['version']!=VERSION or complete['version']!=VERSION
                or complete['contract_sha256']!=dc.file_sha256(out/'contract.json')
                or complete['outputs']!=contract['outputs'] or contract['environment']!=dc.decoder_environment()
                or not {ROLE_FILE,'policy.json','anonymous_summary.json'}<=set(contract['outputs'])
                or set(contract['source_snapshot'])!=set(SNAPSHOT_FILES)):raise ValueError()
        parent=dc.private_output(contract['parent']['path'])
        if (str(parent)!=contract['parent']['path'] or dc.file_sha256(parent/'contract.json')!=contract['parent']['sha256']
                or dc.file_sha256(parent/'complete.json')!=contract['parent']['complete_sha256']):raise ValueError()
        frozen=dc.read_contract(parent,source_mode='evaluation',qualified_source_root=qualified_source_root)
        if frozen['contract_sha256']!=contract['parent']['sha256']:raise ValueError()
        for name,digest in contract['outputs'].items():
            path=(out/name).resolve()
            if not path.is_relative_to(out) or dc.file_sha256(path)!=digest:raise ValueError()
        for name,digest in contract['source_snapshot'].items():
            if contract['outputs'].get('execution_source/'+name)!=digest:raise ValueError()
        for name,digest in contract['source_files'].items():
            path=dc.source_path(name)
            if str(path)!=name or dc.file_sha256(path)!=digest:raise ValueError()
        references={**contract['source_files'],**{str((out/name).resolve()):digest for name,digest in contract['outputs'].items()}}
        rows=dc.read_json(out/ROLE_FILE);seen=set()
        groups={dc.patient_group(row) for key in dc.ROLE_FILES for row in frozen[key]}
        for row in rows:
            key=dc.identity(row)
            if row['role']!=ROLE or row['split']!='test' or key in seen or dc.patient_group(row) in groups:raise ValueError()
            seen.add(key)
            for field in ('image','anatomy','lesion'):
                if row.get(field):
                    path=str(Path(row[field]).expanduser().resolve())
                    if path!=row[field] or references.get(path)!=row['source_hashes'][field] or dc.file_sha256(path)!=references[path]:raise ValueError()
        if not rows:raise ValueError()
        return {'rows':rows,'contract_sha256':dc.file_sha256(out/'contract.json'),
                'parent_contract_sha256':contract['parent']['sha256']}
    except Exception:
        raise ValueError('Historical test role verification failed') from None

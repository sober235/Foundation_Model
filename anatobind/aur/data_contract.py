"""Frozen, private data roles for unified AUR training.

Qualification reads labels; Stage I receives a separate image-only allowlist.
No function in this module reports patient identifiers or input paths in errors.
"""
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
import platform
import shutil
from pathlib import Path
import time

import nibabel as nib
from nibabel.processing import resample_from_to
import numpy as np
import scipy

from anatobind.aur.labels import discrete_labels
from anatobind.aur.resample import _target_grid
from anatobind.aur.samples import ALL_LESION_VALUES

VERSION = 'aur_data_contract_v1'
DATA_ROOT = Path('/data2/congcong/data/FM_data')
PRIVATE_ROOT = Path('/home/congcongliu')
REPAIR_ROOT = PRIVATE_ROOT/'outputs/aur_data_audit_20261019/labels/repair_train_v1'
REPAIRED_TRAIN = REPAIR_ROOT/'samples_train_repaired_v1_private.json'
REPAIRED_TRAIN_SHA = '40feb9b06a56657935a62873a7e52bb57446935b6b107e2a31f7f24fccd24f28'
REPAIR_COMPLETE_SHA = 'f6d0b1de9eb01bb3234bb3571fac6ac48522f6cff0217f3f792ac0dc9b7fbb46'
REPO = Path(__file__).resolve().parents[2]
PRIOR_QC_ROOT = REPAIR_ROOT.parent
PRIOR_QC_FILES = tuple(PRIOR_QC_ROOT/name for name in ('census_records_private.json','census_scope_private.json','audit_complete.json'))
PRIOR_QC_COMPLETE_SHA = '9456df66daa09a28b97117a4eebd4aef21deea2644a0a97fe104a3f76f829506'
SUPERVISED_FIELDS = ('source','patient','case','sequence','source_sequence','image','anatomy','lesion',
                     'a_supervised','u_supervised','r_supervised','u_values','a_ignore_values',
                     'native_spacing','resampled','thick_slice')
IMAGE_FIELDS = ('source','patient','case','sequence','source_sequence','image','native_spacing','resampled','thick_slice')
ROLE_FILES = {'train_supervised':'train_supervised.json','train_ssl':'train_ssl.json',
              'validation':'validation.json','validation_ssl':'validation_ssl.json'}


def file_sha256(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''): h.update(block)
    return h.hexdigest()


def json_digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def private_output(path, *, private_root=PRIVATE_ROOT):
    path=Path(path).expanduser().resolve()
    if not path.is_relative_to(Path(private_root).resolve()) or path == Path(private_root).resolve():
        raise ValueError('Output requires a private home directory')
    if any((parent/'.git').is_file() or (parent/'.git/HEAD').is_file()
           or ((parent/'HEAD').is_file() and (parent/'objects').is_dir() and (parent/'config').is_file())
           for parent in (path,*path.parents)):
        raise ValueError('Output requires a private directory outside Git')
    return path


# Public spelling used by formal runners and their actual writers.
validate_private_output = private_output


def source_path(path, *, data_root=DATA_ROOT, repair_root=REPAIR_ROOT):
    path=Path(path).expanduser().resolve()
    approved_evidence={p.resolve() for p in PRIOR_QC_FILES}
    if path not in approved_evidence and not any(path.is_relative_to(Path(root).resolve()) for root in (data_root,repair_root)):
        raise ValueError('Input source is outside approved roots')
    return path


def write_private_json(path, value):
    path=Path(path)
    missing=[]; parent=path.parent
    while not parent.exists(): missing.append(parent); parent=parent.parent
    for directory in reversed(missing): directory.mkdir(mode=0o700)
    if (path.parent.stat().st_mode & 0o077):
        # Only the requested artifact directory is hardened, never a shared ancestor.
        path.parent.chmod(0o700)
    with os.fdopen(os.open(path,os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600),'w') as f:
        json.dump(value,f,sort_keys=True,indent=1); f.write('\n')


def patient_group(row):
    for key in ('source','patient','case','sequence','source_sequence'):
        if key not in row or row[key] is None or not str(row[key]):
            raise ValueError('Invalid patient metadata')
    return f"{row['source']}:{row['patient']}"


def identity(row):
    return (row['source'],str(row['patient']),str(row['case']),row['source_sequence'])


def validate_patient_roles(current, ssl, val, *, repaired_train=None):
    validation={f'{s}:{p}' for s,patients in val.items() for p in patients}
    splits={}; observations={}
    for row in current:
        group=patient_group(row); split=row.get('split')
        if split not in ('train','test'): raise ValueError('Invalid source split')
        if group in splits and splits[group]!=split: raise ValueError('Patient split conflict')
        splits[group]=split
        key=identity(row)
        if key in observations: raise ValueError('Duplicate source observation')
        observations[key]=row
        if split=='test' and group in validation: raise ValueError('Historical test intersects validation')
    for row in ssl:
        group=patient_group(row)
        expected='val' if group in validation else 'train'
        if row.get('split')!='train' or row.get('ssl_split')!=expected or splits.get(group)=='test':
            raise ValueError('SSL patient split conflict')
    if repaired_train is not None:
        expected={identity(r) for r in current if r['split']=='train' and patient_group(r) not in validation}
        actual=set()
        for row in repaired_train:
            if patient_group(row) in validation: raise ValueError('Repaired TRAIN intersects validation')
            key=identity(row)
            if key not in expected or row.get('split')!='train' or key in actual:
                raise ValueError('Repaired TRAIN identity split conflict')
            actual.add(key)
        if expected!=actual: raise ValueError('Repaired TRAIN membership differs')
    return validation


def common_row(row, split, role, fields):
    values={k:row[k] for k in fields if k in row}
    for key in ('image','anatomy','lesion'):
        if values.get(key): values[key]=str(Path(values[key]).expanduser().resolve())
    return {**values,'patient_group':patient_group(row),
            'split':split,'role':role,'input_modes':['2d','3d'],
            'identity_scope':'source_qualified_not_biological_deduplication'}


def image_only_row(row, *, split):
    return common_row(row,split,'train_ssl' if split=='train' else 'ssl_validation',IMAGE_FIELDS)


def supervised_row(row, *, split, native_status='existing_train_qualification'):
    out=common_row(row,split,'train_supervised' if split=='train' else 'development_validation',SUPERVISED_FIELDS)
    a=bool(row.get('a_supervised',True)); u=bool(row.get('u_supervised',False))
    r=bool(row.get('r_supervised',True)) and a and u
    if native_status=='source_empty': u=r=False
    out.update(a_supervised=a,u_supervised=u,r_supervised=r)
    out['qualification_reason']={'a':'thick_slice_disabled' if not a else 'existing_anatomy_pseudo_label',
        'u':('source_empty_unknown' if native_status=='source_empty' else 'sequence_visible' if u else 'unknown_or_unannotated_sequence'),
        'r':('geometry_pseudo_3d_only' if r else 'unknown_or_unqualified')}
    out['label_provenance']={'anatomy':'SynthSeg_pseudo_label','relation':'geometry_pseudo_NOT_EVIDENCE' if r else 'unknown',
                             'lesion_state':native_status,'negative_truth':False,
                             'planar_relation_qualified':False,
                             'u_reference':'sequence_conditioned_disease_annotation_not_healthy_truth'}
    return out


def inspect_geometry(img):
    affine=np.asarray(img.affine,dtype=np.float64)
    unit=img.header.get_xyzt_units()[0]
    if len(img.shape)!=3 or min(img.shape)<=0 or unit not in ('mm','unknown') or not np.isfinite(affine).all():
        raise ValueError('Invalid image geometry or physical units')
    spacing=np.linalg.norm(affine[:3,:3],axis=0)
    determinant=abs(float(np.linalg.det(affine[:3,:3])))
    if not np.isfinite(determinant) or determinant<=0 or not np.isfinite(spacing).all() or (spacing<=0).any():
        raise ValueError('Invalid image geometry or physical units')
    return {'shape':list(img.shape),'spacing_mm':spacing.tolist(),'spatial_units':unit,
            'units_provenance':'declared_mm' if unit=='mm' else 'legacy_unknown_assumed_mm',
            'affine_sha256':json_digest(affine.tolist())}


def inspect_label(path, image=None, *, decode=True):
    try:
        label=nib.load(str(path)); inspect_geometry(label)
        if image is not None:
            if label.shape!=image.shape or not np.allclose(label.affine,image.affine,atol=1e-3,rtol=1e-5):
                raise ValueError('Invalid label grid')
        return discrete_labels(np.asarray(label.dataobj)) if decode else None
    except Exception:
        raise ValueError('Invalid label data or geometry') from None


def freeze_policy(out, validation):
    members=[{k:r[k] for k in ('source','patient','case','source_sequence')} for r in validation]
    isles={patient_group(r) for r in validation if r['source']=='isles'}
    write_private_json(Path(out)/'validation_membership_private.json',members)
    policy={'version':VERSION,'discrete_atol':1e-5,'decoder':'discrete_labels',
            'isles_selection':'all_development_validation_patients','isles_patient_count':len(isles),
            'membership_sha256':json_digest(members),'source_empty':'unknown_U_R_disabled',
            'pdgm_sequence_empty':'retain_sequence_supervision_explicit_provenance_not_healthy',
            'repair_grid':'_target_grid(native_anatomy)','historical_test_dense':'sealed',
            'model_predictions_read':False,'new_development_label_QC_exposure':True}
    write_private_json(Path(out)/'policy.json',policy)
    return policy


def source_snapshot():
    names=['anatobind/aur/data_contract.py','scripts/aur_qualify_training_data.py',
           'anatobind/aur/labels.py','anatobind/aur/resample.py','anatobind/aur/samples.py']
    return {name:file_sha256(REPO/name) for name in names}


def hash_files(paths, workers=2):
    paths=sorted(set(map(str,paths)))
    with ThreadPoolExecutor(max_workers=min(2,max(1,workers))) as pool:
        return dict(zip(paths,pool.map(file_sha256,paths)))


def coverage(rows):
    result={'rows':len(rows),'patients':len({patient_group(r) for r in rows}),'sources':{}}
    for source in sorted({r['source'] for r in rows}):
        group=[r for r in rows if r['source']==source]
        result['sources'][source]={'rows':len(group),'patients':len({patient_group(r) for r in group})}
        if any('u_supervised' in r for r in group):
            result['sources'][source].update({f'{task}_eligible_rows':sum(bool(r.get(f'{task}_supervised')) for r in group) for task in ('a','u','r')})
    return result


def decoder_environment():
    return {'python':platform.python_version(),'numpy':np.__version__,'nibabel':nib.__version__,
            'scipy':scipy.__version__,'discrete_atol':1e-5}


def strings(value):
    """Internal traversal; private keys and values are never printed."""
    if isinstance(value,dict):
        for item in value.values(): yield from strings(item)
    elif isinstance(value,list):
        for item in value: yield from strings(item)
    elif isinstance(value,str): yield value


def prior_qc():
    """Load the identified prior QC records, preserving their provenance limits."""
    if file_sha256(PRIOR_QC_FILES[2])!=PRIOR_QC_COMPLETE_SHA:
        raise ValueError('Prior QC completion differs')
    if file_sha256(REPAIR_ROOT/'complete.json')!=REPAIR_COMPLETE_SHA:
        raise ValueError('Prior repair completion differs')
    committed=set(strings(read_json(REPAIR_ROOT/'complete.json')))
    candidates=[]
    for path in REPAIR_ROOT.glob('*.json'):
        if path.name.startswith('samples_'): continue
        value=read_json(path)
        if isinstance(value,list) and value and all('new_path' in item and 'new_sha256' in item for item in value):
            if file_sha256(path) not in committed: raise ValueError('Prior repair ledger integrity differs')
            candidates.append((path,value))
    if len(candidates)!=1: raise ValueError('Prior repair ledger is ambiguous')
    path,repairs=candidates[0]
    return read_json(PRIOR_QC_FILES[0]),repairs,[*PRIOR_QC_FILES,path]


def verify_training_qc(train, hashes, records, repairs, *, supplement=()):
    """Reuse label-array QC only where the current bytes match its recorded digest."""
    anatomy={}; lesion_hashes={}
    for record in records:
        for field,digest_field,bucket in (('anatomy_path','sha256',anatomy),('lesion_path','lesion_sha256',lesion_hashes)):
            if not record.get(field) or not record.get(digest_field): continue
            path=str(Path(record[field]).resolve()); digest=record[digest_field]
            if path in bucket and bucket[path]!=digest: raise ValueError('Prior QC digest conflict')
            bucket[path]=digest
    restored={str(Path(r['new_path']).resolve()):r['new_sha256'] for r in repairs}
    additional={}
    for record in supplement:
        if record.get('qualified') is not True: raise ValueError('TRAIN supplemental QC is incomplete')
        path=str(Path(record['path']).resolve())
        if path in additional: raise ValueError('Duplicate TRAIN supplemental QC')
        additional[path]=record['sha256']
    native_empty={str(Path(r['lesion']).resolve()) for r in train if r.get('lesion') and
                  r['source']=='isles' and r['sequence']=='DWI' and not r['u_supervised']}
    counts=Counter()
    for row in train:
        for key in ('anatomy','lesion'):
            if not row.get(key): continue
            path=str(Path(row[key]).resolve())
            if key=='anatomy': expected=anatomy.get(path)
            elif path in restored: expected=restored[path]
            elif path in native_empty: expected=lesion_hashes.get(path)
            else: expected=additional.get(path)
            if expected is None or hashes.get(path)!=expected: raise ValueError('TRAIN prior QC integrity differs')
            counts[key+'_rows']+=1
            counts['repaired_lesion_rows']+=int(key=='lesion' and path in restored)
            counts['known_source_empty_lesion_rows']+=int(key=='lesion' and path in native_empty)
            counts['supplemental_lesion_rows']+=int(key=='lesion' and path in additional)
    return dict(counts)


def training_label_state(values, row):
    """Validate source categories and preserve sequence-conditioned disease labels."""
    values=discrete_labels(values)
    allowed=ALL_LESION_VALUES.get(row['source'])
    if allowed is None or not np.isin(values,(0,*allowed)).all():
        raise ValueError('Unexpected TRAIN label category')
    if not np.any(values): raise ValueError('Unproven empty TRAIN disease label')
    if not set(row.get('u_values',())).issubset(allowed) or not set(row.get('a_ignore_values',())).issubset(allowed):
        raise ValueError('Invalid TRAIN label sequence values')
    visible=int(np.count_nonzero(np.isin(values,row.get('u_values',()))))
    state='sequence_empty' if row['u_supervised'] and visible==0 else 'source_positive'
    if state=='sequence_empty' and row['source']!='pdgm':
        raise ValueError('Unproven empty TRAIN sequence label')
    return {'lesion_state':state,'sequence_visible_voxels':visible,'negative_truth':False}


def _save_execution_source(out, snapshot):
    for name,digest in snapshot.items():
        dest=Path(out)/'execution_source'/name
        dest.parent.mkdir(parents=True,exist_ok=True,mode=0o700)
        shutil.copyfile(REPO/name,dest);dest.chmod(0o600)
        if file_sha256(dest)!=digest: raise ValueError('Execution source changed')


def qualify_training_arrays(out, initial):
    """Decode the missing unique TRAIN U arrays once, inheriting frozen A QC.

    Previously repaired and source-empty labels retain their original evidence.
    The old execution is immutable; this supplemental execution has its own code.
    """
    out=private_output(out);initial=private_output(initial)
    if out.parent!=initial or out.exists(): raise ValueError('Invalid supplemental QC destination')
    old=_verify_tree(initial,check_paths=True,read_sources=False,frozen_execution=True)
    if file_sha256(REPAIRED_TRAIN)!=old['inputs'][str(REPAIRED_TRAIN)]: raise ValueError('Frozen TRAIN input differs')
    train=read_json(REPAIRED_TRAIN);records,repairs,evidence=prior_qc()
    restored={str(Path(r['new_path']).resolve()) for r in repairs}
    empty={str(Path(r['lesion']).resolve()) for r in train if r.get('lesion') and
           r['source']=='isles' and r['sequence']=='DWI' and not r['u_supervised']}
    groups={}
    for row in train:
        if row.get('lesion'):
            path=str(source_path(row['lesion']))
            if path not in restored|empty: groups.setdefault(path,[]).append(row)
    snapshot=source_snapshot();started=time.monotonic()
    out.mkdir(mode=0o700);_save_execution_source(out,snapshot)
    write_private_json(out/'scope_private.json',{'initial_contract_sha256':file_sha256(initial/'contract.json'),
        'paths':sorted(groups),'expected_hashes':{p:old['source_files'][p] for p in groups},
        'reused_repaired_unique':len(restored),'reused_native_empty_unique':len(empty)})
    ledger,counts,sequence_counts,source_counts=_inspect_training_groups(groups,old['source_files'])
    checks=verify_training_qc(train,old['source_files'],records,repairs,supplement=ledger)
    if source_snapshot()!=snapshot: raise ValueError('Supplemental QC source changed')
    summary={'unique_arrays':len(ledger),'rows':counts['rows'],'source_unique_arrays':dict(source_counts),
             'sequence_states':dict(sequence_counts),'prior_qc_reuse':checks,'seconds':round(time.monotonic()-started,3),
             'historical_test_dense_read':False,'validation_arrays_reread':False}
    write_private_json(out/'ledger_private.json',ledger);write_private_json(out/'anonymous_summary.json',summary)
    outputs={str(p.relative_to(out)):file_sha256(p) for p in out.rglob('*') if p.is_file()}
    write_private_json(out/'complete.json',{'version':VERSION,'outputs':outputs,'source_snapshot':snapshot,
        'environment':decoder_environment(),'initial_contract_sha256':file_sha256(initial/'contract.json')})
    return summary


def _inspect_training_groups(groups, hashes):
    ledger=[];counts=Counter();sequence_counts=Counter();source_counts=Counter()
    for path,rows in sorted(groups.items()):
        stat=Path(path).stat();fingerprint=(stat.st_size,stat.st_mtime_ns,stat.st_ino)
        digest=file_sha256(path)
        if digest!=hashes[path]: raise ValueError('Frozen TRAIN U integrity differs')
        label=nib.load(path);inspect_geometry(label)
        values=inspect_label(path);states=[]
        for row in rows:
            image=nib.load(str(source_path(row['image'])));inspect_geometry(image)
            if label.shape!=image.shape or not np.allclose(label.affine,image.affine,atol=1e-3,rtol=1e-5):
                raise ValueError('Invalid TRAIN label grid')
            state=training_label_state(values,row);states.append({'identity':list(identity(row)),**state})
            sequence_counts[state['lesion_state']]+=1
        now=Path(path).stat()
        if fingerprint!=(now.st_size,now.st_mtime_ns,now.st_ino): raise ValueError('TRAIN U changed during QC')
        ledger.append({'path':path,'sha256':digest,'qualified':True,'source':rows[0]['source'],
                       'categories':np.unique(values).tolist(),'states':states})
        source_counts[rows[0]['source']]+=1;counts['unique_arrays']+=1;counts['rows']+=len(rows)
        if len(ledger)%50==0:
            print(json.dumps({'training_u_qc_processed':len(ledger),'training_u_qc_total':len(groups)},sort_keys=True),flush=True)
    return ledger,counts,sequence_counts,source_counts


def load_training_supplement(path, initial_contract_sha256):
    path=Path(path).resolve();complete=read_json(path/'complete.json')
    if (complete['version']!=VERSION or complete['initial_contract_sha256']!=initial_contract_sha256
            or complete['environment']!=decoder_environment()):
        raise ValueError('Supplemental QC contract mismatch')
    required={'ledger_private.json','anonymous_summary.json','scope_private.json'}
    if not required<=set(complete['outputs']):
        raise ValueError('Supplemental QC commitments missing')
    if set(complete['source_snapshot'])!=set(source_snapshot()):
        raise ValueError('Supplemental QC execution source closure is incomplete')
    for name,digest in complete['outputs'].items():
        member=(path/name).resolve()
        if not member.is_relative_to(path) or file_sha256(member)!=digest:
            raise ValueError('Supplemental QC integrity differs')
    for name,digest in complete['source_snapshot'].items():
        if complete['outputs'].get('execution_source/'+name)!=digest:
            raise ValueError('Supplemental QC source commitment missing')
    return read_json(path/'ledger_private.json'),read_json(path/'anonymous_summary.json')


def _verify_tree(out, *, check_paths, read_sources=True, frozen_execution=False,
                 evaluation_source=False, qualified_source_root=None, cache=None, visited=None):
    cache={} if cache is None else cache
    visited=set() if visited is None else visited
    out=Path(out).resolve()
    if out in visited: raise ValueError()
    visited.add(out)
    if check_paths: private_output(out)

    def digest(path):
        path=Path(path).resolve()
        if path not in cache: cache[path]=file_sha256(path)
        return cache[path]

    contract=read_json(out/'contract.json'); complete=read_json(out/'complete.json')
    # Require an environment on the evaluated leaf. Older immutable parents
    # may predate that field; check their environment below when recorded.
    if evaluation_source and not frozen_execution and 'environment' not in contract: raise ValueError()
    if contract['version']!=VERSION or complete['version']!=VERSION or complete['contract_sha256']!=digest(out/'contract.json'):
        raise ValueError()
    if complete['outputs']!=contract['outputs']: raise ValueError()
    if not {*ROLE_FILES.values(),'anonymous_summary.json'}<=set(contract['outputs']): raise ValueError()
    approved=set()
    inherited=contract.get('inherited_contract')
    if inherited:
        parent_path=Path(inherited['path']).resolve()
        # A metadata-only version can inherit only its immutable direct parent.
        if parent_path!=(out.parent/'contract.json').resolve(): raise ValueError()
        if digest(parent_path)!=inherited['sha256'] or digest(parent_path.parent/'complete.json')!=inherited['complete_sha256']:
            raise ValueError()
        parent=_verify_tree(parent_path.parent,check_paths=check_paths,read_sources=read_sources,
                            frozen_execution=True,evaluation_source=evaluation_source,
                            qualified_source_root=qualified_source_root,cache=cache,visited=visited)
        approved={str((parent_path.parent/name).resolve()) for name in parent['outputs']}
        approved.update((str(parent_path),str(parent_path.parent/'complete.json')))
    for name,expected in contract['outputs'].items():
        member=(out/name).resolve()
        if not member.is_relative_to(out) or digest(member)!=expected: raise ValueError()
    if not frozen_execution or evaluation_source:
        # Check what the loaders will open, rather than only the earlier target
        # of a symlink at qualification time. Local repairs must have an output
        # commitment in this contract or its verified parent.
        reference_hashes={str(Path(p).resolve()):h for p,h in contract['source_files'].items()}
        reference_hashes.update({str((out/p).resolve()):h for p,h in contract['outputs'].items()})
        reference_hashes.update({p:digest(p) for p in approved})
        for name in ROLE_FILES.values():
            for row in read_json(out/name):
                for field in ('image','anatomy','lesion'):
                    if not row.get(field): continue
                    path=str(Path(row[field]).expanduser().resolve())
                    if (path!=row[field] or path not in reference_hashes
                            or row.get('source_hashes',{}).get(field)!=reference_hashes[path]):
                        raise ValueError()
    if read_sources:
        for bucket in ('inputs','source_files'):
            for name,expected in contract[bucket].items():
                if check_paths and str(Path(name).resolve()) not in approved: source_path(name)
                if digest(name)!=expected: raise ValueError()
    for name,expected in contract['source_snapshot'].items():
        if evaluation_source:
            member=None
            for prefix in ('execution_source','training_qc/execution_source'):
                candidate=(out/prefix/name).resolve()
                if (candidate.is_relative_to((out/prefix).resolve()) and
                        contract['outputs'].get(prefix+'/'+name)==expected):
                    member=candidate; break
            if member is None and qualified_source_root is not None:
                root=Path(qualified_source_root).expanduser().resolve()
                member=(root/name).resolve()
                if not member.is_relative_to(root): raise ValueError()
            if member is None: raise ValueError()
        elif frozen_execution:
            member=(out/'execution_source'/name).resolve()
            if not member.is_relative_to(out/'execution_source') or contract['outputs'].get('execution_source/'+name)!=expected:
                raise ValueError()
        else:
            member=(REPO/name).resolve()
            if not member.is_relative_to(REPO): raise ValueError()
        if digest(member)!=expected: raise ValueError()
    if (not frozen_execution or evaluation_source) and 'environment' in contract and contract['environment']!=decoder_environment(): raise ValueError()
    return contract


def verify_contract(path, *, check_paths=True, source_mode='live', qualified_source_root=None):
    """Verify all bytes and role links, using live code by default.

    Evaluation explicitly checks the immutable qualified execution source,
    while retaining current role-reference, input-byte and environment checks.
    It never updates a historical source commitment to accept feature code.
    """
    try:
        if source_mode not in ('live','evaluation') or (qualified_source_root is not None and source_mode!='evaluation'):
            raise ValueError()
        out=Path(path).expanduser().resolve()
        if out.is_file(): out=out.parent
        return _verify_tree(out,check_paths=check_paths,evaluation_source=source_mode=='evaluation',
                            qualified_source_root=qualified_source_root)
    except Exception:
        raise ValueError('Data contract integrity verification failed') from None


def finalize_qualification(out, initial):
    """Publish a metadata-only successor of a completed, immutable QC execution.

    No label conversion, label-array QC, image reads or source-byte hashing is
    repeated here. The final replay independently verifies the complete chain.
    """
    out=private_output(out); initial=private_output(initial)
    if out.parent!=initial or out.exists(): raise ValueError('Invalid metadata finalization destination')
    old=_verify_tree(initial,check_paths=True,read_sources=False,frozen_execution=True)
    for path,expected in old['inputs'].items():
        source_path(path)
        if file_sha256(path)!=expected: raise ValueError('Frozen metadata input differs')
    records,repairs,evidence=prior_qc()
    supplemental=initial/'supplemental_train_qc_v1'
    supplement,supplement_summary=load_training_supplement(supplemental,file_sha256(initial/'contract.json'))
    states={tuple(state['identity']):state['lesion_state'] for record in supplement for state in record['states']}
    repaired=read_json(REPAIRED_TRAIN)
    original={identity(r):r for r in read_json(DATA_ROOT/'derived/aur/samples_1mm_v1.json')}
    roles={key:read_json(initial/name) for key,name in ROLE_FILES.items()}
    original_train={identity(r):r for r in repaired}
    for key in ('train_supervised','validation'):
        updated=[]
        for previous in roles[key]:
            row=(original_train if key=='train_supervised' else original)[identity(previous)]
            status=previous['label_provenance']['lesion_state']
            if key=='train_supervised':
                status=states.get(identity(row),status)
            fixed=supervised_row(row,split=previous['split'],native_status=status)
            for field in ('geometry','source_hashes'): fixed[field]=previous[field]
            if previous.get('lesion'): fixed['lesion']=str(Path(previous['lesion']).resolve())
            else: fixed['lesion']=previous.get('lesion')
            updated.append(fixed)
        roles[key]=updated
    checks=verify_training_qc(repaired,old['source_files'],records,repairs,supplement=supplement)
    summary={**old['summary'],**{key:coverage(value) for key,value in roles.items()},'prior_qc_reuse':checks,
             'metadata_only_successor':True,'initial_execution_contract_sha256':file_sha256(initial/'contract.json'),
             'initial_metadata_issue':'absent_legacy_r_flag_was_disabled; successor_restores_qualified_3d_pseudo_R',
             'train_array_qc':'prior_frozen_anatomy_and_35_repairs_plus_new_819_unique_U_array_QC',
             'supplemental_training_qc':supplement_summary,
             'prior_census_commitment':'identified_existing_metadata_frozen_now; not_claimed_in_old_completion'}
    inputs={**old['inputs'],**{str(path):file_sha256(path) for path in evidence}}
    inputs[str(initial/'contract.json')]=file_sha256(initial/'contract.json')
    inputs[str(initial/'complete.json')]=file_sha256(initial/'complete.json')
    for key,name in ROLE_FILES.items(): write_private_json(out/name,roles[key])
    for path in supplemental.rglob('*'):
        if path.is_file():
            dest=out/'training_qc'/path.relative_to(supplemental)
            dest.parent.mkdir(parents=True,exist_ok=True,mode=0o700)
            shutil.copyfile(path,dest);dest.chmod(0o600)
    write_private_json(out/'anonymous_summary.json',summary)
    write_private_json(out/'lineage.json',{'operation':'metadata_only_successor','initial_contract_sha256':file_sha256(initial/'contract.json'),
        'label_array_qc_repeated':False,'label_conversion_repeated':False,'source_bytes_rehashed':False,
        'supplemental_training_qc_complete_sha256':file_sha256(supplemental/'complete.json'),
        'initial_source_snapshot':old['source_snapshot'],'prior_qc_evidence_sha256':{str(p):file_sha256(p) for p in evidence},
        'final_replay_required':True})
    outputs={str(path.relative_to(out)):file_sha256(path) for path in out.rglob('*') if path.is_file()}
    contract={**old,'inputs':inputs,'outputs':outputs,'source_snapshot':source_snapshot(),'environment':decoder_environment(),
        'summary':summary,'inherited_contract':{'path':str(initial/'contract.json'),'sha256':file_sha256(initial/'contract.json'),
                                             'complete_sha256':file_sha256(initial/'complete.json')}}
    write_private_json(out/'contract.json',contract)
    write_private_json(out/'complete.json',{'version':VERSION,'contract_sha256':file_sha256(out/'contract.json'),'outputs':outputs})
    return summary


def read_contract(path, verify=True, *, source_mode='live', qualified_source_root=None):
    out=Path(path).expanduser().resolve()
    if out.is_file(): out=out.parent
    contract=verify_contract(out,source_mode=source_mode,qualified_source_root=qualified_source_root) if verify else read_json(out/'contract.json')
    result={key:read_json(out/name) for key,name in ROLE_FILES.items()}
    return {**result,'version':contract['version'],'contract_sha256':file_sha256(out/'contract.json'),
            'summary':contract['summary']}


def qualify(out, *, repaired_train=REPAIRED_TRAIN, native=None, current=None, ssl=None, val_patients=None, workers=2):
    """Create one immutable version; defaults are the frozen production source files."""
    out=private_output(out)
    if out.exists(): raise ValueError('Output already exists; use a new private version')
    native=Path(native or DATA_ROOT/'derived/aur/samples.json')
    current=Path(current or DATA_ROOT/'derived/aur/samples_1mm_v1.json')
    ssl=Path(ssl or DATA_ROOT/'derived/aur/ssl_manifest_v1/samples_ssl.json')
    val_patients=Path(val_patients or DATA_ROOT/'derived/aur/ssl_manifest_v1/val_patients.json')
    repaired_train=Path(repaired_train)
    paths=[source_path(p) for p in (native,current,ssl,val_patients,repaired_train,REPAIR_ROOT/'complete.json')]
    if file_sha256(repaired_train)!=REPAIRED_TRAIN_SHA or file_sha256(REPAIR_ROOT/'complete.json')!=REPAIR_COMPLETE_SHA:
        raise ValueError('Frozen TRAIN repair integrity differs')
    inputs={str(p):file_sha256(p) for p in paths}
    rows=read_json(current); originals=read_json(native); ssl_rows=read_json(ssl); val=read_json(val_patients)
    train=read_json(repaired_train)
    validation_ids=validate_patient_roles(rows,ssl_rows,val,repaired_train=train)
    native_by_id={identity(r):r for r in originals}
    if len(native_by_id)!=len(originals) or {identity(r) for r in rows}!=set(native_by_id):
        raise ValueError('Native and current identity linkage differs')
    validation=[r for r in rows if r['split']=='train' and patient_group(r) in validation_ids]
    started=time.monotonic(); freeze_policy(out,validation)
    snapshot=source_snapshot(); source_paths=set(); geometry={}; label_cache={}; native_cache={}; repaired={}; ledger=[]
    qc=Counter(); qualified=[]

    def resolve(value):
        p=source_path(value); source_paths.add(str(p)); return p

    def header(value):
        p=resolve(value)
        if str(p) not in geometry:
            image=nib.load(str(p)); geometry[str(p)]=inspect_geometry(image)
        return nib.load(str(p))

    def label(value, image=None, decode=True):
        p=resolve(value)
        if decode and str(p) in label_cache: return label_cache[str(p)]
        values=inspect_label(p,image,decode=decode)
        if decode: label_cache[str(p)]=values
        return values

    for r in train:
        image=header(r['image']); label(r['anatomy'],image,decode=False)
        if r.get('lesion'): label(r['lesion'],image,decode=False)
        q=supervised_row(r,split='train')
        # The repaired manifest records three native-empty DWI rows explicitly.
        if r['source']=='isles' and r['sequence']=='DWI' and not r['u_supervised']:
            q=supervised_row(r,split='train',native_status='source_empty')
        qualified.append(q); qc['train_header_grid_rows']+=1

    val_qualified=[]
    for r in validation:
        image=header(r['image']); label(r['anatomy'],image)
        native_row=native_by_id[identity(r)]; status='unannotated'
        les=None
        if r.get('lesion'):
            les=label(r['lesion'],image)
            original_path=resolve(native_row['lesion'])
            if str(original_path) not in native_cache:
                native_img=nib.load(str(original_path)); inspect_geometry(native_img)
                native_values=inspect_label(original_path)
                native_cache[str(original_path)]=(native_img,native_values)
            native_img,native_values=native_cache[str(original_path)]
            status='source_positive' if np.isin(native_values,r.get('a_ignore_values',[])).any() else 'source_empty'
            if status=='source_positive' and r.get('u_supervised') and not np.isin(native_values,r['u_values']).any():
                status='sequence_empty'
            new=dict(r)
            if r['source']=='isles':
                if str(original_path) not in repaired:
                    # Freeze all validation members before reading their native label arrays.
                    native_anatomy=header(native_row['anatomy'])
                    grid=_target_grid(native_anatomy)
                    decoded=nib.Nifti1Image(native_values,native_img.affine)
                    restored=discrete_labels(np.asarray(resample_from_to(decoded,grid,order=0,mode='constant',cval=0).dataobj))
                    if restored.shape!=image.shape or not np.allclose(grid[1],image.affine,atol=1e-3,rtol=1e-5):
                        raise ValueError('Qualified validation target grid differs')
                    changed=not np.array_equal(restored,les)
                    item={'source':'isles','patient_group':patient_group(r),'source_positive':bool(np.any(native_values)),
                          'derived_positive':bool(np.any(les)),'qualified_positive':bool(np.any(restored)),
                          'changed':changed,'before_voxels':int(np.count_nonzero(les)),
                          'after_voxels':int(np.count_nonzero(restored))}
                    if changed:
                        repair_dir=out/'validation_repair_v1'; repair_dir.mkdir(mode=0o700,exist_ok=True)
                        destination=repair_dir/f'label_{len(repaired):04d}.nii.gz'
                        obj=nib.Nifti1Image(restored,grid[1]); obj.header.set_xyzt_units('mm'); nib.save(obj,str(destination)); destination.chmod(0o600)
                        inspect_label(destination,image)
                        repaired[str(original_path)]=str(destination)
                        qc['validation_isles_repaired_cases']+=1
                        qc['validation_isles_restored_voxels']+=int(np.count_nonzero(restored))-int(np.count_nonzero(les))
                        from anatobind.aur.targets import lesion_instances
                        vm=abs(float(np.linalg.det(image.affine[:3,:3])))
                        before_inst,_=lesion_instances(les,[1],vm); after_inst,_=lesion_instances(restored,[1],vm)
                        qc['validation_isles_restored_instances']+=int(after_inst.max())-int(before_inst.max())
                    else: repaired[str(original_path)]=r['lesion']
                    ledger.append(item)
                    qc['validation_isles_cases']+=1
                    qc['validation_isles_native_empty_cases']+=int(not np.any(native_values))
                    qc['validation_isles_native_positive_cases']+=int(np.any(native_values))
                    qc['validation_isles_old_positive_cases']+=int(np.any(les))
                new['lesion']=repaired[str(original_path)]
                if new['lesion']!=r['lesion']: status='native_positive_repaired' if np.any(native_values) else 'source_empty'
            r=new
        q=supervised_row(r,split='val',native_status=status); val_qualified.append(q)
        qc['validation_decoded_rows']+=1
        # Do not hold all validation label arrays in RAM.
        label_cache.clear()
    ssl_train=[]; ssl_val=[]
    for r in ssl_rows:
        header(r['image'])
        split='val' if patient_group(r) in validation_ids else 'train'
        q=image_only_row(r,split=split)
        (ssl_train if split=='train' else ssl_val).append(q)
    roles={'train_supervised':qualified,'train_ssl':ssl_train,'validation':val_qualified,'validation_ssl':ssl_val}
    # Verify source-qualified train/validation/test identity separation without opening historical test arrays.
    train_ids={patient_group(r) for r in qualified+ssl_train}; held_ids={patient_group(r) for r in val_qualified+ssl_val}
    test_ids={patient_group(r) for r in rows if r['split']=='test'}
    if train_ids&held_ids or train_ids&test_ids or held_ids&test_ids: raise ValueError('Patient role intersection')
    source_hashes=hash_files(source_paths,workers)
    records,old_repairs,evidence=prior_qc()
    exempt={str(Path(r['new_path']).resolve()) for r in old_repairs}
    exempt.update(str(Path(r['lesion']).resolve()) for r in train if r.get('lesion') and
                  r['source']=='isles' and r['sequence']=='DWI' and not r['u_supervised'])
    train_groups={}
    for r in train:
        if r.get('lesion'):
            path=str(source_path(r['lesion']))
            if path not in exempt: train_groups.setdefault(path,[]).append(r)
    training_qc,_,_,_=_inspect_training_groups(train_groups,source_hashes)
    prior_checks=verify_training_qc(train,source_hashes,records,old_repairs,supplement=training_qc)
    inputs.update({str(path):file_sha256(path) for path in evidence})
    local_hashes={str(p):file_sha256(p) for p in (out/'validation_repair_v1').glob('*.nii.gz')} if (out/'validation_repair_v1').exists() else {}
    for group in roles.values():
        for r in group:
            r['geometry']=geometry[str(Path(r['image']).resolve())]
            r['source_hashes']={k:source_hashes.get(str(Path(r[k]).resolve()),local_hashes.get(str(Path(r[k]).resolve())))
                                for k in ('image','anatomy','lesion') if r.get(k)}
            if not all(r['source_hashes'].values()): raise ValueError('Missing source integrity commitment')
    summary={'version':VERSION,**{key:coverage(group) for key,group in roles.items()},'qc':dict(qc),
             'train_validation_patient_intersection':0,'train_test_patient_intersection':0,
             'validation_test_patient_intersection':0,'source_files':len(source_hashes),
             'source_bytes':sum(Path(p).stat().st_size for p in source_hashes),
             'historical_test_dense_read':False,'biological_cross_source_deduplication':'not_established',
             'train_array_qc':'prior_frozen_anatomy_and_repairs_plus_new_unique_U_array_QC',
             'seconds':round(time.monotonic()-started,3),'prior_qc_reuse':prior_checks}
    for key,name in ROLE_FILES.items(): write_private_json(out/name,roles[key])
    write_private_json(out/'validation_qc_private.json',ledger)
    write_private_json(out/'training_qc_private.json',training_qc)
    write_private_json(out/'source_geometry_private.json',geometry)
    write_private_json(out/'anonymous_summary.json',summary)
    outputs={str(p.relative_to(out)):file_sha256(p) for p in sorted(out.rglob('*')) if p.is_file()}
    # Inputs must still match after qualification/hash work, before final completion.
    if any(file_sha256(p)!=digest for p,digest in inputs.items()): raise ValueError('Input integrity changed during qualification')
    if source_snapshot()!=snapshot: raise ValueError('Qualification source changed during execution')
    contract={'version':VERSION,'inputs':inputs,'source_files':source_hashes,'source_snapshot':snapshot,
              'outputs':outputs,'summary':summary,'policy_sha256':file_sha256(out/'policy.json'),
              'role_files':ROLE_FILES,'repair_evidence':'frozen_prior_train_repair_hashes',
              'source_roots':[str(DATA_ROOT),str(REPAIR_ROOT)],'environment':decoder_environment()}
    write_private_json(out/'contract.json',contract)
    write_private_json(out/'complete.json',{'version':VERSION,'contract_sha256':file_sha256(out/'contract.json'),'outputs':outputs})
    return summary

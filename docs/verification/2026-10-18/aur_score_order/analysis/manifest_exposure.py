"""Read-only manifest eligibility audit; no images, outcomes or patient IDs are emitted."""
import argparse
from collections import Counter,defaultdict
import hashlib
import json
from pathlib import Path


def audit():
    root=Path('/data2/congcong/data/FM_data/derived/aur')
    expected={'samples_1mm_v1.json':'fb96acd741aa6c0ee689eb772a4332304501c8d63ebd7ccddcd657f41eb86d5d',
        'ssl_manifest_v1/val_patients.json':'475f91d97d8b03f312df560f98f39ab81c068c39d0a24c632d8efa5fd52d9d34',
        'ssl_manifest_v1/samples_ssl.json':'91f84fd479a94be6872e83cdc2a5a2efad14dc214a88e7003f3e948f6d46bb0e'}
    for name,h in expected.items():
        if hashlib.sha256((root/name).read_bytes()).hexdigest()!=h:raise ValueError('Manifest commitment changed')
    rows,val,ssl=[json.loads((root/name).read_text()) for name in expected]
    identity=lambda r:(r['source'],str(r['patient']))
    held={(s,str(p)) for s,ps in val.items() for p in ps}
    split=lambda r:'test' if r['split']=='test' else 'val' if identity(r) in held else 'train'
    count=lambda rs:dict(rows=len(rs),source_qualified_patients=len({identity(r) for r in rs}))
    stages={t:count([r for r in rows if split(r)==t]) for t in ('train','val','test')}
    u={t:count([r for r in rows if split(r)==t and r['u_supervised']]) for t in stages}
    per_source={s+'/'+q:{t:count([r for r in rows if (r['source'],r['sequence'])==(s,q)
        and r['u_supervised'] and split(r)==t]) for t in stages}
        for s,q in sorted({(r['source'],r['sequence']) for r in rows if r['u_supervised']})}
    sets={t:{identity(r) for r in rows if split(r)==t} for t in stages}
    ssl_train={identity(r) for r in ssl if r['ssl_split']=='train'}
    by_id=defaultdict(set);by_ssl_id=defaultdict(set);by_sibbms=defaultdict(set)
    for r in rows:
        by_id[str(r['patient'])].add(r['source'])
        if r['source']=='sibbms':by_sibbms[str(r['patient'])].add(split(r))
    for r in ssl:by_ssl_id[str(r['patient'])].add(r['source'])
    subset=Counter()
    path=root.parents[1]/'SibBMS_ms/sibbms/Output/Annotation'
    for d in path.iterdir():
        if d.is_dir():
            membership=by_sibbms.get('MS_'+d.name,set())
            subset[next(iter(membership)) if len(membership)==1 else 'unknown']+=1
    return dict(scope='Manifest eligibility, not exact sampled-crop exposure or biological cross-source deduplication',
        manifest_sha256=expected,stage_II_III=stages,U_supervised=u,U_per_source_sequence=per_source,
        stage_I={t:count([r for r in ssl if r['ssl_split']==t]) for t in ('train','val')},
        principal_split_intersections={a+'/'+b:len(sets[a]&sets[b]) for a,b in [('train','val'),('train','test'),('val','test')]},
        principal_patients_eligible_for_stage_I_gradient={t:len(ids&ssl_train) for t,ids in sets.items()},
        principal_cross_source_patient_string_clashes=sum(len(s)>1 for s in by_id.values()),
        stage_I_cross_source_patient_string_clashes=sum(len(s)>1 for s in by_ssl_id.values()),
        planned_SibBMS_annotation_patient_split=dict(subset),
        limitations=['TRAIN folds constrain calibrator only; base model already trained onTRAIN.',
            'Existing val/six and historical test are previously inspected; no fresh independent evaluation cohort established.',
            'Source-qualified manifest disjointness does not prove biological identity resolution across sources.'])


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--out',type=Path,required=True);args=parser.parse_args()
    result=audit();args.out.parent.mkdir(parents=True,exist_ok=True,mode=0o700)
    with args.out.open('x') as stream:json.dump(result,stream,indent=2)
    print(json.dumps(result,indent=2))

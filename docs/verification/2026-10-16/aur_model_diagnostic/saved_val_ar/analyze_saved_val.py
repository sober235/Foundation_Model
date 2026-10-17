#!/usr/bin/env python3
"""Read saved val metrics, write only anonymous aggregates; no model imports or inference.

Run with CUDA_VISIBLE_DEVICES='' and *_NUM_THREADS=1, under nice -n 19.
Only the input JSON files are read. No raw identities or instance records are copied.
"""
import argparse
import ast
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median

BASE = Path('/data0/congcong/code/Project_Doing/foundation_model-conformance/docs/verification/2026-10-09/anatobind_brain_ssl_first/eval/c2_val_20261014_1344')
REPO = Path('/home/congcongliu/src/Foundation_Model-conformance-work')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def clean(x):
    if isinstance(x, float) and not math.isfinite(x):
        return None
    if isinstance(x, dict):
        return {str(k): clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [clean(v) for v in x]
    return x


def write_json(path, obj):
    path.write_text(json.dumps(clean(obj), indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def write_csv(path, rows):
    cols = list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(clean(rows))


def finite(vals):
    return [float(x) for x in vals if isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)]


def stats(vals):
    vals = list(vals)
    v = sorted(finite(vals))
    def quantile(q):
        if not v:
            return None
        z = q * (len(v) - 1)
        lo, hi = math.floor(z), math.ceil(z)
        return v[lo] + (v[hi] - v[lo]) * (z - lo)
    return {'n_total': len(vals), 'n_valid': len(v), 'n_nonfinite_or_null': len(vals)-len(v), 'n_zero':sum(x==0 for x in v),
            'mean': mean(v) if v else None, 'min': min(v) if v else None,
            'p25': quantile(.25), 'median': median(v) if v else None,
            'p75': quantile(.75), 'max': max(v) if v else None}


def schema(values):
    values = list(values)
    out = {'n_present': len(values), 'type_counts': dict(Counter(type(v).__name__ for v in values))}
    nums = [v for v in values if isinstance(v, float)]
    if nums:
        out['n_nonfinite'] = sum(not math.isfinite(v) for v in nums)
    ds = [v for v in values if isinstance(v, dict)]
    if ds:
        keys = sorted(set().union(*(v.keys() for v in ds)))
        out['fields'] = {k: schema(v[k] for v in ds if k in v) for k in keys}
    ls = [v for v in values if isinstance(v, list)]
    if ls:
        out['length_min'] = min(map(len, ls))
        out['length_max'] = max(map(len, ls))
        elements = [v for a in ls for v in a]
        if elements:
            out['items'] = schema(elements)
    return out


def literal_assignment(path, key):
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == key for t in node.targets):
            return ast.literal_eval(node.value)
    raise ValueError('mapping constant not found')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=Path, default=BASE)
    parser.add_argument('--repo', type=Path, default=REPO)
    parser.add_argument('--output', type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    cases = json.loads((args.input/'cases.json').read_text())
    aggregate = json.loads((args.input/'aggregate.json').read_text())
    student = literal_assignment(args.repo/'anatobind/anatomy/labels.py', 'STUDENT')
    host_names = [name for cid, name, labels, rep in student if 1 <= cid <= 13] + ['no_host']
    entities = literal_assignment(args.repo/'anatobind/eval/lookup.py', 'BRAIN_ALL')
    names_path = args.repo/'docs/verification/2026-09-16-brain-probe/probe_common.py'
    entity_names = literal_assignment(names_path, 'LABEL_NAMES')
    entity_to_host = {int(l): name for cid, name, ls, rep in student for l in ls}
    def side(h):
        name = host_names[h]
        return 'none' if h == 13 else ('left' if name.endswith('_left') else 'right' if name.endswith('_right') else 'midline')
    def tissue(h):
        name = host_names[h]
        return None if h == 13 else name.removesuffix('_left').removesuffix('_right')

    write_json(args.output/'schema.json', {'cases': schema(cases), 'aggregate': schema([aggregate]),
               'host_index_mapping': dict(enumerate(host_names)), 'entity_labels_order': entities})
    a_summary, a_hosts, a_entities = [], [], []
    agroups = {('all_supervised', 'ALL', 'ALL'): [c for c in cases if c['a_supervised']],
               ('all_unsupervised', 'ALL', 'ALL'): [c for c in cases if not c['a_supervised']]}
    for supervised in [True, False]:
        scope = 'supervised' if supervised else 'unsupervised'
        subset = [c for c in cases if c['a_supervised'] == supervised]
        for source in sorted({c['source'] for c in subset}):
            agroups[(scope, source, 'ALL')] = [c for c in subset if c['source']==source]
            for seq in sorted({c['sequence'] for c in subset if c['source']==source}):
                agroups[(scope, source, seq)] = [c for c in subset if c['source']==source and c['sequence']==seq]
    for (scope, source, seq), group in agroups.items():
        ident = {'scope': scope, 'source': source, 'sequence': seq, 'n_volumes': len(group),
                 'n_patients': len({(c['source'], c['patient']) for c in group})}
        sm = {**ident, 'host_macro': stats(c['a']['host_macro'] for c in group)['mean'],
              'entity_macro': stats(c['a']['entity_macro'] for c in group)['mean']}
        a_summary.append(sm)
        for h in host_names[:13]:
            a_hosts.append({**ident, 'host': h, **stats(c['a']['hosts'].get(h) for c in group)})
        for l in entities:
            a_entities.append({**ident, 'entity_label': l, 'entity_name_repo':entity_names[l], 'host_or_landmark': entity_to_host.get(l, 'non_host_CSF'),
                               **stats(c['a']['entities'].get(str(l)) for c in group)})

    rrows = []
    nonignored = Counter()
    no_r_counted = Counter()
    r_missing_reason = Counter()
    for ci, c in enumerate(cases):
        gs = [g for g in (c.get('u') or {}).get('gt', []) if not g.get('ignore', False)]
        nonignored[c['source']] += len(gs)
        if not c.get('r') and gs:
            no_r_counted[c['source']] += len(gs)
            r_missing_reason[f"r_supervised={c['r_supervised']}"] += len(gs)
        by_id = {g['instance']: g for g in gs}
        for row in (c.get('r') or {}).get('instances', []):
            assert row['instance'] in by_id, 'R GT id mismatch'
            gt = by_id[row['instance']]
            assert gt['volume_mm3'] == row['volume_mm3'], 'R GT volume mismatch'
            assert gt['box'] == row['box'], 'R GT box mismatch'
            assert gt['stratum'] == row['stratum'], 'R GT stratum mismatch'
            assert len(row['r_probs']) == 14 and 0 <= row['truth'] <= 13, 'R vocabulary mismatch'
            # Private grouping values remain in memory, never written as records.
            rr = dict(row, source=c['source'], sequence=c['sequence'], _case_index=ci,
                      _private_patient=(c['source'], c['patient']))
            rr['side'] = side(row['truth'])
            rr['truth_host'] = host_names[row['truth']]
            rr['outcome'] = ('both_right' if row['b0']==row['truth'] else 'rescue') if row['r']==row['truth'] else ('harm' if row['b0']==row['truth'] else 'both_wrong')
            rrows.append(rr)

    def rblock(rows):
        n = len(rows)
        if not n:
            return {'n': 0}
        counts = Counter(r['outcome'] for r in rows)
        nr = counts['both_right'] + counts['rescue']
        nb = counts['both_right'] + counts['harm']
        return {'n': n, 'n_volumes': len({r['_case_index'] for r in rows}),
                'n_patients': len({r['_private_patient'] for r in rows}),
                'aba_r': nr/n, 'aba_b0': nb/n, 'delta_r_minus_b0': (nr-nb)/n,
                **{k:counts[k] for k in ['both_right','both_wrong','rescue','harm']},
                'rescue_per_b0_wrong': counts['rescue']/(n-nb) if nb<n else None,
                'harm_per_b0_right': counts['harm']/nb if nb else None,
                'side_r':mean(side(r['r'])==side(r['truth']) for r in rows),
                'side_b0':mean(side(r['b0'])==side(r['truth']) for r in rows),
                'tissue_r':mean(tissue(r['r'])==tissue(r['truth']) for r in rows),
                'tissue_b0':mean(tissue(r['b0'])==tissue(r['truth']) for r in rows),
                'n_zero_overlap':sum(r['zero_overlap'] for r in rows),
                'mean_query_iou':stats(r['query_iou'] for r in rows)['mean'],
                'mean_local_dice':stats(r['local_dice'] for r in rows)['mean'],
                'mean_in_crop_share':stats(r['in_crop_share'] for r in rows)['mean'],
                'median_volume_mm3':stats(r['volume_mm3'] for r in rows)['median']}

    def query_bin(r):
        x = r['query_iou']
        if not math.isfinite(x): return 'missing'
        if x == 0: return '=0'
        for hi, name in [(.01,'(0,0.01)'),(.1,'[0.01,0.1)'),(.3,'[0.1,0.3)'),(.5,'[0.3,0.5)')]:
            if x < hi: return name
        return '[0.5,1]'
    def local_bin(r):
        x = r['local_dice']
        if not math.isfinite(x): return 'missing'
        for hi, name in [(.6,'<0.6'),(.7,'[0.6,0.7)'),(.8,'[0.7,0.8)'),(.9,'[0.8,0.9)')]:
            if x < hi: return name
        return '[0.9,1]'
    def crop_bin(r):
        x = r['in_crop_share']
        if not math.isfinite(x): return 'missing'
        if x < .5: return '<0.5'
        if x < .9: return '[0.5,0.9)'
        if x < 1: return '[0.9,1)'
        return '=1'
    def joint(r):
        return f"query_iou{' >=0.5' if r['query_iou']>=.5 else ' <0.5'};local_dice{' >=0.8' if r['local_dice']>=.8 else ' <0.8'}"
    dims = {'source':lambda r:r['source'], 'source_sequence':lambda r:f"{r['source']}/{r['sequence']}",
            'size_equivalent_diameter_mm':lambda r:r['stratum'], 'truth_side':lambda r:r['side'],
            'truth_host':lambda r:r['truth_host'], 'query_iou':query_bin, 'local_dice':local_bin,
            'in_crop_share':crop_bin, 'joint_quality':joint,
            'source_by_query_iou':lambda r:f"{r['source']};{query_bin(r)}",
            'source_by_local_dice':lambda r:f"{r['source']};{local_bin(r)}",
            'source_by_size':lambda r:f"{r['source']};{r['stratum']}",
            'source_by_side':lambda r:f"{r['source']};{r['side']}",
            'source_by_joint_quality':lambda r:f"{r['source']};{joint(r)}"}
    r_strata = [{'dimension':'ALL','stratum':'ALL',**rblock(rrows)}]
    for dim, fn in dims.items():
        groups = defaultdict(list)
        for r in rrows: groups[fn(r)].append(r)
        for key, rs in sorted(groups.items()):
            r_strata.append({'dimension':dim,'stratum':key,**rblock(rs)})
    confusions = []
    for scope in ['ALL'] + sorted({r['source'] for r in rrows}):
        selected = [r for r in rrows if scope == 'ALL' or r['source'] == scope]
        for method in ['r', 'b0']:
            counts = Counter((r['truth'], r[method]) for r in selected)
            for (truth, pred), n in sorted(counts.items()):
                confusions.append({'source':scope,'method':method,'truth':host_names[truth],
                                   'predicted':host_names[pred],'n':n})
    outcomes = []
    for outcome in ['both_right','rescue','harm','both_wrong']:
        rows = [r for r in rrows if r['outcome']==outcome]
        for field in ['query_iou','local_dice','in_crop_share','volume_mm3']:
            outcomes.append({'outcome':outcome, 'field':field, **stats(r[field] for r in rows)})

    ac = next(x for x in a_summary if x['scope']=='all_supervised')
    rc = rblock(rrows)
    validations = {
      'n_cases':len(cases), 'n_a_supervised':sum(c['a_supervised'] for c in cases),
      'n_u_nonnull':sum(c['u'] is not None for c in cases), 'n_r_nonnull':sum(c['r'] is not None for c in cases),
      'n_controlled':len(rrows), 'nonignored_gt_by_source':dict(nonignored),
      'counted_gt_without_r_by_source':dict(no_r_counted), 'counted_gt_without_r_reason':dict(r_missing_reason),
      'a_host_macro_difference_from_saved':ac['host_macro']-aggregate['a']['host_macro'],
      'a_entity_macro_difference_from_saved':ac['entity_macro']-aggregate['a']['entity_macro'],
      'r_aba_difference_from_saved':rc['aba_r']-aggregate['r']['controlled']['aba_r'],
      'b0_aba_difference_from_saved':rc['aba_b0']-aggregate['r']['controlled']['aba_b0'],
      'rescue_harm_counts_match':all(rc[k]==aggregate['r']['controlled']['rescue_harm'][k] for k in ['both_right','both_wrong','rescue','harm']),
      'all_controlled_rows_match_gt_volume_box_stratum':True,
      'r_numeric_field_checks':{f:stats(r[f] for r in rrows) for f in ['query_iou','local_dice','in_crop_share','volume_mm3']},
      'entity_zero_counts_supervised':{str(l):{'n_finite':len(finite(c['a']['entities'][str(l)] for c in cases if c['a_supervised'])),
          'n_zero':sum(c['a']['entities'][str(l)] == 0 for c in cases if c['a_supervised'])} for l in entities},
    }
    assert max(abs(validations[k]) for k in ['a_host_macro_difference_from_saved','a_entity_macro_difference_from_saved','r_aba_difference_from_saved','b0_aba_difference_from_saved']) < 1e-12
    assert validations['rescue_harm_counts_match']
    provenance = {'claim_type':'computed_from_saved_val_metrics', 'evidence':'NOT_EVIDENCE',
      'input': {f:sha(args.input/f) for f in ['cases.json','aggregate.json']},
      'script_sha256':sha(Path(__file__)),
      'mapping_sha256':{f:sha(args.repo/f) for f in ['anatobind/anatomy/labels.py','anatobind/eval/lookup.py','docs/verification/2026-09-16-brain-probe/probe_common.py']},
      'analysis':'Read-only CPU aggregate analysis; no inference, model changes, raw-array reads or identity exports.',
      'bin_notes':{'query_iou':'saved GT-centred crop dense mask >0.5 IoU, not whole-volume detection IoU',
                   'local_dice':'saved full-volume hard-A 13-host macro within bind window, not coarse R input',
                   'size':'existing equivalent spherical diameter bins in mm, not volume bins',
                   'side':'derived from pseudo-truth host index, not predicted side',
                   'bins':'fixed descriptive bins; no thresholds are selected or changed'},
      'missing_fields':['selected_query_id','bind_window','R coarse A masks','R coarse U masks','A presence','U presence',
                        'R candidate-host geometry','local crop host pseudo-truth','voxelwise lesion-inside/outside A errors',
                        'raw anatomical/lesion masks','SynthSeg reference-quality judgment','human host truth'],
      'caveats':['All A/R references are SynthSeg pseudo-labels and remain NOT_EVIDENCE.',
                 'R is instance-weighted and some patients contribute multiple sequences/lesions; strata are descriptive, not independent causal tests.',
                 'Controlled R omits whole-volume score filtering but uses a GT-centred selected predicted query; B0 uses complete GT instance plus whole-volume hard A.',
                 'Saved local A Dice and query IoU do not directly measure coarse/presence-gated R input masks.']}
    write_csv(args.output/'a_source_sequence_summary.csv',a_summary)
    write_csv(args.output/'a_source_sequence_hosts.csv',a_hosts)
    write_csv(args.output/'a_source_sequence_entities.csv',a_entities)
    write_csv(args.output/'r_controlled_strata.csv',r_strata)
    write_csv(args.output/'r_context_by_outcome.csv',outcomes)
    write_csv(args.output/'r_host_confusions.csv',confusions)
    write_json(args.output/'analysis.json',{'provenance':provenance,'validation':validations,
               'a_summary':a_summary, 'a_hosts':a_hosts, 'a_entities':a_entities,
               'r_controlled_strata':r_strata,'r_context_by_outcome':outcomes,'r_host_confusions':confusions})
    def fmt(x):
        return f'{x:.4f}' if isinstance(x, float) else str(x)
    def table(headers, rows):
        return '\n'.join(['| '+' | '.join(headers)+' |','|'+'|'.join('---' for _ in headers)+'|']+
                         ['| '+' | '.join(fmt(v) for v in row)+' |' for row in rows])
    a_seq = [r for r in a_summary if r['scope']=='supervised' and r['sequence']!='ALL']
    zero_labels = [5,44,18,26,58]
    zero_seq = []
    for seqrow in a_seq:
        fields = [seqrow['source'],seqrow['sequence'],seqrow['n_volumes']]
        for label in zero_labels:
            ent = next(r for r in a_entities if r['scope']=='supervised' and r['source']==seqrow['source'] and r['sequence']==seqrow['sequence'] and r['entity_label']==label)
            fields.append(f"{ent['n_zero']}/{ent['n_valid']}")
        zero_seq.append(fields)
    def rtable(dimension):
        return table(['group','n','ABA R','ABA B0','rescue','harm'],
            [[r['stratum'],r['n'],r.get('aba_r'),r.get('aba_b0'),r.get('rescue'),r.get('harm')]
             for r in r_strata if r['dimension']==dimension])
    aq = [r for r in a_entities if r['scope']=='all_supervised']
    lines = [
      '# Anonymous saved-val A/R analysis — 2026-10-16',
      '', 'All A/R references are SynthSeg pseudo-labels: **NOT_EVIDENCE**. These are computed descriptive aggregates from the frozen saved validation metrics; no inference or model change was run.',
      '', '## Integrity and scope',
      '', f"405 saved volumes; 394 A-supervised volumes from 96 source-qualified patients; 11 A-unsupervised volumes reported separately. Controlled R has 716 instances from 134 volumes / 81 patients. The 718 nonignored U GT instances include two without R because `r_supervised=False` (one BMSR, one ISLES).",
      '', 'Recomputed A host/entity macro and R/B0 ABA match `aggregate.json` within 2.3e-16; both-right/both-wrong/rescue/harm counts match exactly (285/94/122/215). All 716 R GT volumes, boxes and size strata match the corresponding U GT rows. No numeric R context field is missing or nonfinite.',
      '', 'Input hashes:', '', table(['file','SHA256'],[[k,v] for k,v in provenance['input'].items()]),
      '', '## A source × sequence',
      '', 'These are mean row macro Dice scores over A-supervised volumes, using the original nan-mean convention. Repeated patient volumes are not independent replications. Full 13-host and 32-entity breakdowns, valid counts and zero counts are in the two corresponding CSVs.',
      '', table(['source','sequence','volumes','host macro','entity macro'],[[r['source'],r['sequence'],r['n_volumes'],r['host_macro'],r['entity_macro']] for r in a_seq]),
      '', 'The aggregate host macro 0.8035 masks ISLES/DWI 0.7237, SibBMS/T2 0.7402, BMSR/FLAIR 0.7841, and PDGM/FLAIR 0.7925. The weakest overall hosts are left other-deep-grey 0.7088 and left/right cortex 0.7179/0.7196. On ISLES/DWI, left other-deep-grey is 0.5965, left basal ganglia 0.6333, and left cortex 0.6367; SibBMS/T2 left/right cortex are 0.6180/0.6137.',
      '', '## A entity collapse candidates',
      '', 'Five entity labels have exactly zero Dice on every finite A-supervised volume. Label names below are parsed from the repository `docs/verification/2026-09-16-brain-probe/probe_common.py::LABEL_NAMES`, not recalled from an external ontology. Current `anatobind/anatomy/labels.py::STUDENT` groups 5/44 in ventricles; `anatobind/eval/geometry.py::LANDMARKS` describes inferior-lateral ventricles as landmarks.',
      '', table(['label','repository name','finite','zero','nonfinite'],[[r['entity_label'],r['entity_name_repo'],r['n_valid'],r['n_zero'],r['n_nonfinite_or_null']] for r in aq if r['entity_label'] in zero_labels]),
      '', 'Source × sequence counts below are zero / finite; denominator volumes appears separately. Every finite value is zero. Nonfinite values use the evaluator definition: both reference and prediction are empty. A finite zero proves no intersection, but cannot distinguish absent prediction, wrong identity, spatial mismatch or presence filtering without voxel/presence data.',
      '', table(['source','sequence','volumes','5 zero/finite','44 zero/finite','18 zero/finite','26 zero/finite','58 zero/finite'],zero_seq),
      '', 'All 32 nonbackground entity means:',
      '', table(['label','repository name','mean Dice','finite','zero'],[[r['entity_label'],r['entity_name_repo'],r['mean'],r['n_valid'],r['n_zero']] for r in aq]),
      '', '## R controlled context strata',
      '', 'Overall ABA(R)=0.5684 versus B0=0.6983; R rescues 122 B0 errors and harms 215 B0-correct instances. R harms 43.0% of B0-correct instances. Counts are instance-weighted and source, lesion size and query quality covary. Query IoU is for the selected dense >0.5 query within the GT-centred crop; it is not whole-volume detection IoU and not a direct measurement of the coarse R mask.',
      '', 'By source × sequence:', '', rtable('source_sequence'),
      '', 'By selected query IoU:', '', rtable('query_iou'),
      '', '604/716 instances have query IoU <0.1 and contain 197/215 harms (91.6%) versus 99/122 rescues. The >=0.1 group has n=112, 23 rescues and 18 harms. Query-IoU >=0.5 has n=66 and R/B0=0.6667/0.6515, but 59/66 are PDGM, four BMSR and three ISLES; this is not evidence that improving query IoU alone will cause the gain. Median query IoU is 0.00318 overall, 0.00240 for harms, 0.00329 for rescues.',
      '', 'By saved local A Dice:', '', rtable('local_dice'),
      '', '405 instances have local A Dice >=0.8, yet R/B0=0.5407/0.7012 with 68 rescues and 133 harms. Harms have mean local Dice 0.7911, compared with 0.7872 for rescues. This saved whole-volume hard-A statistic does not support a simple concentration of harm in low local Dice. It does not evaluate R\'s coarse or presence-gated A context.',
      '', 'Joint quality:', '', rtable('joint_quality'),
      '', 'By existing equivalent spherical diameter in mm:', '', rtable('size_equivalent_diameter_mm'),
      '', 'Instances <5 mm account for 120 harms versus 52 rescues, with R/B0=0.5326/0.7252. Instances >=10 mm are closer (32 harms/28 rescues; R/B0=0.6319/0.6564). All 716 in-crop shares are **exactly 1.0**: no saved GT-instance truncation is observed, and this field cannot stratify the cohort. Host-target/context boundary discrepancies remain unmeasured.',
      '', 'By pseudo-truth side:', '', rtable('truth_side'),
      '', 'No no-host truths occur. Side agreement is 0.9106 for R versus 0.9818 for B0, so tissue choice is not the only observed error axis. Small midline strata require caution.',
      '', 'By pseudo-truth host:', '', rtable('truth_host'),
      '', 'R predicts cortex for 511/716 instances (truth cortex: 404/716); 158/215 harms predict cortex, including 103 harms on white-matter truths. White-matter truths total 180: R/B0 correct counts are 32/142, rescues/harms 5/115. R has zero correct answers on the 33 thalamus/basal-ganglia/other-deep-grey truths. These descriptive biases motivate inspection of coarse geometry, candidate inputs and 14-class probabilities, without identifying a causal mechanism from the saved scores.',
      '', '## Supported, not observed, unresolved',
      '', table(['status','observation'],[
        ['supported','Five A entities have zero finite Dice across sources/sequences; raw count/presence diagnosis is needed.'],
        ['supported','R harms are common with poor selected-query IoU and on small/white-matter truths; source and size confounding remain.'],
        ['supported','High saved local full-volume A Dice does not remove R harm.'],
        ['not observed','GT instance truncation by in_crop_share; all 716 are one.'],
        ['unresolved','R coarse/presence-gated A/U mask quality, geometry/candidate inputs, crop/full-volume host-target discrepancies and relation-head contribution.'],
        ['unresolved','A lesion-inside/outside error, SynthSeg reference-quality issues and true human relation accuracy.']]),
      '', 'The replay should inspect selected query ID, coarse inputs, presence, candidate geometry, probabilities and local/full host targets. This analysis does not choose or claim a model repair. All A/R pseudo-reference findings remain NOT_EVIDENCE.',
      '', '## Reproduction and artifacts',
      '', '```sh', "env CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 nice -n 19 python /tmp/aur-ar-saved-val-analysis-20261016/analyze_saved_val.py", '```',
      '', '`schema.json` / `SCHEMA.md`: exact machine/readable schemas; `analysis.json`: all anonymous results and validation/provenance; `a_source_sequence_summary.csv`, `a_source_sequence_hosts.csv`, `a_source_sequence_entities.csv`; `r_controlled_strata.csv`, `r_context_by_outcome.csv`, `r_host_confusions.csv`; `run.log`: compact aggregate stdout. No case/patient identities or individual instance rows are exported.',
    ]
    (args.output/'REPORT.md').write_text('\n'.join(lines)+'\n')
    # Compact stdout is aggregate-only.
    print(json.dumps({'validation':validations,'a_summary':a_summary,
                     'r_strata':[r for r in r_strata if r['dimension'] in ['ALL','source','query_iou','local_dice','in_crop_share','joint_quality','truth_side','size_equivalent_diameter_mm']]},indent=2))


if __name__ == '__main__':
    main()

"""Cross-module contracts for the frozen optional query-CC decoder."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from anatobind.aur import infer as I
from anatobind.aur.model import AnatoBindBrain

TINY = dict(embed=32, depths=(1, 1, 1, 1), heads=(1, 1, 1, 1), window=(2, 4, 4), patch=(2, 4, 4), d_model=16,
            n_events=4, mask_dim=4, pixel_dim=8, dec_layers=1, dec_heads=2, rel_layers=1, rel_heads=2, use_checkpoint=False)


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).resolve().parents[1] / 'scripts' / (name + '.py'))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_query_profile_keeps_anatomy_and_original_accumulators_identical():
    torch.manual_seed(8)
    model = AnatoBindBrain(**TINY).eval()
    image = np.random.default_rng(4).normal(size=(8, 18, 18)).astype(np.float32)
    args = model, image, np.eye(4), (8, 16, 16), torch.device('cpu')
    before = I.predict_volume(*args, batch_size=2)
    after = I.predict_volume(*args, batch_size=2, decoder_profile='query_cc_v1', voxel_mm3=1., decoder_workers=2)
    for key in before:
        np.testing.assert_array_equal(after[key], before[key])
    inst, rows = I.decoded_instances(after, 1.)
    assert inst.dtype == np.int32 and inst.shape == image.shape
    assert len(rows) == int(inst.max())
    for r in rows:
        assert r['detector_origin']['window'] and 'query' in r['detector_origin']
    legacy = I.predict_volume(*args, decoder_profile='legacy_max', batch_size=2)
    for key in before:
        np.testing.assert_array_equal(legacy[key], before[key])
    gt = np.zeros(image.shape, np.int32)
    gt[2:5, 6:10, 6:10] = 1
    r0 = I.bind_instances(model, image, np.eye(4), gt, (8,16,16), torch.device('cpu'))
    I.predict_volume(*args, batch_size=2, decoder_profile='query_cc_v1', voxel_mm3=1.)
    r1 = I.bind_instances(model, image, np.eye(4), gt, (8,16,16), torch.device('cpu'))
    assert r0[0]['query'] == r1[0]['query'] and r0[0]['window'] == r1[0]['window']
    np.testing.assert_array_equal(r0[0]['host_probs'], r1[0]['host_probs'])


def test_profile_validation_happens_before_inference():
    with pytest.raises(ValueError, match='decoder'):
        I.predict_volume(None, np.zeros((2, 2, 2)), np.eye(4), (2, 2, 2), torch.device('cpu'), decoder_profile='typo')
    with pytest.raises(ValueError, match='voxel'):
        I.predict_volume(None, np.zeros((2, 2, 2)), np.eye(4), (2, 2, 2), torch.device('cpu'), decoder_profile='query_cc_v1')
    with pytest.raises(ValueError, match='threshold'):
        I.decoded_instances({'decoder_profile': 'query_cc_v1'}, 1., threshold=.4)


def test_records_keep_detection_and_actual_binding_origins_separate():
    source = dict(window=[[0, 8], [0, 16], [0, 16]], query=3, component=1, rank=1)
    row = dict(instance=1, box=[1, 2, 3, 4, 5, 6], n_voxels=27, volume_mm3=27., score=.8, detector_origin=source)
    bound = dict(instance=1, host=0, host_probs=np.array([1.] + [0.]*13), query=1, query_iou=.4,
                 window=[[2, 10], [1, 17], [3, 19]], in_crop_share=1.)
    r = I.lesion_records([row], [bound])[0]
    assert r['detector_origin'] == source
    assert r['binding_origin']['query'] == 1 and r['binding_origin']['window'] == bound['window']


def test_entry_parsers_expose_explicit_default_compatible_profiles():
    ev = load_script('aur_eval')
    inf = load_script('infer_anatobind_brain')
    args = ['--checkpoint','c.pt','--samples','s.json','--out','o','--cpu','--decoder-profile','query_cc_v1']
    assert ev.parser().parse_args(args).decoder_profile == 'query_cc_v1'
    args = ['--checkpoint','c.pt','--nifti','i.nii.gz','--out','o','--cpu']
    assert inf.parser().parse_args(args).decoder_profile == 'legacy_max'


def test_shard_merge_rejects_mixed_profiles_and_changed_locked_rules():
    m = load_script('aur_eval_merge')
    from anatobind.aur.query_decoder import decoder_rules
    base = dict(checkpoint_sha256='same', split='val', samples_sha256='data', crop=[128,160,160],
                sources=None, limit=None, instance_threshold=.3, stage='III', e2e=True)
    legacy = dict(base)
    query = {**base,'decoder_profile':'query_cc_v1','decoder_rules':decoder_rules('query_cc_v1')}
    assert m.validate_decoder_metadata([legacy,legacy])[0] == 'legacy_max'
    with pytest.raises(ValueError, match='decoder'):
        m.validate_decoder_metadata([legacy,query])
    with pytest.raises(ValueError, match='decoder'):
        m.validate_decoder_metadata([query,{**query,'decoder_rules':{'wrong':True}}])
    with pytest.raises(ValueError, match='contract'):
        m.validate_decoder_metadata([query,{**query,'crop':[64,64,64]}])


def test_fixed_score_floor_limits_only_predicted_binding_not_u_or_controlled_r(tmp_path):
    import nibabel as nib
    from anatobind.aur import eval as E
    torch.manual_seed(3)
    model = AnatoBindBrain(**TINY).eval()
    with torch.no_grad():
        model.events.presence.weight.zero_(); model.events.presence.bias.fill_(2)
        model.masks.embed_full[-1].weight.zero_(); model.masks.embed_full[-1].bias.zero_()
    shape=(16,16,8)
    for name,arr in [('image',np.ones(shape,np.float32)),('anatomy',np.full(shape,2,np.int16)),
                     ('lesion',np.pad(np.ones((4,4,4),np.int16),((2,10),(2,10),(2,2))))]:
        nib.save(nib.Nifti1Image(arr,np.eye(4)),str(tmp_path/(name+'.nii.gz')))
    row=dict(case='fixture',patient='fixture',source='pdgm',sequence='FLAIR',u_supervised=True,
             u_values=[1],a_ignore_values=[1],**{n:str(tmp_path/(n+'.nii.gz')) for n in ('image','anatomy','lesion')})
    res=E.evaluate_case(model,row,(8,16,16),torch.device('cpu'),decoder_profile='query_cc_v1',bind_score_threshold=1.)
    assert len(res['u']['dets']) == 1 and res['u']['dets'][0]['score'] == .5
    assert len(res['r']['instances']) == 1 and res['e2e'] == []
    assert len(res['legacy_reference_u']['dets']) == 1
    with pytest.raises(ValueError, match='binding.*threshold'):
        E.aggregate([res], {}, fixed_thresholds={'pdgm':.7})
    # Raising the threshold also changes Hungarian assignment, not merely coverage.
    res['e2e_score_floor']=.7
    with pytest.raises(ValueError, match='binding.*threshold'):
        E.aggregate([res], {}, fixed_thresholds={'pdgm':.8})


def test_binding_subset_compacts_map_but_retains_detector_ids():
    from anatobind.aur import eval as E
    inst=np.array([[[1,1,2,2,3,3]]],np.int32)
    rows=[dict(instance=1,score=.3),dict(instance=2,score=.9),dict(instance=3,score=.4)]
    compact,original=E.predicted_binding_subset(inst,rows,.7)
    np.testing.assert_array_equal(compact,np.array([[[0,0,1,1,0,0]]],np.int32))
    assert original == {1:2}


def test_fixed_point_binding_truth_does_not_lose_gt_to_filtered_low_score_box():
    from anatobind.aur import eval as E
    gt=[dict(instance=1,box=[0,0,0,2,2,2])]
    dets=[dict(instance=1,box=[0,0,0,2,2,3],score=.8),dict(instance=2,box=[0,0,0,2,2,2],score=.1)]
    assert E.predicted_detection_truth(gt,dets,np.array([4]),.7) == {1:4}

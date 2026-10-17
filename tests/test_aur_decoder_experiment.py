"""Fail-closed provenance and paired-reference validation before real replay."""
import copy
import importlib.util
from pathlib import Path

import pytest


def module():
    spec=importlib.util.spec_from_file_location('aur_compare_decoders',Path(__file__).resolve().parents[1]/'scripts/aur_compare_decoders.py')
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);return mod


def fixture():
    gt=dict(instance=1,box=[0,0,0,3,3,3],n_voxels=27,volume_mm3=27.)
    det=dict(instance=1,box=[0,0,0,3,3,3],n_voxels=27,volume_mm3=27.,score=.8)
    r=dict(instance=1,truth=0,r=0,b0=0,r_probs=[1.]+[0.]*13,query_iou=.5,in_crop_share=1.)
    saved=dict(source='pdgm',patient='fixture',case='fixture',sequence='FLAIR',a={'host_macro':.8,'entities':{'2':.9}},
               u={'gt':[gt],'dets':[det]},r={'instances':[r]},e2e=[])
    actual=copy.deepcopy(saved);actual['u']['dets'][0]['score']=.9
    actual['legacy_reference_u']={'dets':copy.deepcopy(saved['u']['dets'])}
    return saved,actual


def test_reference_verification_allows_only_the_new_u_output_to_change():
    m=module();saved,actual=fixture();m.verify_reference(actual,saved)
    for field in ('a','r','legacy_reference_u'):
        bad=copy.deepcopy(actual)
        if field=='a':bad['a']['host_macro']=.7
        elif field=='r':bad['r']['instances']=[]
        else:bad[field]['dets'][0]['score']=.7
        with pytest.raises(ValueError,match='reproduction'):
            m.verify_reference(bad,saved)


def test_resume_refuses_different_code_or_rules(tmp_path):
    m=module();out=tmp_path/'run';contract={'code_sha':'a','decoder_rules':{'version':1}}
    m.freeze_contract(out,contract,False)
    m.freeze_contract(out,contract,True)
    with pytest.raises(FileExistsError):m.freeze_contract(out,contract,False)
    with pytest.raises(ValueError,match='contract'):
        m.freeze_contract(out,{**contract,'code_sha':'changed'},True)

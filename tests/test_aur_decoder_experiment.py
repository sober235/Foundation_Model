"""Fail-closed provenance and paired-reference validation before real replay."""
import copy
import importlib.util
import json
from pathlib import Path
import signal
import subprocess
import sys

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


def test_raw_input_manifest_detects_changes_before_unprocessed_rows_run(tmp_path):
    m=module();p=tmp_path/'image.bin';p.write_bytes(b'first')
    row=dict(source='pdgm',patient='p',case='c',sequence='FLAIR',image=str(p),anatomy=str(p),lesion=None)
    before=m.input_manifest([row]);p.write_bytes(b'replaced');after=m.input_manifest([row])
    assert before != after
    out=tmp_path/'run';m.freeze_contract(out,{'raw':before},False)
    with pytest.raises(ValueError,match='contract'):m.freeze_contract(out,{'raw':after},True)


def test_report_finalization_recovers_partial_and_reuses_complete_report(tmp_path):
    m=module()
    from anatobind.aur import eval as E
    agg=E.aggregate([],{});dest=tmp_path/'query';dest.mkdir();(dest/'REPORT.md').write_text('interrupted')
    m.finalize_report(dest,agg,[],{})
    assert json.loads((dest/'cases.json').read_text()) == []
    assert list(tmp_path.glob('query.incomplete-*'))
    m.finalize_report(dest,agg,[],{})


def test_hard_deadline_terminates_even_a_running_work_item():
    path=Path(__file__).resolve().parents[1]/'scripts/aur_compare_decoders.py'
    source="import importlib.util,time; s=importlib.util.spec_from_file_location('x',%r); m=importlib.util.module_from_spec(s); s.loader.exec_module(m)\nwith m.hard_deadline(.2): time.sleep(5)" % str(path)
    proc=subprocess.run([sys.executable,'-c',source],capture_output=True,timeout=25)
    assert proc.returncode == -signal.SIGTERM,proc.stderr.decode()

# tests/test_aur_labels.py
import numpy as np
import pytest

import anatobind.aur.labels as L


def test_entity_map_round_trips_the_32_structures_and_drops_everything_else():
    assert L.N_ENTITIES == 32 and len(set(L.ENTITY_LABELS)) == 32
    seg = np.array(L.ENTITY_LABELS + (0, 77, 255), np.int16).reshape(5, 7)
    e = L.entity_map(seg)
    assert e.dtype == np.uint8 and list(e.ravel()[:32]) == list(range(1, 33)) and list(e.ravel()[32:]) == [0, 0, 0]
    back = L.entity_to_synthseg(e)
    assert back.dtype == np.int16 and list(back.ravel()[:32]) == list(L.ENTITY_LABELS) and list(back.ravel()[32:]) == [0, 0, 0]
    with pytest.raises(ValueError, match="outside"):
        L.entity_map(np.array([300]))
    with pytest.raises(ValueError, match="outside"):
        L.entity_to_synthseg(np.array([33]))


def test_host_map_follows_the_s4_host_classes_and_leaves_landmarks_out():
    assert L.N_HOSTS == 13 and L.NO_HOST == 13 and L.N_HOST_CLASSES == 14 and L.HOST_NAMES[-1] == "no_host"
    seg = np.array([2, 41, 4, 24, 16, 11, 50, 0, 7, 46], np.int16)
    h = L.host_map(seg)
    names = [L.HOST_NAMES[v - 1] if v else "none" for v in h]
    assert names == ["white_matter_left", "white_matter_right", "none", "none", "brainstem", "basal_ganglia_left",
                     "basal_ganglia_right", "none", "cerebellum_left", "cerebellum_right"]


def test_contralateral_is_an_involution_and_the_brainstem_has_none():
    for h in range(L.N_HOSTS):
        c = L.contralateral(h)
        if L.HOST_SIDE[h] is None:
            assert c is None and L.HOST_NAMES[h] == "brainstem"
        else:
            assert c != h and L.contralateral(c) == h and L.HOST_TISSUE[c] == L.HOST_TISSUE[h]
    assert len(L.TISSUES) == 7


def test_entity_host_mapping_and_sequence_types():
    assert L.ENTITY_HOST[L.ENTITY_INDEX[11]] == L.HOST_NAMES.index("basal_ganglia_left")
    assert L.ENTITY_HOST[L.ENTITY_INDEX[4]] == -1 and L.ENTITY_HOST[L.ENTITY_INDEX[24]] == -1
    assert L.ENTITY_HOST[L.ENTITY_INDEX[16]] == L.HOST_NAMES.index("brainstem")
    assert L.SEQ_TYPES == ("T1", "T1c", "T2", "FLAIR", "DWI", "ADC") and L.SEQ_INDEX["DWI"] == 4

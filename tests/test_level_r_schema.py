import pytest

from anatobind.level_r.schema import (
    ADJACENCY, AMBIGUITY, LESION_TYPES, LOBES, LOCAL_QUALITY, MAX_ACCEPTABLE, PRIMARY_HOSTS, SIDES, TOPOGRAPHY, InvalidLabel,
    enums, validate_adjudication, validate_label,
)

GOOD = {"primary_host": "white_matter", "acceptable_hosts": ["white_matter", "cortex"], "topography": "juxtacortical",
        "adjacency": ["adjacent_to_cortex"], "ambiguity": "two_host", "not_a_lesion": False, "lesion_type": "nonspecific_wm_lesion",
        "side": "image_left", "lobe": "frontal", "local_quality": "good", "confidence": 4, "comment": "贴皮层"}
NAL = {"not_a_lesion": True, "local_quality": "poor", "confidence": 2, "adjacency": []}


def test_valid_label_is_normalised_and_keeps_every_field():
    out = validate_label(GOOD)
    assert out == {**GOOD, "comment": "贴皮层"}


@pytest.mark.parametrize("bad", [
    {**GOOD, "primary_host": None},                                    # 缺主宿主
    {**GOOD, "primary_host": "ventricle"},                             # 不在枚举里
    {**GOOD, "acceptable_hosts": ["white_matter", "cortex", "thalamus"]},   # 集合超 2
    {**GOOD, "acceptable_hosts": ["cortex"]},                          # 集合不含主宿主
    {**GOOD, "acceptable_hosts": ["white_matter", "white_matter"]},    # 重复
    {**GOOD, "confidence": 0}, {**GOOD, "confidence": 6}, {**GOOD, "confidence": "4"}, {**GOOD, "confidence": True},
    {**GOOD, "topography": "lobar"}, {**GOOD, "ambiguity": "unsure"}, {**GOOD, "local_quality": "ok"},
    {**GOOD, "adjacency": ["none", "adjacent_to_cortex"]},             # none 排斥其他
    {**GOOD, "adjacency": ["near_cortex"]},
    {**GOOD, "not_a_lesion": True},                                    # 不是病灶却给了宿主
    {**GOOD, "lesion_type": None}, {**GOOD, "lesion_type": ""}, {**GOOD, "lesion_type": "mass"},   # 病灶类型必填且在枚举里
    {k: v for k, v in GOOD.items() if k != "lesion_type"},
    {**GOOD, "side": None}, {**GOOD, "side": "left"}, {**GOOD, "side": "patient_left"},           # 侧别按屏幕左右
    {**GOOD, "lobe": None}, {**GOOD, "lobe": "limbic"}, {k: v for k, v in GOOD.items() if k != "lobe"},
])
def test_invalid_labels_raise(bad):
    with pytest.raises(InvalidLabel):
        validate_label(bad)


def test_not_a_lesion_answer_needs_no_host_topography_or_ambiguity():
    out = validate_label(NAL)
    assert out["primary_host"] is None and out["acceptable_hosts"] == [] and out["topography"] is None
    assert out["ambiguity"] is None and out["not_a_lesion"] is True and out["comment"] == ""
    assert out["lesion_type"] is None and out["side"] is None and out["lobe"] is None


@pytest.mark.parametrize("key, value", [("lesion_type", "lacunar_infarct"), ("side", "midline"), ("lobe", "not_applicable")])
def test_a_not_a_lesion_answer_carries_no_lesion_type_side_or_lobe(key, value):
    with pytest.raises(InvalidLabel):
        validate_label({**NAL, key: value})
    assert validate_label({**NAL, key: None})[key] is None and validate_label({**NAL, key: ""})[key] is None


def test_adjudication_requires_a_reason_and_not_quality_or_confidence():
    p = {k: v for k, v in GOOD.items() if k not in ("local_quality", "confidence")}
    with pytest.raises(InvalidLabel):
        validate_adjudication(p)
    out = validate_adjudication({**p, "reason": "皮层信号连续"})
    assert out["reason"] == "皮层信号连续" and out["local_quality"] is None and out["confidence"] is None
    assert (out["lesion_type"], out["side"], out["lobe"]) == ("nonspecific_wm_lesion", "image_left", "frontal")
    with pytest.raises(InvalidLabel):                                   # the ruling needs lesion type, side and lobe too
        validate_adjudication({**p, "side": None, "reason": "皮层信号连续"})


def test_enums_expose_every_list_and_the_cap():
    e = enums()
    assert e == {"primary_hosts": list(PRIMARY_HOSTS), "topography": list(TOPOGRAPHY), "adjacency": list(ADJACENCY),
                 "ambiguity": list(AMBIGUITY), "local_quality": list(LOCAL_QUALITY), "lesion_types": list(LESION_TYPES),
                 "sides": list(SIDES), "lobes": list(LOBES), "max_acceptable": MAX_ACCEPTABLE}
    assert "other" in PRIMARY_HOSTS and "none" in ADJACENCY and MAX_ACCEPTABLE == 2


def test_lesion_type_side_and_lobe_take_the_decided_values():
    assert LESION_TYPES == ("nonspecific_wm_lesion", "lacunar_infarct", "perivascular_space", "other")
    assert SIDES == ("image_left", "image_right", "midline")           # the side as seen on screen, not the patient's side
    assert LOBES == ("frontal", "parietal", "temporal", "occipital", "insular", "not_applicable")

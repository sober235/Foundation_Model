import pytest

from anatobind.level_r.schema import (
    ADJACENCY, AMBIGUITY, LOCAL_QUALITY, MAX_ACCEPTABLE, PRIMARY_HOSTS, TOPOGRAPHY, InvalidLabel, enums,
    validate_adjudication, validate_label,
)

GOOD = {"primary_host": "white_matter", "acceptable_hosts": ["white_matter", "cortex"], "topography": "juxtacortical",
        "adjacency": ["adjacent_to_cortex"], "ambiguity": "two_host", "not_a_lesion": False, "local_quality": "good",
        "confidence": 4, "comment": "贴皮层"}


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
])
def test_invalid_labels_raise(bad):
    with pytest.raises(InvalidLabel):
        validate_label(bad)


def test_not_a_lesion_answer_needs_no_host_topography_or_ambiguity():
    out = validate_label({"not_a_lesion": True, "local_quality": "poor", "confidence": 2, "adjacency": []})
    assert out["primary_host"] is None and out["acceptable_hosts"] == [] and out["topography"] is None
    assert out["ambiguity"] is None and out["not_a_lesion"] is True and out["comment"] == ""


def test_adjudication_requires_a_reason_and_not_quality_or_confidence():
    p = {k: v for k, v in GOOD.items() if k not in ("local_quality", "confidence")}
    with pytest.raises(InvalidLabel):
        validate_adjudication(p)
    out = validate_adjudication({**p, "reason": "皮层信号连续"})
    assert out["reason"] == "皮层信号连续" and out["local_quality"] is None and out["confidence"] is None


def test_enums_expose_every_list_and_the_cap():
    e = enums()
    assert e == {"primary_hosts": list(PRIMARY_HOSTS), "topography": list(TOPOGRAPHY), "adjacency": list(ADJACENCY),
                 "ambiguity": list(AMBIGUITY), "local_quality": list(LOCAL_QUALITY), "max_acceptable": MAX_ACCEPTABLE}
    assert "other" in PRIMARY_HOSTS and "none" in ADJACENCY and MAX_ACCEPTABLE == 2

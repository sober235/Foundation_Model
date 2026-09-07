from collections import Counter

from anatobind.data_engine.splits import five_fold_by_scan, stratum_of


def test_stratum_buckets():
    assert [stratum_of(n) for n in (0, 1, 2, 3, 9)] == [0, 1, 1, 2, 2]


def test_every_scan_gets_exactly_one_fold_and_sizes_are_balanced():
    scans = [f"MTR_{i:03d}" for i in range(155)]
    strata = {s: stratum_of(i % 4) for i, s in enumerate(scans)}
    folds = five_fold_by_scan(scans, strata, seed=0)
    assert set(folds) == set(scans)
    sizes = Counter(folds.values())
    assert set(sizes) == {0, 1, 2, 3, 4} and max(sizes.values()) - min(sizes.values()) <= 1


def test_strata_are_spread_across_folds_and_result_is_deterministic():
    scans = [f"MTR_{i:03d}" for i in range(155)]
    strata = {s: stratum_of(i % 4) for i, s in enumerate(scans)}
    a = five_fold_by_scan(scans, strata, seed=0)
    b = five_fold_by_scan(scans, strata, seed=0)
    assert a == b
    per_fold = Counter((a[s], strata[s]) for s in scans)
    for st in (0, 1, 2):
        counts = [per_fold[(f, st)] for f in range(5)]
        assert max(counts) - min(counts) <= 1

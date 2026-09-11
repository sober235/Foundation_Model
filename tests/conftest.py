import pytest

from synth import write_synthetic_export


@pytest.fixture
def synthetic_m1r(tmp_path):
    """Five synthetic scans, one per fold, exported and cached. Imports stay local so the rest of the
    suite does not depend on modules this fixture needs."""
    from anatobind.train.cache import cache_scan

    export, cache = tmp_path / "m1r", tmp_path / "cache"
    scans = [f"MTR_{i:03d}" for i in range(1, 6)]
    write_synthetic_export(export, scans, {s: i for i, s in enumerate(scans)})
    for s in scans:
        cache_scan(export / s, cache / s)
    return export, cache, scans

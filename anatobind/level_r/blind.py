"""What the browser may receive (spec R5). Whitelists, not blacklists: a new registry column can never leak by
accident. FORBIDDEN is the list of key substrings that must never appear in anything sent out; assert_blind walks a
whole response and is applied to every JSON body by the server, so the whitelists and the blacklist guard each other."""
LESION_KEYS = ("lesion_id", "code", "volume_code", "z0", "z1", "boxes")
VOLUME_KEYS = ("shape", "spacing_slice_mm", "spacing_row_mm", "spacing_col_mm", "window")
FORBIDDEN = ("label", "d_interface", "delta_d", "d1_", "stratum", "series", "patient", "stem", "file", "host_lookup",
             "host_class", "band", "synthseg", "lookup")


class BlindingError(ValueError):
    pass


def _check_key(k):
    low = str(k).lower()
    for f in FORBIDDEN:
        if f in low:
            raise BlindingError(f"key {k!r} matches forbidden substring {f!r}")


def _pick(d, keys):
    out = {k: d[k] for k in keys if k in d}
    for k in out:
        _check_key(k)
    return out


def blind_lesion(d):
    return _pick(d, LESION_KEYS)


def blind_volume_meta(d):
    return _pick(d, VOLUME_KEYS)


def assert_blind(obj):
    """Raise BlindingError if any dict key (at any depth) matches FORBIDDEN; return obj unchanged otherwise."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            _check_key(k)
            assert_blind(v)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            assert_blind(v)
    return obj

"""Level R annotation schema (spec §5; v2.6 §3, §7.5). English keys go into the database, the Chinese labels live in
the app. validate_label is the single validation both the server and the tests rely on; the browser repeats the same
rules for immediate feedback but is never trusted."""

PRIMARY_HOSTS = ("white_matter", "cortex", "thalamus", "basal_ganglia", "brainstem", "cerebellum", "other")
TOPOGRAPHY = ("periventricular", "juxtacortical", "cortical", "deep_white_matter", "infratentorial")
ADJACENCY = ("adjacent_to_cortex", "adjacent_to_ventricle", "crosses_boundary", "none")
AMBIGUITY = ("certain", "two_host", "multi_structure", "insufficient_resolution")
LOCAL_QUALITY = ("good", "fair", "poor")
MAX_ACCEPTABLE = 2
NOT_A_LESION = "not_a_lesion"      # the class a not_a_lesion answer takes in agreement statistics
MAX_COMMENT = 2000


class InvalidLabel(ValueError):
    pass


def _choice(p, key, allowed, optional):
    v = p.get(key)
    if optional and v in (None, ""):
        return None
    if v not in allowed:
        raise InvalidLabel(f"{key} {v!r} must be one of {allowed}")
    return v


def validate_label(p, require_quality=True):
    """Return the normalised answer or raise InvalidLabel. p: the JSON body of one submission.
    require_quality=False (adjudications) drops local_quality and confidence."""
    out = {"not_a_lesion": bool(p.get("not_a_lesion", False))}
    acc = p.get("acceptable_hosts") or []
    if not isinstance(acc, list) or len(acc) != len(set(acc)):
        raise InvalidLabel("acceptable_hosts must be a list without repeats")
    if len(acc) > MAX_ACCEPTABLE:
        raise InvalidLabel(f"acceptable_hosts has {len(acc)} entries; at most {MAX_ACCEPTABLE}")
    for h in acc:
        if h not in PRIMARY_HOSTS:
            raise InvalidLabel(f"unknown host {h!r}")
    host = p.get("primary_host") or None
    if out["not_a_lesion"]:
        if host is not None or acc:
            raise InvalidLabel("a not_a_lesion answer carries no host")
    else:
        if host not in PRIMARY_HOSTS:
            raise InvalidLabel(f"primary_host {host!r} is required and must be one of {PRIMARY_HOSTS}")
        if host not in acc:
            raise InvalidLabel("acceptable_hosts must contain primary_host")
    out["primary_host"], out["acceptable_hosts"] = host, list(acc)
    out["topography"] = _choice(p, "topography", TOPOGRAPHY, optional=out["not_a_lesion"])
    out["ambiguity"] = _choice(p, "ambiguity", AMBIGUITY, optional=out["not_a_lesion"])
    adj = p.get("adjacency") or []
    if not isinstance(adj, list) or len(adj) != len(set(adj)) or any(a not in ADJACENCY for a in adj):
        raise InvalidLabel(f"adjacency must be a subset of {ADJACENCY}")
    if "none" in adj and len(adj) > 1:
        raise InvalidLabel("adjacency 'none' excludes the other choices")
    out["adjacency"] = list(adj)
    if require_quality:
        out["local_quality"] = _choice(p, "local_quality", LOCAL_QUALITY, optional=False)
        c = p.get("confidence")
        if isinstance(c, bool) or not isinstance(c, int) or not 1 <= c <= 5:
            raise InvalidLabel("confidence must be an integer 1..5")
        out["confidence"] = c
    else:
        out["local_quality"], out["confidence"] = None, None
    out["comment"] = str(p.get("comment") or "")[:MAX_COMMENT]
    return out


def validate_adjudication(p):
    out = validate_label(p, require_quality=False)
    reason = str(p.get("reason") or "").strip()
    if not reason:
        raise InvalidLabel("reason is required for an adjudication")
    out["reason"] = reason[:MAX_COMMENT]
    return out


def enums():
    return {"primary_hosts": list(PRIMARY_HOSTS), "topography": list(TOPOGRAPHY), "adjacency": list(ADJACENCY),
            "ambiguity": list(AMBIGUITY), "local_quality": list(LOCAL_QUALITY), "max_acceptable": MAX_ACCEPTABLE}

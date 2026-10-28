"""Observation-local geometry, with separate TRAIN and held-out entries, NOT_EVIDENCE.

The source host map is categorical (0 absent, 1..13 host), on the *complete*
source instance grid. Legacy host probabilities cannot establish overlap:
their nearest-host fallback can produce the same one-hot as real overlap.
"""
import numpy as np
from scipy import ndimage

from anatobind.aur.labels import N_HOSTS, N_HOST_CLASSES, contralateral, discrete_labels
from anatobind.eval.lesion_components import min_voxels_for

POLICY = "geometry_weak2d_v1"


def _categorical(values, maximum=None):
    data = discrete_labels(values, dtype=np.int64)
    if (data < 0).any() or (maximum is not None and (data > maximum).any()):
        raise ValueError("planar geometry contains invalid categorical labels")
    return data


def _plane(values):
    data = np.asarray(values)
    if data.ndim == 3 and data.shape[0] == 1:
        data = data[0]
    if data.ndim != 2:
        raise ValueError("transformed planar geometry must contain one plane")
    return data


def _counts(hosts, mask):
    return np.bincount(hosts[mask], minlength=N_HOSTS + 1)[1:]


def _unique_primary(counts):
    if counts.sum() == 0:
        return -1, "no_overlap"
    if np.count_nonzero(counts == counts.max()) != 1:
        return -1, "tied_host"
    return int(counts.argmax()), ""


def _qualify_geometry(source_instances, source_host_map, transformed_plane,
                                  valid, a_ignore, source_instance_ids, pixel_area_mm2, *,
                                  voxel_volume_mm3, gate="", a_supervised=True,
                                  u_supervised=True, r_supervised=True, min_area_mm2=1.0, max_instances=64):
    """Return aligned weak targets for the final transformed observation components.

    ``transformed_plane`` has categorical ``instance`` (renumbered 1..N) and
    ``host_map`` arrays on the valid/A-ignore grid. Its one-based source IDs can
    repeat when a single source 26-component splits into planar 8-components.
    A component must still be a single 8-component after all transforms.

    Ineligible rows have host=-1, zero probabilities/weight and no provenance;
    -1 means unavailable, never NO_HOST. Adapters retain their original targets
    for these rows and exclude them with r_eligible. Negatives follow the AUR
    contralateral/second-overlap contract, using only the observed second host.
    """
    if any(not np.isfinite(v) or v <= 0 for v in
           (pixel_area_mm2, voxel_volume_mm3, min_area_mm2)):
        raise ValueError("planar geometry requires finite positive physical measures")
    source = _categorical(source_instances)
    source_hosts = _categorical(source_host_map, N_HOSTS)
    if source.ndim != 3 or source_hosts.shape != source.shape:
        raise ValueError("source instance and host maps must share one volume grid")
    instance = _categorical(_plane(transformed_plane["instance"]))
    hosts = _categorical(_plane(transformed_plane["host_map"]), N_HOSTS)
    valid = _plane(valid)
    ignore = _plane(a_ignore)
    if any(v.shape != instance.shape for v in (hosts, valid, ignore)):
        raise ValueError("transformed planar maps must share one observation grid")
    if (not np.isin(valid, [0, 1]).all() or not np.isin(ignore, [0, 1]).all()):
        raise ValueError("planar valid and anatomy-ignore masks must be binary")
    valid, ignore = valid.astype(bool), ignore.astype(bool)
    ids = _categorical(source_instance_ids)
    if ids.ndim != 1 or (ids <= 0).any() or (max_instances is not None and len(ids) > max_instances):
        raise ValueError("planar source identities must align within query capacity")
    n = len(ids)
    if not np.array_equal(np.unique(instance[instance > 0]), np.arange(1, n + 1)):
        raise ValueError("planar component identities do not align with source mapping")
    source_boxes = ndimage.find_objects(source)
    if any(k > len(source_boxes) or source_boxes[k - 1] is None for k in ids):
        raise ValueError("planar source identity is absent from source volume")
    result = {"r_eligible": np.zeros(n, bool), "host": np.full(n, -1, np.int64),
              "probs": np.zeros((n, N_HOST_CLASSES), np.float32),
              "negatives": np.full((n, 2), -1, np.int64),
              "weight": np.zeros(n, np.float32), "provenance": [""] * n,
              "reasons": [""] * n, "evidence": "NOT_EVIDENCE"}
    gate = (gate or ("anatomy_unqualified" if not a_supervised else
            "lesion_unqualified" if not u_supervised else
            "relation_unqualified" if not r_supervised else ""))
    evidence = {}
    for source_id in np.unique(ids):
        box = source_boxes[source_id - 1]
        mask = source[box] == source_id
        _, count = ndimage.label(mask, structure=np.ones((3, 3, 3), bool))
        reason = ("source_not_single_component" if count != 1 else
                  "source_below_volume" if mask.sum() < min_voxels_for(voxel_volume_mm3) else "")
        primary, overlap_reason = _unique_primary(_counts(source_hosts[box], mask))
        evidence[int(source_id)] = (primary, reason or ("source_" + overlap_reason if overlap_reason else ""))
    for i, source_id in enumerate(ids):
        mask = instance == i + 1
        _, count = ndimage.label(mask, structure=np.ones((3, 3), bool))
        if count != 1:
            raise ValueError("each transformed planar identity must be a single 8-component")
        primary, reason = evidence[int(source_id)]
        counts = _counts(hosts, mask & valid)  # Deliberately independent of A-ignore.
        plane_primary, plane_reason = _unique_primary(counts)
        reason = (gate or reason or
                  ("component_outside_valid" if np.any(mask & ~valid) else "") or
                  ("plane_below_area" if mask.sum() * pixel_area_mm2 + 1e-9 < max(1.0, min_area_mm2) else "") or
                  ("plane_" + plane_reason if plane_reason else "") or
                  ("plane_source_host_disagree" if plane_primary != primary else "") or
                  ("host_not_visible" if not np.any((hosts == primary + 1) & valid & ~ignore) else ""))
        result["reasons"][i] = reason
        if reason:
            continue
        result["r_eligible"][i] = True
        result["host"][i] = primary
        result["probs"][i, primary] = 1.0
        contra = contralateral(primary)
        result["negatives"][i, 0] = -1 if contra is None else contra
        counts[primary] = 0
        if counts.max() > 0:
            result["negatives"][i, 1] = int(counts.argmax())
        result["weight"][i] = 1.0
        result["provenance"][i] = POLICY
    return result


def qualify_planar_weak_relations(source_instances, source_host_map, transformed_plane,
                                 valid, a_ignore, source_instance_ids, pixel_area_mm2, *,
                                 voxel_volume_mm3, split="train", a_supervised=True,
                                 u_supervised=True, r_supervised=True, min_area_mm2=1.0,
                                 max_instances=64):
    """TRAIN-only weak supervision; held-out rows remain unavailable."""
    return _qualify_geometry(source_instances,source_host_map,transformed_plane,valid,a_ignore,
        source_instance_ids,pixel_area_mm2,voxel_volume_mm3=voxel_volume_mm3,
        gate="not_train" if split != "train" else "",a_supervised=a_supervised,
        u_supervised=u_supervised,r_supervised=r_supervised,min_area_mm2=min_area_mm2,
        max_instances=max_instances)


def planar_evaluation_reference(source_instances, source_host_map, transformed_plane,
                                valid, a_ignore, source_instance_ids, pixel_area_mm2, *,
                                plane_index, split, role, **kwargs):
    """Held-out observation geometry only; never establishes a passing gate.

    Uses the same pure overlap qualification as TRAIN without relabeling the
    row's role. The caller supplies the actual unaugmented canonical plane and
    its source identities, rather than inherited nearest-host probabilities.
    """
    if split != "val" or role != "development_validation":
        raise ValueError("geometry reference requires development validation")
    source, hosts = np.asarray(source_instances), np.asarray(source_host_map)
    instance = _plane(transformed_plane['instance'])
    if (source.ndim!=3 or hosts.shape!=source.shape or type(plane_index) is not int or
            not 0<=plane_index<source.shape[0] or instance.shape!=source.shape[1:] or
            not np.array_equal(_plane(transformed_plane['host_map']),hosts[plane_index])):
        raise ValueError('evaluation source plane geometry differs')
    for i,k in enumerate(source_instance_ids,1):
        if np.any(source[plane_index][instance==i]!=k):
            raise ValueError('evaluation source plane identity differs')
    result = _qualify_geometry(source_instances,source_host_map,transformed_plane,valid,a_ignore,
        source_instance_ids,pixel_area_mm2,max_instances=None,**kwargs)
    result["provenance"] = ["geometry_evaluation2d_v1" if ok else ""
                            for ok in result["r_eligible"]]
    return result

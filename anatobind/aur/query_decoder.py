"""Frozen, GT-free per-query connected-component instance decoding.

Only present queries cross to CPU as float32. Pending query work is bounded;
retained candidates contain bounding-box packed masks rather than voxel indices.
"""
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import dataclass

import numpy as np
import torch
from scipy import ndimage

from anatobind.eval.lesion_components import MIN_MM3, min_voxels_for

DECODER_PROFILES = ('legacy_max', 'query_cc_v1')
_STRUCTURE = np.ones((3, 3, 3), dtype=bool)


def decoder_rules(profile):
    """Return an independent JSON-serializable description of the frozen rules."""
    common = dict(presence_threshold=.5, presence_comparison='>', mask_threshold=.3,
                  mask_comparison='>=', connectivity=26, min_volume_mm3=MIN_MM3)
    if profile == 'legacy_max':
        return dict(common, profile=profile, fusion='gaussian_weighted_window_mean_of_present_query_max',
                    instances='global_connected_components', score='fused_probability_component_mean',
                    score_filter='after_component_extraction')
    if profile == 'query_cc_v1':
        return dict(common, profile=profile, fusion='independent_window_query_components',
                    score='raw_float32_parent_component_mean_float64_accumulation',
                    priority=['descending_score', 'window_start_zyx', 'query', 'component'],
                    nms_iou=.5, nms_comparison='>=', nms_masks='original_valid_global_masks',
                    ownership='all_nms_survivors_in_priority_order_before_residual_components',
                    residual_score='inherit_parent', backfill=False, residual_nms=False,
                    score_filter='after_completed_ownership_and_residual_floor')
    raise ValueError(f'Unknown decoder profile: {profile!r}')


@dataclass(frozen=True)
class _Candidate:
    window: tuple
    query: int
    component: int
    box: tuple
    packed: np.ndarray
    n_voxels: int
    score: float

    @property
    def slices(self):
        return tuple(slice(a, b) for a, b in self.box)

    def mask(self):
        shape = tuple(b - a for a, b in self.box)
        return np.unpackbits(self.packed, count=int(np.prod(shape))).reshape(shape).astype(bool)

    @property
    def priority(self):
        return (-self.score, tuple(a for a, _ in self.window), self.query, self.component)


def _query_candidates(prob, query, window, valid, offset, floor):
    """Label one already-clipped query; retain only packed eligible components."""
    labels, raw_count = ndimage.label((prob >= .3) & valid, structure=_STRUCTURE)
    sizes = np.bincount(labels.ravel(), minlength=raw_count + 1)
    candidates = []
    for component, slices in enumerate(ndimage.find_objects(labels), 1):
        n = int(sizes[component])
        if slices is None or n < floor:
            continue
        mask = labels[slices] == component
        score = float(np.sum(prob[slices][mask], dtype=np.float64) / n)
        box = tuple((sl.start + a, sl.stop + a) for sl, a in zip(slices, offset))
        candidates.append(_Candidate(window, query, component, box, np.packbits(mask.ravel()), n, score))
    return raw_count, candidates


def _numpy(value, dtype):
    if isinstance(value, torch.Tensor):
        value = value.detach().to(device='cpu', dtype=torch.float32).numpy()
    return np.array(value, dtype=dtype, copy=True)


def _mask_iou(candidate, candidate_mask, earlier):
    """Exact original-mask IoU, with a safe intersection-volume upper bound."""
    overlap = tuple((max(a, c), min(b, d)) for (a, b), (c, d) in zip(candidate.box, earlier.box))
    if any(b <= a for a, b in overlap):
        return 0.
    upper = min(candidate.n_voxels, earlier.n_voxels, int(np.prod([b - a for a, b in overlap])))
    # IoU >= .5 requires intersection >= (n1+n2)/3.
    if 3 * upper < candidate.n_voxels + earlier.n_voxels:
        return 0.
    local = tuple(slice(a - c, b - c) for (a, b), (c, _) in zip(overlap, candidate.box))
    previous = tuple(slice(a - c, b - c) for (a, b), (c, _) in zip(overlap, earlier.box))
    intersection = int(np.count_nonzero(candidate_mask[local] & earlier.mask()[previous]))
    return intersection / (candidate.n_voxels + earlier.n_voxels - intersection)


class QueryCCDecoder:
    """Bounded observer for natural-window predictions, followed by deterministic decoding.

    ``nms_trace`` contains only local candidate ranks and exact suppression IoUs;
    the public stats and instance rows contain no case or patient identifiers.
    """
    def __init__(self, shape, voxel_mm3, workers=4):
        if len(shape) != 3 or any(int(n) != n or n <= 0 for n in shape):
            raise ValueError('shape must contain three positive integer dimensions')
        if not np.isfinite(voxel_mm3) or voxel_mm3 <= 0:
            raise ValueError('voxel_mm3 must be finite and positive')
        if int(workers) != workers or workers <= 0:
            raise ValueError('workers must be positive')
        self.shape = tuple(int(n) for n in shape)
        self.voxel_mm3 = float(voxel_mm3)
        self.workers = min(int(workers), 4)
        self.floor = min_voxels_for(self.voxel_mm3)
        self._executor = ThreadPoolExecutor(max_workers=self.workers)
        self._pending = deque()
        self._candidates = []
        self._raw_components = 0
        self._finished = False
        self.nms_trace = []

    def _drain_one(self):
        raw_count, candidates = self._pending.popleft().result()
        self._raw_components += raw_count
        self._candidates.extend(candidates)

    def __call__(self, *, window, event_prob, event_presence, entity_prob, valid, weight):
        if self._finished:
            raise RuntimeError('Cannot collect predictions after finish')
        window = tuple(tuple(int(x) for x in pair) for pair in window)
        expected = tuple(b - a for a, b in window)
        if len(window) != 3 or any(n <= 0 for n in expected):
            raise ValueError('window must contain three nonempty half-open intervals')
        if tuple(event_prob.shape[1:]) != expected or tuple(valid.shape) != expected:
            raise ValueError('prediction and valid shapes must match the window')
        presence = _numpy(event_presence, np.float32)
        if presence.shape != (len(event_prob),):
            raise ValueError('event_presence must have one probability per query')
        overlap = tuple((max(a, 0), min(b, n)) for (a, b), n in zip(window, self.shape))
        if any(b <= a for a, b in overlap):
            return
        src = tuple(slice(a - c, b - c) for (a, b), (c, _) in zip(overlap, window))
        valid_cpu = _numpy(valid[src], np.float32) > 0
        offset = tuple(a for a, _ in overlap)
        for query in np.flatnonzero(presence > .5):
            if len(self._pending) >= self.workers:
                self._drain_one()
            # Copy each present query before submission: callers may reuse CPU tensors.
            prob = _numpy(event_prob[(int(query),) + src], np.float32)
            if not np.isfinite(prob).all():
                raise ValueError('event_prob must contain finite probabilities')
            future = self._executor.submit(_query_candidates, prob, int(query), window, valid_cpu, offset, self.floor)
            self._pending.append(future)

    def finish(self):
        """Finish ownership before splitting/flooring; return int32 instances and provenance."""
        if self._finished:
            raise RuntimeError('finish can only be called once')
        self._finished = True
        try:
            while self._pending:
                self._drain_one()
        finally:
            self._executor.shutdown(wait=True)
        candidates = sorted(self._candidates, key=lambda c: c.priority)
        survivors = []
        for rank, candidate in enumerate(candidates, 1):
            mask = candidate.mask()
            for earlier_rank, earlier in survivors:
                iou = _mask_iou(candidate, mask, earlier)
                if iou >= .5:
                    self.nms_trace.append(dict(rank=rank, suppressor_rank=earlier_rank, iou=iou))
                    break
            else:
                survivors.append((rank, candidate))
        owner = np.zeros(self.shape, np.int32)
        for rank, candidate in survivors:
            view = owner[candidate.slices]
            view[candidate.mask() & (view == 0)] = rank
        instance = np.zeros(self.shape, np.int32)
        rows = []
        fully_occluded, tiny_fragments = 0, 0
        for rank, candidate in survivors:
            residual = owner[candidate.slices] == rank
            if not residual.any():
                fully_occluded += 1
                continue
            labels, count = ndimage.label(residual, structure=_STRUCTURE)
            sizes = np.bincount(labels.ravel(), minlength=count + 1)
            for child, slices in enumerate(ndimage.find_objects(labels), 1):
                n = int(sizes[child])
                if slices is None or n < self.floor:
                    tiny_fragments += 1
                    continue
                box = tuple((s.start + a, s.stop + a) for s, (a, _) in zip(slices, candidate.box))
                global_slices = tuple(slice(a, b) for a, b in box)
                index = len(rows) + 1
                instance[global_slices][labels[slices] == child] = index
                origin = dict(window=[list(w) for w in candidate.window], query=candidate.query,
                              component=candidate.component, rank=rank, parent_score=candidate.score,
                              parent_voxels=candidate.n_voxels, residual_child=child,
                              retained_fraction=n / candidate.n_voxels)
                rows.append(dict(instance=index, box=[a for a, _ in box] + [b for _, b in box],
                                 n_voxels=n, volume_mm3=n * self.voxel_mm3, score=candidate.score,
                                 detector_origin=origin))
        stats = dict(raw_components=self._raw_components, eligible_candidates=len(candidates),
                     suppressed_candidates=len(self.nms_trace), fully_occluded_candidates=fully_occluded,
                     tiny_fragments=tiny_fragments, final_instances=len(rows),
                     provenance_contract=dict(profile='query_cc_v1', query='natural_window_detector_query',
                                              score='fixed_parent_score', retained_fraction='child_voxels/parent_voxels',
                                              component='one_based_original_raster_label',
                                              rank='one_based_canonical_candidate_rank',
                                              residual_child='one_based_residual_raster_label'))
        return instance, rows, stats


def filter_instances(instance_map, rows, score_threshold):
    """Apply an inclusive score threshold (None keeps all), compacting IDs and origins."""
    source = np.asarray(instance_map)
    kept = []
    # An ID lookup avoids one whole-volume scan per kept instance.
    lookup = np.zeros(int(source.max(initial=0)) + 1, np.int32)
    for row in rows:
        if score_threshold is None or row['score'] >= score_threshold:
            copied = deepcopy(row)
            copied['instance'] = len(kept) + 1
            lookup[row['instance']] = copied['instance']
            kept.append(copied)
    return lookup[source], kept

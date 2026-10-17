"""Two-pass, truth-free confidence measurements on unchanged query-CC instances.

Only packed candidates, the final label map, scalar measurements and hashes persist.
The second pass must observe exactly the first pass's float32 query probabilities.
"""
from collections import defaultdict
from copy import deepcopy
import hashlib
import json

import numpy as np
import torch

from anatobind.aur.infer import _inside
from anatobind.aur.query_decoder import QueryCCDecoder


def array_digest(value):
    value = np.asarray(value)
    h = hashlib.sha256(str((value.shape, value.dtype.str)).encode())
    h.update(value.tobytes(order='C'))
    return h.hexdigest()


def _numpy(value):
    if isinstance(value, torch.Tensor):
        value = value.detach().to(device='cpu', dtype=torch.float32).numpy()
    return np.array(value, dtype=np.float32, copy=True, order='C')


class ResidualConfidenceAudit:
    def __init__(self, shape, voxel_mm3, workers=4):
        self.decoder = QueryCCDecoder(shape, voxel_mm3, workers)
        self.shape = self.decoder.shape
        self._phase, self._closed = 'first', False
        self._events, self._keys = [], set()
        self._replayed = 0
        self._parents, self._children = {}, []
        self._children_by_parent = defaultdict(list)
        self._groups = defaultdict(lambda: defaultdict(list))

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.decoder._executor.shutdown(wait=True, cancel_futures=True)
        self._closed = True

    def _require(self, phase):
        if self._closed or self._phase != phase:
            raise ValueError('Invalid confidence diagnostic phase')

    def _event(self, window, event_prob, event_presence, valid):
        if len(window) != 3 or any(len(p) != 2 or p[0] >= p[1] for p in window):
            raise ValueError('Invalid three-dimensional window')
        if any(int(v) != v for p in window for v in p):
            raise ValueError('Window coordinates must be integers')
        key = tuple(tuple(int(v) for v in p) for p in window)
        expected = tuple(b - a for a, b in key)
        if event_prob.dtype not in (torch.float32, np.dtype('float32')):
            raise ValueError('Raw probability input must be float32, not a quantized cache')
        if event_prob.ndim != 4 or tuple(event_prob.shape[1:]) != expected or tuple(valid.shape) != expected:
            raise ValueError('Observer probability/validity shape differs from window')
        presence = _numpy(event_presence)
        if presence.shape != (len(event_prob),) or not np.isfinite(presence).all():
            raise ValueError('Invalid query presence')
        if ((presence < 0) | (presence > 1)).any():
            raise ValueError('Presence is not a probability')
        dst, src = _inside(key, self.shape)
        valid_cpu = _numpy(valid[src])
        if not np.isfinite(valid_cpu).all():
            raise ValueError('Nonfinite validity')
        record = dict(window=[list(p) for p in key], presence_sha256=array_digest(presence),
                      valid_sha256=array_digest(valid_cpu), queries={})
        return key, dst, src, valid_cpu, presence, record

    @staticmethod
    def _plane(event_prob, query, src):
        plane = _numpy(event_prob[(int(query),) + src])
        if not np.isfinite(plane).all() or ((plane < 0) | (plane > 1)).any():
            raise ValueError('Invalid raw query probability')
        return plane

    def observe_first(self, *, window, event_prob, event_presence, entity_prob, valid, weight):
        self._require('first')
        key, _, src, _, presence, record = self._event(window, event_prob, event_presence, valid)
        if key in self._keys:
            raise ValueError('Duplicate first-pass window')
        for q in np.flatnonzero(presence > .5):
            record['queries'][int(q)] = array_digest(self._plane(event_prob, q, src))
        self.decoder(window=window, event_prob=event_prob, event_presence=event_presence,
                     entity_prob=entity_prob, valid=valid, weight=weight)
        self._events.append(record)
        self._keys.add(key)

    def decode(self, fused_probability):
        self._require('first')
        fused = np.asarray(fused_probability)
        if fused.dtype != np.float32 or fused.shape != self.shape or not np.isfinite(fused).all():
            raise ValueError('Fused control must be a finite float32 volume on the same grid')
        if ((fused < 0) | (fused > 1)).any():
            raise ValueError('Fused control is not a probability')
        inst, rows, stats = self.decoder.finish()
        self._phase = 'replay'
        if inst.shape != self.shape or not np.issubdtype(inst.dtype, np.integer) or (inst < 0).any():
            raise ValueError('Invalid decoded instance map')
        if [r['instance'] for r in rows] != list(range(1, len(rows) + 1)) or inst.max(initial=0) > len(rows):
            raise ValueError('Decoded child IDs are not continuous')
        self._inst, self._returned_rows, self._rows = inst, rows, deepcopy(rows)
        counts = np.bincount(inst.ravel(), minlength=len(rows) + 1)
        sums = np.bincount(inst.ravel(), weights=fused.ravel(), minlength=len(rows) + 1)
        candidates = sorted(self.decoder._candidates, key=lambda c: c.priority)
        self._child_parent = np.zeros(len(rows) + 1, np.int64)
        expected_events = {tuple(tuple(p) for p in r['window']): r for r in self._events}
        for row in rows:
            k, origin = row['instance'], row['detector_origin']
            rank = origin['rank']
            if not 1 <= rank <= len(candidates):
                raise ValueError('Unknown original parent rank')
            candidate = candidates[rank - 1]
            key = candidate.window
            if (counts[k] != row['n_voxels'] or counts[k] <= 0
                    or row['score'] != candidate.score or origin['parent_score'] != candidate.score
                    or origin['parent_voxels'] != candidate.n_voxels
                    or origin['retained_fraction'] != row['n_voxels'] / candidate.n_voxels
                    or origin['query'] != candidate.query or origin['component'] != candidate.component
                    or tuple(tuple(p) for p in origin['window']) != key
                    or key not in expected_events or candidate.query not in expected_events[key]['queries']):
                raise ValueError('Child count or parent provenance mismatch')
            self._parents[rank] = candidate
            self._child_parent[k] = rank
            self._children.append(dict(instance=k, parent_score=candidate.score,
                parent_voxels=candidate.n_voxels, n_voxels=row['n_voxels'],
                retained_fraction=origin['retained_fraction'], child_fused_mean=float(sums[k] / counts[k]),
                child_raw_mean=None))
            self._children_by_parent[rank].append(self._children[-1])
        for rank, candidate in self._parents.items():
            selected = self._child_parent[inst[candidate.slices]] == rank
            mask = candidate.mask()
            n = sum(r['n_voxels'] for r in self._children_by_parent[rank])
            if np.count_nonzero(selected) != n or (selected & ~mask).any():
                raise ValueError('Child footprint is not contained in its original parent')
            self._groups[candidate.window][candidate.query].append(rank)
        self._parent_measurements = {}
        self._map_digest = array_digest(inst)
        inst.flags.writeable = False
        return inst, rows, stats

    def observe_replay(self, *, window, event_prob, event_presence, entity_prob, valid, weight):
        self._require('replay')
        key, dst, src, valid_cpu, presence, record = self._event(window, event_prob, event_presence, valid)
        if self._replayed >= len(self._events):
            raise ValueError('Extra or duplicate replay window')
        expected = self._events[self._replayed]
        if any(record[k] != expected[k] for k in ('window', 'presence_sha256', 'valid_sha256')):
            raise ValueError('Replay window, presence or validity changed')
        labels = self._inst[dst]
        counts = np.bincount(labels.ravel(), minlength=len(self._children) + 1)
        for q in np.flatnonzero(presence > .5):
            q = int(q)
            plane = self._plane(event_prob, q, src)
            record['queries'][q] = array_digest(plane)
            if record['queries'][q] != expected['queries'].get(q):
                raise ValueError('Replay raw float32 probability changed')
            ranks = self._groups.get(key, {}).get(q, [])
            if not ranks:
                continue
            sums = np.bincount(labels.ravel(), weights=plane.ravel(), minlength=len(self._children) + 1)
            for rank in ranks:
                candidate = self._parents[rank]
                local = tuple(slice(a - s.start, b - s.start) for (a, b), s in zip(candidate.box, dst))
                p, mask = plane[local], candidate.mask()
                if p.shape != mask.shape or (mask & ~(valid_cpu[local] > 0)).any():
                    raise ValueError('Parent lies outside its replay window/validity')
                parent_sum = float(np.sum(p[mask], dtype=np.float64))
                if parent_sum / candidate.n_voxels != candidate.score:
                    raise ValueError('Original parent mean did not replay exactly')
                selected = self._child_parent[self._inst[candidate.slices]] == rank
                if (selected & ~mask).any():
                    raise ValueError('Child footprint changed')
                removed = mask & ~selected
                retained_n, removed_n = int(selected.sum()), int(removed.sum())
                self._parent_measurements[rank] = dict(rank=rank, parent_score=candidate.score,
                    parent_voxels=candidate.n_voxels, final_children=len(self._children_by_parent[rank]),
                    retained_voxels=retained_n, removed_voxels=removed_n,
                    retained_raw_mean=float(np.sum(p[selected], dtype=np.float64) / retained_n),
                    removed_raw_mean=float(np.sum(p[removed], dtype=np.float64) / removed_n) if removed_n else None)
                for child in self._children_by_parent[rank]:
                    k = child['instance']
                    if counts[k] != child['n_voxels'] or child['child_raw_mean'] is not None:
                        raise ValueError('Child is incomplete or measured more than once')
                    child['child_raw_mean'] = float(sums[k] / counts[k])
        if record['queries'] != expected['queries']:
            raise ValueError('Replay present-query membership changed')
        self._replayed += 1

    def finish_replay(self):
        self._require('replay')
        if self._replayed != len(self._events) or any(r['child_raw_mean'] is None for r in self._children):
            raise ValueError('Replay is missing windows or child measurements')
        if self._returned_rows != self._rows or array_digest(self._inst) != self._map_digest:
            raise ValueError('Original decoded rows/map changed during measurement')
        if self._parent_measurements.keys() != self._parents.keys():
            raise ValueError('Replay is missing parent measurements')
        self._phase = 'finished'
        events = deepcopy(self._events)
        return dict(children=deepcopy(self._children), parents=list(self._parent_measurements.values()),
            window_hashes=events, checks=dict(first_windows=len(events), replay_windows=self._replayed,
                present_query_planes=sum(len(r['queries']) for r in events), raw_probability_planes_exact=True,
                original_parent_means_exact=True, decoded_rows_and_map_unchanged=True,
                window_hashes_sha256=hashlib.sha256(json.dumps(events, sort_keys=True).encode()).hexdigest()))

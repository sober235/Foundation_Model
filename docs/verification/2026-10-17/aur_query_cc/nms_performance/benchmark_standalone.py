"""Compare current decoder finish with scalar revision 4f9fdbf on synthetic masks.

Run from the repository root, or pass --repo explicitly:
  python benchmark_standalone.py --repo /path/to/checkout --output /tmp/nms-rerun.json

No copied source modules, models, raw data, or GPU execution are needed. The
baseline is read from local Git history; the vector implementation is imported
from the current checkout. Output contains only synthetic aggregate measurements.
Timings are single runs on the executing host, not end-to-end validation gains.
"""
import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import types

BASELINE = '4f9fdbf:anatobind/aur/query_decoder.py'
BASELINE_SHA256 = '1af288b3d526bc179b608124ec4f186032494a49beca5c9431a7acb5e5d5c7a2'
SCENARIOS = [('disjoint_solid', 2400), ('clustered_solid', 1200),
             ('clustered_sparse', 1200), ('nested_solid', 1200),
             ('equalbox_sparse', 1000)]


def candidates(kind, n, np, candidate_class):
    """Frozen fixtures: preserve seed, generation order, geometry, and tie scores."""
    rng = np.random.default_rng(20261017)
    items = []
    for i in range(n):
        if kind == 'disjoint_solid':
            start = np.array([i // 225, (i // 15) % 15, i % 15]) * 9
            size = rng.integers(3, 8, size=3)
        elif kind == 'nested_solid':
            size = np.repeat(1 + i % 32, 3)
            start = (34 - size) // 2
        elif kind == 'equalbox_sparse':
            start = np.zeros(3, dtype=int)
            size = np.repeat(12, 3)
        else:
            start = rng.integers(0, 48, size=3)
            size = rng.integers(4, 15, size=3)
        box = tuple((int(a), int(a + b)) for a, b in zip(start, size))
        if kind.endswith('sparse'):
            if kind == 'equalbox_sparse':
                mask = np.zeros(tuple(size), dtype=bool)
                mask[:, :, 0] = True
                for z in range(int(size[0])):
                    mask[z, rng.choice(int(size[1]), 2, replace=False), :] = True
            else:
                mask = rng.random(tuple(size)) < .12
                mask[::3, :, :] = True
                mask[:, 0, :] = True
        else:
            mask = np.ones(tuple(size), dtype=bool)
        window = tuple((int(a // 4 * 4), int(a // 4 * 4 + 64)) for a in start)
        items.append(candidate_class(window, i % 31, i + 1, box,
                                     np.packbits(mask.ravel()), int(mask.sum()),
                                     .9 - (i % 101) * .001))
    rng.shuffle(items)
    shape = tuple(max(c.box[axis][1] for c in items) + 1 for axis in range(3))
    return items, shape


def run(module, items, shape):
    decoder = module.QueryCCDecoder(shape, 10., workers=1)
    decoder._candidates = items
    decoder._raw_components = len(items)
    original_iou = module._mask_iou
    counter = [0]

    def counted(*args):
        counter[0] += 1
        return original_iou(*args)

    module._mask_iou = counted
    try:
        start = time.perf_counter()
        output = decoder.finish()
        elapsed = time.perf_counter() - start
    finally:
        module._mask_iou = original_iou
    return output, decoder.nms_trace, elapsed, counter[0]


def timeout(*_):
    raise TimeoutError('115-second overall CPU benchmark budget exceeded')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, default=Path.cwd(),
                        help='Current checkout containing the local baseline Git revision')
    parser.add_argument('--output', type=Path, required=True,
                        help='New JSON results file; existing files are never overwritten')
    args = parser.parse_args()
    if args.output.exists():
        parser.error('--output must name a new file to preserve archived measurements')

    os.nice(19)
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
        os.environ[key] = '4'
    signal.signal(signal.SIGALRM, timeout)
    signal.alarm(115)
    try:
        import numpy as np
        import torch
        torch.set_num_threads(4)
        torch.set_num_interop_threads(1)

        repo = args.repo.resolve()
        source = subprocess.check_output(['git', '-C', str(repo), 'show', BASELINE], timeout=10)
        if hashlib.sha256(source).hexdigest() != BASELINE_SHA256:
            raise RuntimeError('Baseline source differs from the frozen benchmark revision')
        sys.path.insert(0, str(repo))
        original = types.ModuleType('_nms_scalar_baseline_4f9fdbf')
        sys.modules[original.__name__] = original  # Required by dataclass module lookup.
        exec(compile(source, BASELINE, 'exec'), original.__dict__)
        vector = importlib.import_module('anatobind.aur.query_decoder')
        expected = repo / 'anatobind/aur/query_decoder.py'
        if Path(vector.__file__).resolve() != expected:
            raise RuntimeError('Decoder import did not resolve to the requested checkout')
        print(json.dumps(dict(seed=20261017, budget_seconds=115,
            baseline_sha256=BASELINE_SHA256,
            checkout_decoder_sha256=hashlib.sha256(expected.read_bytes()).hexdigest())),
            file=sys.stderr, flush=True)

        records = []
        for kind, n in SCENARIOS:
            items, shape = candidates(kind, n, np, original._Candidate)
            baseline, trace, slow_time, slow_calls = run(original, items, shape)
            proposed, new_trace, fast_time, fast_calls = run(vector, items, shape)
            np.testing.assert_array_equal(baseline[0], proposed[0])
            if baseline[1:] != proposed[1:] or trace != new_trace:
                raise AssertionError('Finish rows, stats, or suppression traces differ')
            row = dict(kind=kind, candidates=n, survivors=n-len(trace),
                       final_instances=baseline[2]['final_instances'],
                       baseline_seconds=slow_time, vector_seconds=fast_time,
                       speedup=slow_time/fast_time, baseline_pair_calls=slow_calls,
                       vector_pair_calls=fast_calls, equal_map_rows_stats_trace=True)
            records.append(row)
            print(json.dumps(row), flush=True)
        with args.output.open('x') as output:
            output.write(json.dumps(records, indent=2) + '\n')
    finally:
        signal.alarm(0)


if __name__ == '__main__':
    main()

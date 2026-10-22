"""Finite, private TRAIN capability pilot using an explicitly inherited runtime.

Run with PYTHONPATH pointing at the registered frozen runtime. This external
controller deliberately does not put its checkout on the runtime import path.
No checkpoint here is a formal supervised completion artifact.
"""
import argparse
from copy import deepcopy
import ctypes
import gc
import hashlib
import importlib.util
import json
import multiprocessing
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

MODES = ('2d', '3d')
FAILED = {'status': 'FAILED', 'evidence': 'NOT_EVIDENCE'}
TOTAL_SECONDS = 10800
PHASE_LIMITS = {'scan': 1800, 'II': 3600, 'III': 3600}


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def _private_path(value):
    """Normalize once; permit only private home destinations outside any Git tree."""
    try:
        path = Path(value).expanduser().resolve()
        if not path.is_relative_to(Path.home().resolve()):
            raise ValueError
        if any((parent / '.git').exists() for parent in (path, *path.parents)):
            raise ValueError
        return path
    except Exception:
        raise ValueError('invalid private destination') from None


def _json(path, value):
    """Exclusive writes prevent evidence replacement, including failure records."""
    with os.fdopen(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), 'w') as stream:
        json.dump(value, stream, sort_keys=True, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())


def _read(path):
    return json.loads(Path(path).read_text())


def make_stage_configs(parent_cfg):
    """Resolve both complete recipes without changing the registered parent's data recipe."""
    result = {}
    for stage, rates in [('II', {'backbone': 5e-5, 'heads': 5e-4}),
                         ('III', {'backbone': 1e-5, 'heads': 2.5e-4, 'relation': 5e-4})]:
        cfg = deepcopy(parent_cfg)
        cfg.pop('ssl_2d_lr_scale', None)
        cfg.update(purpose='resource_probe', steps=512, lr=rates, warmup_steps=32,
                   backbone_warm_steps=0, backbone_warm_scale=1,
                   validation_every=128, checkpoint_every=64, max_seconds=3600)
        result[stage] = cfg
    return result


def _metrics():
    spec = importlib.util.spec_from_file_location('aur_capability_metrics_external',
                                                 Path(__file__).with_name('aur_capability_metrics.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _state_digest(value):
    """Canonical complete checkpoint commitment, independent of torch zip bytes."""
    import torch
    digest = hashlib.sha256()
    def add(item):
        if isinstance(item, torch.Tensor):
            tensor = item.detach().cpu().contiguous()
            digest.update(str(tensor.dtype).encode())
            digest.update(str(tuple(tensor.shape)).encode())
            if tensor.numel():
                digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(item, dict):
            digest.update(b'{')
            for key in sorted(item, key=lambda k: (type(k).__name__, str(k))):
                add(key)
                add(item[key])
            digest.update(b'}')
        elif isinstance(item, (tuple, list)):
            digest.update((type(item).__name__ + ':' + str(len(item)) + '[').encode())
            for member in item:
                add(member)
            digest.update(b']')
        else:
            digest.update((type(item).__name__ + ':' + repr(item) + '\0').encode())
    add(value)
    return digest.hexdigest()


def _portable_batch(value):
    """Keep metadata numeric and safe to reload without unrestricted pickle."""
    import numpy as np
    import torch
    if isinstance(value, np.ndarray):
        return torch.from_numpy(value.copy())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: _portable_batch(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(_portable_batch(item) for item in value)
    return value


def _measure(runner, panels):
    import torch
    from anatobind.aur.unified_run import _restore_rng
    helper, rng = _metrics(), runner._rng()
    training = {name: module.training for name, module in runner.model.named_modules()}
    runner.model.eval()
    try:
        torch.manual_seed(runner.cfg['seed'] + 19901)
        if runner.device.type == 'cuda':
            torch.cuda.manual_seed_all(runner.cfg['seed'] + 19901)
        with torch.no_grad():
            return {mode: [helper.measure_batch(runner.model, runner._batch(deepcopy(batch), mode))
                           for batch in panels[mode]] for mode in MODES}
    finally:
        _restore_rng(runner, rng)
        for name, module in runner.model.named_modules():
            module.training = training[name]


def _train_stage(stage, cfg, identity, initializer, gate, panels, out, device,
                 *, boundaries=(0, 128, 256, 512), on_measure=None):
    """Functional stage executor; smaller explicit budgets are library-test only."""
    import torch
    from anatobind.aur.unified_run import UnifiedRun
    runner = UnifiedRun(stage, cfg, identity, device, initializer=initializer, g1_report=gate)
    stage_dir = Path(out) / stage
    stage_dir.mkdir(mode=0o700)
    records, recovery = {}, None
    def measure():
        records[runner.step_index] = _measure(runner, panels)
        torch.save(records[runner.step_index], stage_dir / f'measure_{runner.step_index:04d}.pt')
        os.chmod(stage_dir / f'measure_{runner.step_index:04d}.pt', 0o600)
        if on_measure is not None:
            on_measure(stage, runner.step_index, records[runner.step_index])
    def factory(mode, draw):
        return deepcopy(panels[mode][draw % len(panels[mode])])
    measure()
    if device.startswith('cuda'):
        torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    with os.fdopen(os.open(stage_dir / 'steps.jsonl', os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), 'w') as logs:
        while runner.step_index < cfg['steps']:
            log = runner.step(factory)
            log['elapsed_seconds'] = time.monotonic() - started
            logs.write(json.dumps(log, sort_keys=True, allow_nan=False) + '\n')
            logs.flush()
            if runner.step_index % cfg['checkpoint_every'] == 0:
                path = stage_dir / f'checkpoint_{runner.step_index:04d}.pt'
                runner.save(path)
                os.chmod(path, 0o600)
                if recovery is None:
                    before = _state_digest(torch.load(path, map_location='cpu', weights_only=True))
                    step = runner.step_index
                    del runner
                    gc.collect()
                    if device.startswith('cuda'):
                        torch.cuda.empty_cache()
                    runner = UnifiedRun.resume(path, cfg, identity, device)
                    replay = stage_dir / 'recovery_state.pt'
                    runner.save(replay)
                    os.chmod(replay, 0o600)
                    after = _state_digest(torch.load(replay, map_location='cpu', weights_only=True))
                    if before != after:
                        raise ValueError('recovery state mismatch')
                    recovery = {'verified': True, 'step': step, 'state_sha256': before}
                    _json(stage_dir / 'recovery.json', recovery)
            if runner.step_index in boundaries:
                measure()
    resources = {'elapsed_seconds': time.monotonic() - started, 'exposures': runner.exposures,
                 'peak_allocated_bytes': torch.cuda.max_memory_allocated() if device.startswith('cuda') else 0}
    _json(stage_dir / 'resources.json', resources)
    return runner.export(), records, recovery


def _validate_registration(reg):
    expected = {'schema_version', 'runtime_root', 'runtime_sha256', 'inputs', 'controllers', 'stage_configs'}
    if (set(reg) != expected or reg['schema_version'] != 1
            or set(reg['inputs']) != {'data_contract', 'parent_export', 'parent_completion', 'g1_report', 'g1_completion'}
            or set(reg['controllers']) != {'aur_supervised_capability.py', 'aur_capability_metrics.py'}
            or set(reg['stage_configs']) != {'II', 'III'}):
        raise ValueError('invalid registration')
    for item in reg['inputs'].values():
        if set(item) != {'path', 'sha256'} or str(Path(item['path']).expanduser().resolve()) != item['path']:
            raise ValueError('invalid registered input')
        if _sha(item['path']) != item['sha256']:
            raise ValueError('registered input changed')
    for name, expected_sha in reg['controllers'].items():
        if _sha(Path(__file__).with_name(name)) != expected_sha:
            raise ValueError('controller source changed')
    from anatobind.aur.unified_launcher import REPO, source_snapshot
    from anatobind.aur.data_contract import json_digest
    runtime = Path(reg['runtime_root']).resolve()
    if str(runtime) != reg['runtime_root'] or REPO.resolve() != runtime:
        raise ValueError('imported runtime differs from registration')
    snapshot = source_snapshot()
    if json_digest(snapshot) != reg['runtime_sha256']:
        raise ValueError('runtime source changed')
    return snapshot


def _parent_certificate_matches(parent, completion, export_sha):
    certificate = parent.get('completion_certificate')
    return (isinstance(certificate, dict) and bool(certificate)
            and set(completion) == set(certificate) | {'selected_export_sha256', 'formal_gate_certificate'}
            and completion.get('formal_gate_certificate') is True
            and completion.get('selected_export_sha256') == export_sha
            and all(completion[key] == value for key, value in certificate.items()))


def _g1_parent_matches(gate, certificate, export_sha):
    # The G1 API accepts either a complete format2 checkpoint or its export.
    return (gate.get('checkpoint_step') == certificate['selected_step']
            and gate.get('checkpoint_sha256') in (export_sha, certificate['selected_checkpoint_sha256']))


def _validate_parent_protocol(cfg, parent):
    if (cfg['model'] != parent['model_config'] or cfg['model'].get('planar_merge', False)
            or cfg.get('ssl_2d_lr_scale') != 5 or cfg['points'] != 16384
            or cfg['batch_size'] != 1 or cfg['accumulation_steps'] != 1
            or cfg['crops'] != {'2d': [1, 160, 160], '3d': [128, 160, 160]}):
        raise ValueError('parent is not the frozen LR5 protocol')


def _admit(reg):
    import torch
    from anatobind.aur.data_contract import read_contract
    from anatobind.aur.unified_artifacts import verify_completion_certificate
    from anatobind.aur.unified_run import validate_run_config
    from anatobind.aur.unified_train import _state_hash
    from anatobind.aur.ssl.unified_eval import validate_gate
    snapshot = _validate_registration(reg)
    paths = {key: item['path'] for key, item in reg['inputs'].items()}
    # The normal live TRAIN verifier traverses all committed source/data files.
    frozen = read_contract(paths['data_contract'], verify=True)
    parent = torch.load(paths['parent_export'], map_location='cpu', weights_only=True)
    identity = deepcopy(parent['data_contract'])
    if identity.get('contract_sha256') != frozen['contract_sha256'] or identity.get('runtime_sha256') != reg['runtime_sha256']:
        raise ValueError('parent data or runtime identity differs')
    completion = _read(paths['parent_completion'])
    certificate = parent.get('completion_certificate', {})
    if (not _parent_certificate_matches(parent, completion, reg['inputs']['parent_export']['sha256'])
            or not verify_completion_certificate(parent, data_contract=identity)):
        raise ValueError('parent completion invalid')
    parent_cfg = torch.load(certificate['final_checkpoint_path'], map_location='cpu', weights_only=True)['cfg']
    expected_cfgs = make_stage_configs(parent_cfg)
    if reg['stage_configs'] != expected_cfgs:
        raise ValueError('registered recipes differ from inherited protocol')
    _validate_parent_protocol(parent_cfg, parent)
    for cfg in reg['stage_configs'].values():
        if validate_run_config(cfg) != cfg:
            raise ValueError('incomplete registered recipe')
    gate, complete = _read(paths['g1_report']), _read(paths['g1_completion'])
    if (not validate_gate(gate, data_contract_sha256=frozen['contract_sha256'],
                          backbone_state_sha256=_state_hash(parent['backbone_state_dict']), require_formal=True)
            or complete.get('report_sha256') != reg['inputs']['g1_report']['sha256']
            or complete.get('pass') is not True or complete.get('frozen_backbones_unchanged') is not True
            or not _g1_parent_matches(gate, certificate, reg['inputs']['parent_export']['sha256'])):
        raise ValueError('parent G1 invalid')
    rows = frozen['train_supervised']
    if not rows or any(row.get('role') != 'train_supervised' or row.get('split') != 'train' for row in rows):
        raise ValueError('TRAIN role invalid')
    return snapshot, frozen, parent, identity, gate


def _freeze_panels(reg, frozen, out):
    import torch
    from anatobind.aur.unified_views import QualifiedViews
    helper = _metrics()
    cfg = reg['stage_configs']['II']
    rows = frozen['train_supervised']
    views = QualifiedViews(rows, 'II', cfg, augment=False, validation=False)
    pool = Path(out) / 'pool'
    pool.mkdir(mode=0o700)
    panels, summary, commitments = {}, {}, {}
    for mode in MODES:
        supports, commitments[mode] = [], []
        for draw in range(64):
            batch = _portable_batch(views(mode, draw))
            supports.append(helper.batch_support(batch))
            indices = views.sampler.indices(mode, draw, cfg['batch_size'])
            record = {'draw': draw, 'row_commitments': [_digest(rows[i]) for i in indices],
                      'source_labels': batch['source_labels'], 'tensor_commitment': _state_digest(batch),
                      'component_commitment': _state_digest({key: batch[key] for key in
                          ('instance', 'host', 'host_probs', 'r_eligible', 'r_supervised')})}
            path = pool / f'{mode}_{draw:03d}.pt'
            torch.save(batch, path)
            os.chmod(path, 0o600)
            record['file_sha256'] = _sha(path)
            commitments[mode].append(record)
        selected = helper.select_panel(supports, max_observations=16)
        summary[mode] = helper.panel_support([supports[i] for i in selected])
        if summary[mode].get('status') == 'PASS':
            panels[mode] = [_load_frozen_batch(pool / f'{mode}_{i:03d}.pt', commitments[mode][i]) for i in selected]
        commitments[mode] = {'draws': commitments[mode], 'selected': selected}
    _json(Path(out) / 'panel_commitments.json', commitments)
    _json(Path(out) / 'panel_support.json', summary)
    if any(summary[mode].get('status') != 'PASS' for mode in MODES):
        return None, summary
    return panels, summary


def _load_frozen_batch(path, commitment):
    import torch
    if _sha(path) != commitment['file_sha256']:
        raise ValueError('frozen observation file changed')
    batch = torch.load(path, map_location='cpu', weights_only=True)
    if _state_digest(batch) != commitment['tensor_commitment']:
        raise ValueError('frozen observation tensors changed')
    return batch


def _gpu_admission(device):
    """Check the already resolved physical GPU immediately before model use."""
    if device == 'cpu':
        return {'device': 'cpu'}
    visible = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    if not visible.startswith('GPU-') or ',' in visible or device not in ('cuda', 'cuda:0'):
        raise ValueError('GPU visibility must be one resolved physical UUID')
    try:
        query = subprocess.run(['nvidia-smi', '--query-gpu=uuid,utilization.gpu,memory.used,memory.total',
                                '--format=csv,noheader,nounits'], capture_output=True, text=True,
                               check=True, timeout=10)
        rows = [line.split(',') for line in query.stdout.strip().splitlines()]
        selected = [row for row in rows if row[0].strip() == visible]
        workloads = subprocess.run(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid',
                                    '--format=csv,noheader,nounits'], capture_output=True, text=True,
                                   check=True, timeout=10)
        if len(selected) != 1:
            raise ValueError
        uuid, utilization, used, total = [cell.strip() for cell in selected[0]]
        active = sum(line.split(',')[0].strip() == uuid for line in workloads.stdout.strip().splitlines())
        if active or float(utilization) != 0 or int(total) - int(used) < 8192:
            raise ValueError
        return {'device': 'cuda:0', 'physical_uuid': uuid, 'active_compute_workloads': active,
                'utilization_percent': float(utilization), 'memory_used_mib': int(used),
                'memory_total_mib': int(total)}
    except Exception:
        raise ValueError('GPU is unavailable or no longer idle') from None


def _execute(registration, out, device, send):
    import torch
    torch.set_num_threads(2)
    reg = _read(registration)
    reg_sha = _sha(registration)
    snapshot, frozen, parent, identity, gate = _admit(reg)
    _json(out / 'registration.json', reg)
    _json(out / 'input_commitments.json', {'registration_sha256': reg_sha,
          'runtime_sha256': reg['runtime_sha256'], 'inputs': reg['inputs']})
    _json(out / 'runtime_sources.json', snapshot)
    archive = out / 'controller_sources'
    archive.mkdir(mode=0o700)
    for name, digest in reg['controllers'].items():
        path = archive / name
        with os.fdopen(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), 'wb') as stream:
            stream.write(Path(__file__).with_name(name).read_bytes())
        if _sha(path) != digest:
            raise ValueError('source archive changed')
    send({'phase': 'scan'})
    panels, support = _freeze_panels(reg, frozen, out)
    if panels is None:
        return {'status': 'INSUFFICIENT', 'evidence': 'NOT_EVIDENCE'}
    panel_commitments_sha = _sha(out / 'panel_commitments.json')
    _json(out / 'gpu_admission.json', _gpu_admission(device))
    helper, heads, retention_baseline, aggregates, retained = _metrics(), {}, {}, {}, {}
    def boundary(stage, step, records):
        aggregates.setdefault(stage, {})[step] = {}
        retained.setdefault(stage, {})[step] = {}
        for mode in MODES:
            if stage == 'II' and step == 0:
                heads[mode] = helper.fit_retention(records[mode])
                if heads[mode]['status'] != 'PASS':
                    raise _Insufficient()
            aggregates[stage][step][mode] = helper.aggregate_measurements(records[mode])
            retained[stage][step][mode] = helper.score_retention(records[mode], heads[mode])
            if retained[stage][step][mode]['status'] != 'PASS':
                raise _Insufficient()
            if stage == 'II' and step == 0:
                retention_baseline[mode] = retained[stage][step][mode]
    send({'phase': 'II'})
    second, _, recovery_ii = _train_stage('II', reg['stage_configs']['II'], identity, parent, gate,
                                         panels, out, device, on_measure=boundary)
    send({'phase': 'III'})
    final, _, recovery_iii = _train_stage('III', reg['stage_configs']['III'], identity, second, None,
                                         panels, out, device, on_measure=boundary)
    # Reverify full live data traversal, parents, controller and imported runtime.
    send({'phase': 'close'})
    closing = _admit(reg)
    if closing[0] != snapshot or closing[1]['contract_sha256'] != frozen['contract_sha256'] or _sha(registration) != reg_sha:
        raise ValueError('closing input integrity changed')
    if any(_sha(archive / name) != digest for name, digest in reg['controllers'].items()):
        raise ValueError('controller archive changed')
    if _read(out / 'runtime_sources.json') != snapshot or _read(out / 'registration.json') != reg:
        raise ValueError('archived registration or runtime commitment changed')
    if _sha(out / 'panel_commitments.json') != panel_commitments_sha:
        raise ValueError('frozen observation commitments changed')
    commitments = _read(out / 'panel_commitments.json')
    for mode in MODES:
        for draw, commitment in enumerate(commitments[mode]['draws']):
            if _sha(out / 'pool' / f'{mode}_{draw:03d}.pt') != commitment['file_sha256']:
                raise ValueError('frozen observation archive changed')
    verdicts = {mode: helper.capability_verdict(aggregates['II'][0][mode], aggregates['II'][512][mode],
                {'initial': aggregates['III'][0][mode], 'final': aggregates['III'][512][mode]},
                retention_baseline[mode], retained['III'][512][mode]) for mode in MODES}
    _json(out / 'aggregates.json', {'metrics': aggregates, 'retention': retained, 'verdicts': verdicts})
    torch.save(heads, out / 'retention_heads.pt')
    os.chmod(out / 'retention_heads.pt', 0o600)
    status = 'INSUFFICIENT' if any(v['status'] == 'INSUFFICIENT' for v in verdicts.values()) else 'COMPLETE'
    result = {'status': status, 'gate': 'PASS' if all(v['status'] == 'PASS' for v in verdicts.values()) else 'FAIL',
              'evidence': 'NOT_EVIDENCE', 'steps': {'II': second['step'], 'III': final['step']},
              'recovery_verified': recovery_ii['verified'] and recovery_iii['verified']}
    _json(out / 'result.json', result)
    return result


class _Insufficient(Exception):
    pass


def _supervise(work, *, total_seconds=TOTAL_SECONDS, phase_limits=None):
    """External watchdog kills blocked workers and verifies exit from this boundary."""
    context = multiprocessing.get_context('fork')
    receiver, sender = context.Pipe(duplex=False)
    parent_pid = os.getpid()
    def child():
        try:
            if ctypes.CDLL(None, use_errno=True).prctl(1, signal.SIGKILL, 0, 0, 0) != 0:
                raise RuntimeError()
            if os.getppid() != parent_pid:
                os._exit(1)
            os.umask(0o077)
            current_nice = os.getpriority(os.PRIO_PROCESS, 0)
            if current_nice < 19:
                os.nice(19 - current_nice)
            with open(os.devnull, 'w') as sink:
                os.dup2(sink.fileno(), 1)
                os.dup2(sink.fileno(), 2)
            try:
                result = work(sender.send)
            except _Insufficient:
                result = {'status': 'INSUFFICIENT', 'evidence': 'NOT_EVIDENCE'}
            except BaseException:
                result = dict(FAILED)
            sender.send({'result': result})
        finally:
            sender.close()
    worker = context.Process(target=child)
    start = time.monotonic()
    worker.start()
    sender.close()
    deadline, phase_deadline, result = start + total_seconds, float('inf'), None
    limits = PHASE_LIMITS if phase_limits is None else phase_limits
    try:
        while worker.is_alive():
            now = time.monotonic()
            if now >= min(deadline, phase_deadline):
                result = {'status': 'TIMEOUT', 'evidence': 'NOT_EVIDENCE'}
                worker.kill()
                break
            if receiver.poll(min(.1, max(0, min(deadline, phase_deadline) - now))):
                try:
                    message = receiver.recv()
                except EOFError:
                    break
                if 'phase' in message:
                    phase_deadline = time.monotonic() + limits.get(message['phase'], total_seconds)
                if 'result' in message:
                    result = message['result']
        worker.join(timeout=2)
        if worker.is_alive():
            worker.kill()
            worker.join(timeout=2)
        if worker.is_alive():
            return dict(FAILED)
        while receiver.poll():
            try:
                message = receiver.recv()
            except EOFError:
                break
            if 'result' in message:
                result = message['result']
        return result if result is not None and worker.exitcode == 0 else (
            result if result and result['status'] == 'TIMEOUT' else dict(FAILED))
    finally:
        if worker.is_alive():
            worker.kill()
            worker.join(timeout=2)
        receiver.close()


def run_capability(registration_path, out, device='cpu'):
    """Start a fresh finite job; failures never resume or replace an existing job."""
    owned = False
    try:
        out = _private_path(out)
        if out.exists() or out.is_symlink():
            return dict(FAILED)
        # Registration may include private identifiers: no argument or exception is printed.
        registration = Path(registration_path).expanduser().resolve()
        if device != 'cpu' and device != 'cuda' and not (device.startswith('cuda:') and device[5:].isdigit()):
            return dict(FAILED)
        out.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        out.mkdir(mode=0o700)
        owned = True
        result = _supervise(lambda send: _execute(registration, out, device, send))
        _json(out / 'terminal.json', result)
        return result
    except BaseException:
        if owned:
            try:
                _json(out / 'terminal.json', FAILED)
            except Exception:
                pass
        return dict(FAILED)


class _PrivateParser(argparse.ArgumentParser):
    def error(self, message):
        self.exit(2, 'invalid capability arguments\n')


def main(argv=None):
    parser = _PrivateParser(description='Bounded TRAIN capability check (NOT_EVIDENCE).')
    parser.add_argument('--registration', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args(argv)
    result = run_capability(args.registration, args.out, args.device)
    print(json.dumps(result, sort_keys=True))
    return 0 if result['status'] in ('COMPLETE', 'INSUFFICIENT') else 1


if __name__ == '__main__':
    raise SystemExit(main())

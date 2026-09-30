"""Server preflight and four phase probes of the archived QPAD model.

Usage: python rebuttal/run.py check | diagnose [--config PATH] [--device cpu|cuda:0]
The A/B/C revision suite is provided by experiments.py.
"""
import argparse
import ast
import csv
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = Path(__file__).parent / 'configs/original_phase_probe.json'
PARAMETER_NAMES = {'lambda1', 'lambda2', 'eta1', 'eta2', 'eta3', 'psi'}


def path_from_root(value):
    p = Path(os.path.expandvars(str(value))).expanduser()
    return (p if p.is_absolute() else ROOT / p).resolve()


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_output(*args):
    try:
        return subprocess.check_output(['git', '-C', str(ROOT), *args],
                                       text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def resolve_config(args):
    config_path = args.config.expanduser().resolve()
    config = json.loads(config_path.read_text())
    for key in ['signal_dir', 'anomaly_dir', 'model_source', 'output_root', 'device', 'epochs']:
        value = getattr(args, key, None)
        if value is not None:
            config[key] = str(value) if isinstance(value, Path) else value
    if config['experiment'] != 'original_phase_probe':
        raise ValueError('Only original_phase_probe is implemented in this entry point.')
    for key in ['epochs', 'cpu_threads']:
        if type(config[key]) is not int or config[key] < 1:
            raise ValueError(f'{key} must be a positive integer.')
    if type(config['seed']) is not int or not 0 <= config['seed'] < 2**32:
        raise ValueError('seed must be an integer in [0, 2**32).')
    if set(config['parameters']) != PARAMETER_NAMES:
        raise ValueError('parameters must contain exactly lambda1, lambda2, eta1, eta2, eta3, psi.')
    for key, value in config['parameters'].items():
        if not math.isfinite(value) or value < 0:
            raise ValueError(f'{key} must be finite and nonnegative.')
    for key in ['learning_rate', 'loss_scale']:
        if not math.isfinite(config[key]) or config[key] <= 0:
            raise ValueError(f'{key} must be finite and positive.')
    names = config['signals']
    if not names or len(names) != len(set(names)) or any(not re.fullmatch(r'[A-Za-z0-9_\-]+', n) for n in names):
        raise ValueError('signals must contain unique file stems consisting of letters, digits, _ or -.')
    shifts = config['phase_shifts']
    if not shifts or len(set(shifts)) != len(shifts) or any(not math.isfinite(v) for v in shifts):
        raise ValueError('phase_shifts must contain distinct finite numbers.')
    candidates = {
        'signal_dir': ['data/simulation/synthetic_signal', 'synthetic_signal'],
        'anomaly_dir': ['data/simulation/synthetic_gtanomaly', 'synthetic_gtanomaly'],
        'model_source': ['scripts/sml2_decompose.py', 'sml2_decompose.py'],
    }
    for key, choices in candidates.items():
        if config[key] == 'auto':
            found = [ROOT / p for p in choices if (ROOT / p).exists()]
            if not found:
                raise FileNotFoundError(f'{key}: none of {choices} found; set --{key.replace("_", "-")}.')
            config[key] = str(found[0].resolve())
        else:
            config[key] = str(path_from_root(config[key]))
    config['output_root'] = str(path_from_root(config['output_root']))
    return config, config_path


def load_definitions(path, np, torch):
    tree = ast.parse(path.read_text())
    required = {'Periodic', 'extract_period_segments', 'JointOptimizer'}
    nodes = [n for n in tree.body if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name in required]
    if len(nodes) != 3:
        raise ValueError(f'{path}: expected three original model definitions.')
    namespace = {'np': np, 'torch': torch, 'nn': torch.nn, 'optim': torch.optim}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), namespace)
    return namespace


def preflight(config, config_path):
    import numpy as np
    import torch
    requested = config['device']
    if requested == 'auto':
        requested = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    if requested != 'cpu' and not re.fullmatch(r'cuda:\d+', requested):
        raise ValueError('device must be auto, cpu, or cuda:N.')
    device = torch.device(requested)
    if device.type == 'cuda':
        if not torch.cuda.is_available() or device.index >= torch.cuda.device_count():
            raise RuntimeError(f'Requested {device}, but that CUDA device is unavailable.')
        torch.empty(1, device=device)  # Check that allocation actually works.
    config['device'] = str(device)
    source = Path(config['model_source'])
    definitions = load_definitions(source, np, torch)
    inputs = []
    for name in config['signals']:
        xp = Path(config['signal_dir']) / f'{name}.npy'
        ap = Path(config['anomaly_dir']) / f'{name}_A.npy'
        x, a = np.load(xp, allow_pickle=False), np.load(ap, allow_pickle=False)
        if x.shape != (2000,) or a.shape != x.shape:
            raise ValueError(f'{name}: this archived-model probe requires X and A shapes (2000,), got {x.shape}, {a.shape}.')
        if not np.isfinite(x).all() or not np.isfinite(a).all():
            raise ValueError(f'{name}: input contains nonfinite values.')
        if np.iscomplexobj(x) or np.iscomplexobj(a):
            raise ValueError(f'{name}: real-valued inputs required.')
        inputs.append({'signal': name, 'length': len(x), 'x_path': str(xp), 'a_path': str(ap),
                       'x_sha256': sha256(xp), 'a_sha256': sha256(ap)})
    versions = {}
    for package in ['numpy', 'torch', 'scipy', 'scikit-learn', 'matplotlib']:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    manifest = {
        'stage': 'archived_model_phase_diagnostic',
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'python': sys.version, 'executable': sys.executable, 'platform': platform.platform(),
        'packages': versions, 'torch_cuda_build': torch.version.cuda,
        'device': str(device),
        'device_name': torch.cuda.get_device_name(device) if device.type == 'cuda' else platform.processor() or 'CPU',
        'git_commit': git_output('rev-parse', 'HEAD'),
        'relevant_git_status': git_output('status', '--porcelain', '--untracked-files=all', '--',
                                          'rebuttal', 'scripts/sml2_decompose.py', 'sml2_decompose.py'),
        'source_path': str(source), 'source_sha256': sha256(source),
        'runner_sha256': sha256(Path(__file__)),
        'config_path': str(config_path), 'config_file_sha256': sha256(config_path),
        'resolved_config': config, 'inputs': inputs,
        'number_of_fits': len(config['signals']) * len(config['phase_shifts']),
        'timing_scope': 'optimizer loop including trace collection; CUDA synchronized at both ends',
        'stopping_rule': 'fixed iteration budget; no convergence tolerance',
    }
    return np, torch, definitions, device, manifest


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def fit_one(config, name, shift, definitions, device, np, torch, output):
    x = np.load(Path(config['signal_dir']) / f'{name}.npy', allow_pickle=False)
    truth = np.load(Path(config['anomaly_dir']) / f'{name}_A.npy', allow_pickle=False)
    # Bind the original loss to an explicitly selected device. The legacy
    # constructor automatically picks CUDA even when a CPU run is requested.
    opt = definitions['JointOptimizer'].__new__(definitions['JointOptimizer'])
    opt.device = device
    opt.periodic_model = definitions['Periodic'](K_init=10, signal_length=len(x)).to(device)
    opt.X = torch.tensor(x, dtype=torch.float32, device=device)
    for key, value in config['parameters'].items():
        setattr(opt, key, value)
    model = opt.periodic_model
    with torch.no_grad():
        model.t_k.add_(shift)
    initial_t = model.t_k.detach().clone()
    initial_T = model.T_active.detach().clone()
    adam = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    trace = []
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    for step in range(config['epochs']):
        adam.zero_grad()
        objective = opt.compute_loss()
        if not torch.isfinite(objective):
            raise FloatingPointError(f'Nonfinite loss at step {step}.')
        (objective * config['loss_scale']).backward()
        adam.step()
        if step % 50 == 0:
            trace.append({'step': step, 'objective': float(objective.detach().cpu())})
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    seconds = time.perf_counter() - started
    with torch.no_grad():
        P, A = model()
        final = float(opt.compute_loss().cpu())
    p, a = P.cpu().numpy(), A.cpu().numpy()
    if not math.isfinite(final) or not np.isfinite(p).all() or not np.isfinite(a).all():
        raise FloatingPointError('Nonfinite final objective or components.')
    trace.append({'step': config['epochs'], 'objective': final})
    prefix = f'{name}_phase{shift:g}'
    np.savez_compressed(output / f'{prefix}.npz', P=p, A=a,
                        initial_t=initial_t.cpu().numpy(), final_t=model.t_k.detach().cpu().numpy(),
                        initial_T=initial_T.cpu().numpy(), final_T=model.T_active.detach().cpu().numpy())
    with (output / f'{prefix}_trace.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=['step', 'objective'])
        writer.writeheader()
        writer.writerows(trace)
    return {'signal': name, 'phase_shift_samples': shift, 'status': 'completed_fixed_budget',
            'iterations': config['epochs'], 'final_objective': final,
            'a_rmse_to_saved_ground_truth': float(np.sqrt(np.mean((a.astype(float) - truth)**2))),
            'max_onset_movement_samples': float((model.t_k.detach() - initial_t).abs().max().cpu()),
            'max_duration_movement_samples': float((model.T_active.detach() - initial_T).abs().max().cpu()),
            'seconds': seconds}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['check', 'diagnose'])
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    for key in ['signal-dir', 'anomaly-dir', 'model-source', 'output-root', 'device']:
        parser.add_argument('--' + key)
    parser.add_argument('--epochs', type=int)
    args = parser.parse_args()
    try:
        config, config_path = resolve_config(args)
        np, torch, definitions, device, manifest = preflight(config, config_path)
    except Exception as error:
        print(f'Preflight failed: {type(error).__name__}: {error}', file=sys.stderr)
        return 2
    if args.command == 'check':
        print(json.dumps({'status': 'ready_for_original_diagnostic', **manifest}, ensure_ascii=False, indent=2))
        return 0
    output = Path(config['output_root']) / ('phase_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ'))
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(config['cpu_threads'])
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(config['seed'])
    write_json(output / 'manifest.json', manifest)
    write_json(output / 'config.json', config)
    results = []
    print(f'Output: {output}', flush=True)
    for name in config['signals']:
        for shift in config['phase_shifts']:
            try:
                result = fit_one(config, name, shift, definitions, device, np, torch, output)
            except Exception as error:
                result = {'signal': name, 'phase_shift_samples': shift, 'status': 'failed',
                          'error': f'{type(error).__name__}: {error}'}
            results.append(result)
            print(json.dumps(result, ensure_ascii=False, allow_nan=False), flush=True)
            write_json(output / 'summary.json', {'scope': 'Archived-model diagnostic', 'runs': results})
    failures = sum(r['status'] == 'failed' for r in results)
    print(f'Completed: {len(results) - failures}; failed: {failures}; directory: {output}', flush=True)
    return 1 if failures else 0


if __name__ == '__main__':
    raise SystemExit(main())

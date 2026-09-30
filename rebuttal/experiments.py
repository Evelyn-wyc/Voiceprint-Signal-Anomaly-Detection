"""Necessary QPAD revision experiments: check, run/resume, summarize.

Run from the repository root; see rebuttal/README.md for server commands.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import traceback

import numpy as np
from datasets import SCENARIOS, real_cases, synthetic_case

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
DEFAULT_CONFIG = HERE / 'configs/necessary.json'
METHODS = ['QPAD', 'STL', 'VMD', 'QPGP', 'RPCA']


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def array_digest(array):
    a = np.ascontiguousarray(array)
    return hashlib.sha256(str(a.dtype).encode()+str(a.shape).encode()+a.tobytes()).hexdigest()


def write_json(path, data):
    path = Path(path)
    temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    temp.replace(path)


def resolve_path(path):
    p = Path(path).expanduser()
    return (p if p.is_absolute() else ROOT/p).resolve()


def git(*args):
    try:
        return subprocess.check_output(['git', '-C', str(ROOT), *args], text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def code_hashes():
    return {str(p.relative_to(HERE)): digest(p) for p in sorted(HERE.rglob('*.py'))
            if 'outputs' not in p.relative_to(HERE).parts and 'tests' not in p.relative_to(HERE).parts}


def validate_config(c):
    if c['experiment'] != 'necessary_rebuttal_v1':
        raise ValueError('Unknown experiment configuration.')
    if c['methods'] != METHODS:
        raise ValueError(f'This protocol requires methods in this order: {METHODS}')
    sim = c['simulation']
    if len(set(sim['scenarios'])) != len(sim['scenarios']) or not set(sim['scenarios']) <= set(SCENARIOS):
        raise ValueError('Duplicate or unknown simulation scenarios.')
    if not sim['seeds'] or len(set(sim['seeds'])) != len(sim['seeds']) or any(type(s) is not int or s < 0 for s in sim['seeds']):
        raise ValueError('Simulation seeds must be unique nonnegative integers.')
    if type(sim['length']) is not int or type(sim['period']) is not int or sim['period'] < 8 or sim['length'] < 4*sim['period']:
        raise ValueError('Invalid simulation length/period.')
    for kind in ['simulation', 'real']:
        q = c['qpad'][kind]
        if type(q['epochs']) is not int or q['epochs'] < 1 or type(q['K']) is not int or q['K'] < 3:
            raise ValueError('QPAD epochs and K must be positive integers, K>=3.')
        if set(q['parameters']) != {'lambda1', 'lambda2', 'eta1', 'eta2', 'eta3', 'psi'}:
            raise ValueError('Invalid QPAD parameter names.')
        if any(not np.isfinite(v) or v < 0 for v in q['parameters'].values()):
            raise ValueError('QPAD penalties must be finite and nonnegative.')
    ev = c['evaluation']
    if ev['primary_multiplier'] not in ev['multipliers'] or len(set(ev['multipliers'])) != len(ev['multipliers']) or any(not np.isfinite(v) or v <= 0 for v in ev['multipliers']):
        raise ValueError('Invalid predeclared threshold multipliers.')
    if type(ev['bootstrap_repeats']) is not int or ev['bootstrap_repeats'] < 1:
        raise ValueError('Invalid bootstrap repeat count.')
    init = c['initialization']
    if not init['seeds'] or len(set(init['seeds'])) != len(init['seeds']) or any(type(s) is not int or s < 0 for s in init['seeds']):
        raise ValueError('Initialization seeds must be unique nonnegative integers.')
    for k in ['phase_fraction', 'duration_fraction', 'amplitude_fraction']:
        if not np.isfinite(init[k]) or not 0 <= init[k] < 1:
            raise ValueError(f'Invalid initialization perturbation: {k}')
    for key in ['expected_count', 'length']:
        if type(c['real'][key]) is not int or c['real'][key] < 1:
            raise ValueError(f'Invalid real-data {key}.')
    if len(set(init['signals'])) != len(init['signals']):
        raise ValueError('Duplicate initialization signal names.')
    for key in ['cpu_threads']:
        if type(c[key]) is not int or c[key] < 1:
            raise ValueError(f'{key} must be a positive integer.')
    for key in ['learning_rate', 'loss_scale']:
        if not np.isfinite(c['qpad'][key]) or c['qpad'][key] <= 0:
            raise ValueError(f'Invalid {key}.')
    if type(c['qpad']['trace_every']) is not int or c['qpad']['trace_every'] < 1:
        raise ValueError('Invalid trace interval.')
    b = c['baselines']
    if not 0 < b['period_min_fraction'] < b['period_max_fraction'] <= 0.5:
        raise ValueError('Invalid period estimation range.')
    if not 1 <= b['vmd_background_modes'] < b['vmd_modes']:
        raise ValueError('Invalid VMD mode selection.')
    for k in ['qpgp_maxiter', 'rpca_maxiter']:
        if type(b[k]) is not int or b[k] < 1:
            raise ValueError(f'Invalid {k}.')
    for k in ['vmd_alpha', 'vmd_tolerance', 'rpca_tolerance']:
        if not np.isfinite(b[k]) or b[k] <= 0:
            raise ValueError(f'Invalid {k}.')
    if not b['qpgp_period_factors'] or any(not np.isfinite(v) or v <= 0 for v in b['qpgp_period_factors']):
        raise ValueError('Invalid QPGP period candidates.')


def prepare_cases(config, parts, args):
    cases = []
    if 'A' in parts:
        cases += real_cases(resolve_path(args.data_root), config['real']['expected_count'],
                            config['real']['length'], args.real_signal_dir, args.real_label_dir)
        for case in cases:
            case['id'] = 'real_'+case['id']
    if 'B' in parts or 'C' in parts:
        s = config['simulation']
        for scenario in s['scenarios']:
            for seed in s['seeds']:
                name = f'{scenario}_s{seed}'
                if 'B' not in parts and name not in config['initialization']['signals']:
                    continue
                case = {'id': 'sim_'+name, 'dataset': 'simulation', 'scenario': scenario,
                        'seed': seed, 'group': f'seed_{seed}', 'source_paths': [],
                        **synthetic_case(scenario, seed, s['length'], s['period'])}
                cases.append(case)
        if 'C' in parts:
            missing = set(config['initialization']['signals']) - {c['id'][4:] for c in cases if c['dataset'] == 'simulation'}
            if missing:
                raise ValueError(f'C references unavailable simulation cases: {sorted(missing)}')
    return cases


def tasks_for(cases, config, parts):
    tasks = []
    for case in cases:
        default_methods = config['methods'] if case['dataset'] == 'real' or 'B' in parts else ['QPAD']
        for method in default_methods:
            tasks.append({'id': f'{case["id"]}__{method}__default', 'case_id': case['id'],
                          'method': method, 'initialization_seed': None})
        if 'C' in parts and case['id'].removeprefix('sim_') in config['initialization']['signals']:
            for seed in config['initialization']['seeds']:
                tasks.append({'id': f'{case["id"]}__QPAD__init{seed}', 'case_id': case['id'],
                              'method': 'QPAD', 'initialization_seed': seed})
    return tasks


def environment(device):
    import torch
    import scipy, sklearn, statsmodels, vmdpy, threadpoolctl, tqdm  # Fail before creating a run.
    if device == 'auto':
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    if device != 'cpu' and not (device.startswith('cuda:') and device[5:].isdigit()):
        raise ValueError('Device must be cpu, auto, or cuda:N.')
    target = torch.device(device)
    if target.type == 'cuda':
        if not torch.cuda.is_available() or target.index >= torch.cuda.device_count():
            raise RuntimeError(f'Unavailable CUDA device: {device}')
        torch.empty(1, device=target)
    return target, {'python': sys.version, 'executable': sys.executable, 'platform': platform.platform(),
                    'device': str(target), 'device_name': torch.cuda.get_device_name(target) if target.type == 'cuda' else 'CPU',
                    'torch_cuda_build': torch.version.cuda,
                    'packages': {p: importlib.metadata.version(p) for p in
                                 ['numpy', 'torch', 'scipy', 'scikit-learn', 'statsmodels', 'vmdpy', 'threadpoolctl', 'tqdm']}}


def input_record(case):
    return {k: case[k] for k in ['id', 'dataset', 'scenario', 'group', 'seed']} | {
        'length': len(case['X']),
        'arrays': {k: array_digest(v) for k, v in case.items() if isinstance(v, np.ndarray)},
        'sources': [{'path': path, 'sha256': digest(path)} for path in case['source_paths']]}


def make_manifest(config, parts, cases, tasks, env):
    return {'experiment': config['experiment'], 'created_utc': datetime.now(timezone.utc).isoformat(),
            'git_commit': git('rev-parse', 'HEAD'),
            'rebuttal_git_status': git('status', '--porcelain', '--untracked-files=all', '--', 'rebuttal'),
            'code_sha256': code_hashes(), 'config': config, 'parts': sorted(parts),
            'environment': env, 'inputs': [input_record(c) for c in cases], 'tasks': tasks,
            'timing_scope': 'method setup, period estimation and fitting; CUDA synchronized; excludes file I/O and evaluation'}


def check_resume(old, new):
    for key in ['experiment', 'code_sha256', 'config', 'parts', 'inputs', 'tasks']:
        if old[key] != new[key]:
            raise ValueError(f'Resume rejected: {key} differs. Use a new output directory.')
    for key in ['device', 'device_name', 'packages']:
        if old['environment'][key] != new['environment'][key]:
            raise ValueError(f'Resume rejected: environment {key} differs.')


def run_task(task, case, config, device, output):
    import torch
    from baselines import fit_baseline
    from evaluation import evaluate
    from qpad_fit import fit_qpad
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    extra = {}
    if task['method'] == 'QPAD':
        p, a, meta, extra = fit_qpad(case['X'], case['dataset'], config, device, task['initialization_seed'])
    else:
        p, a, meta = fit_baseline(task['method'], case['X'], config['baselines'])
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    seconds = time.perf_counter()-started
    p, a = np.asarray(p, dtype=float), np.asarray(a, dtype=float)
    if p.shape != case['X'].shape or a.shape != p.shape or not np.isfinite(p).all() or not np.isfinite(a).all():
        raise FloatingPointError('Invalid fitted components.')
    metrics, standardized = evaluate(a, case['X'], case['labels'], config['evaluation']['multipliers'])
    npz = output/'fits'/f'{task["id"]}.npz'
    temp = npz.with_suffix('.tmp')
    with temp.open('wb') as stream:
        np.savez_compressed(stream, P=p, A=a, standardized=standardized, **extra)
    temp.replace(npz)
    return {**task, 'status': 'completed', 'dataset': case['dataset'], 'scenario': case['scenario'],
            'group': case['group'], 'seed': case['seed'], 'seconds': seconds,
            'p_rmse': float(np.sqrt(np.mean((p-case['P_true'])**2))) if 'P_true' in case else None,
            'a_rmse': float(np.sqrt(np.mean((a-case['A_true'])**2))) if 'A_true' in case else None,
            'metadata': meta, 'metrics': metrics, 'artifact_sha256': digest(npz)}


def execute(output, manifest, cases, config, device, resume):
    import torch
    from threadpoolctl import threadpool_limits
    from report import validate_result, summarize
    if output.exists():
        if not resume:
            raise FileExistsError(f'{output} exists. Use --resume to continue this exact run.')
        old = json.loads((output/'manifest.json').read_text())
        check_resume(old, manifest)
        for record in old['inputs']:
            path = output/'inputs'/f'{record["id"]}.npz'
            with np.load(path, allow_pickle=False) as data:
                if {k: array_digest(data[k]) for k in data.files} != record['arrays']:
                    raise ValueError(f'Input snapshot changed: {record["id"]}')
    else:
        if resume:
            raise FileNotFoundError(f'Cannot resume absent output: {output}')
        (output/'inputs').mkdir(parents=True)
        (output/'fits').mkdir()
        write_json(output/'manifest.json', manifest)
        write_json(output/'config.json', config)
        for case in cases:
            np.savez_compressed(output/'inputs'/f'{case["id"]}.npz',
                                **{k: v for k, v in case.items() if isinstance(v, np.ndarray)})
    torch.set_num_threads(config['cpu_threads'])
    by_id = {c['id']: c for c in cases}
    failures = 0
    print(f'Output: {output}\nPlanned fits: {len(manifest["tasks"])}', flush=True)
    with threadpool_limits(limits=config['cpu_threads']):
        for index, task in enumerate(manifest['tasks'], 1):
            result_path = output/'fits'/f'{task["id"]}.json'
            if result_path.exists():
                old = json.loads(result_path.read_text())
                if old.get('status') == 'completed':
                    validate_result(output, task, old, config, by_id[task['case_id']]['X'].shape)
                    print(f'[{index}/{len(manifest["tasks"])}] reuse {task["id"]}', flush=True)
                    continue
            print(f'[{index}/{len(manifest["tasks"])}] fit {task["id"]}', flush=True)
            try:
                result = run_task(task, by_id[task['case_id']], config, device, output)
            except Exception as error:
                failures += 1
                result = {**task, 'status': 'failed', 'error': f'{type(error).__name__}: {error}',
                          'traceback': traceback.format_exc()}
                print(f'FAILED {task["id"]}: {result["error"]}', flush=True)
            write_json(result_path, result)
            with (output/'events.jsonl').open('a') as stream:
                stream.write(json.dumps({'utc': datetime.now(timezone.utc).isoformat(),
                                         'task': task['id'], 'status': result['status'],
                                         'error': result.get('error')}, ensure_ascii=False)+'\n')
            write_json(output/'progress.json', {'last_task': task['id'], 'position': index,
                                                'planned': len(manifest['tasks']), 'failures_this_attempt': failures})
    incomplete = summarize(output)
    print(f'Report: {output / "analysis/report.md"}\nIncomplete or failed: {incomplete}', flush=True)
    return 1 if incomplete else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['check', 'run', 'summarize'])
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    parser.add_argument('--data-root', default=str(ROOT.parent/'Voiceprint'))
    parser.add_argument('--real-signal-dir')
    parser.add_argument('--real-label-dir')
    parser.add_argument('--parts', nargs='+', choices=['A', 'B', 'C'], default=['A', 'B', 'C'])
    parser.add_argument('--device', default='auto')
    parser.add_argument('--output', default='rebuttal/outputs/necessary_v1')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    try:
        output = resolve_path(args.output)
        if args.command == 'summarize':
            from report import summarize
            incomplete = summarize(output)
            print(f'Report: {output / "analysis/report.md"}\nIncomplete or failed: {incomplete}')
            return 1 if incomplete else 0
        config = json.loads(args.config.expanduser().resolve().read_text())
        validate_config(config)
        parts = sorted(set(args.parts))
        cases = prepare_cases(config, parts, args)
        tasks = tasks_for(cases, config, parts)
        device, env = environment(args.device)
        manifest = make_manifest(config, parts, cases, tasks, env)
        if args.command == 'check':
            print(json.dumps({'status': 'ready', 'parts': parts, 'device': str(device),
                              'device_name': env['device_name'],
                              'real_inputs': sum(c['dataset']=='real' for c in cases),
                              'simulation_inputs': sum(c['dataset']=='simulation' for c in cases),
                              'fits_by_method': {m: sum(t['method']==m for t in tasks) for m in config['methods']},
                              'total_fits': len(tasks), 'output': str(output),
                              'packages': env['packages']}, ensure_ascii=False, indent=2))
            return 0
        return execute(output, manifest, cases, config, device, args.resume)
    except Exception as error:
        print(f'{args.command} failed: {type(error).__name__}: {error}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())

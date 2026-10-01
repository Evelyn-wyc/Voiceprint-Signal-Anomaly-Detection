"""Paired QPAD initialization comparison across generating periods."""
import argparse
import copy
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import platform
import shutil
import sys
import time
import traceback

import numpy as np
from baselines import estimate_period
from datasets import synthetic_case
from evaluation import evaluate
from experiments import ROOT, HERE, array_digest, digest, git, resolve_path, write_json
from qpad_fit import fit_qpad
from report import close_values, validate_result

DEFAULT_CONFIG = HERE / 'configs/period_prior.json'
VARIANTS = ('fixed', 'input_estimated')
CODE_FILES = ('period_prior.py', 'period_prior_report.py', 'qpad_fit.py',
              'models/simulation.py', 'datasets.py', 'baselines.py',
              'evaluation.py', 'experiments.py', 'report.py')
LEGACY_FITTER_SHA = 'd2e68b84524b7bd389d2dc2aca35f6aba636e811fe8270739687b4112e294361'
ARRAY_KEYS = ('X', 'P_true', 'A_true', 'noise', 'labels')


def code_hashes():
    return {name: digest(HERE/name) for name in CODE_FILES}


def validate_config(c):
    if c['experiment'] != 'period_prior_v1' or c['simulation']['scenario'] != 'nominal':
        raise ValueError('Expected period_prior_v1 with nominal simulations.')
    sim, q = c['simulation'], c['qpad']
    if type(sim['length']) is not int or sim['length'] < 32:
        raise ValueError('Invalid input length.')
    for key in ('periods', 'seeds'):
        if not sim[key] or len(set(sim[key])) != len(sim[key]) or any(type(v) is not int for v in sim[key]):
            raise ValueError(f'{key} must contain unique integers.')
    if min(sim['seeds']) < 0 or min(sim['periods']) < 8 or 4*max(sim['periods']) > sim['length']:
        raise ValueError('Invalid seeds or fewer than four generating cycles.')
    for key in ('learning_rate', 'loss_scale'):
        if not np.isfinite(q[key]) or q[key] <= 0:
            raise ValueError(f'Invalid {key}.')
    for value in (q['trace_every'], q['simulation']['epochs'], c['cpu_threads']):
        if type(value) is not int or value < 1:
            raise ValueError('Trace interval, epochs and CPU threads must be positive integers.')
    for value in (q['simulation']['K'], c['initialization']['minimum_K']):
        if type(value) is not int or value < 2:
            raise ValueError('K must be an integer of at least two.')
    params = q['simulation']['parameters']
    if set(params) != {'lambda1', 'lambda2', 'eta1', 'eta2', 'eta3', 'psi'} or any(not np.isfinite(v) or v < 0 for v in params.values()):
        raise ValueError('Invalid penalty configuration.')
    if c['initialization']['duration_fraction'] != 0.5:
        raise ValueError('This protocol uses an initial half-cycle scale of p_hat/2.')
    b, ev = c['baselines'], c['evaluation']
    if not 0 < b['period_min_fraction'] < b['period_max_fraction'] <= 0.5:
        raise ValueError('Invalid period search range.')
    if ev['primary_multiplier'] not in ev['multipliers'] or len(set(ev['multipliers'])) != len(ev['multipliers']) or any(not np.isfinite(v) or v <= 0 for v in ev['multipliers']):
        raise ValueError('Invalid thresholds.')


def environment(device):
    import torch
    import scipy, sklearn, matplotlib, threadpoolctl, tqdm
    if device == 'auto':
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    if device != 'cpu' and not (device.startswith('cuda:') and device[5:].isdigit()):
        raise ValueError('Use cpu, auto, or cuda:N.')
    target = torch.device(device)
    if target.type == 'cuda':
        if not torch.cuda.is_available() or target.index >= torch.cuda.device_count():
            raise RuntimeError(f'Unavailable device: {device}')
        torch.empty(1, device=target)
    return target, {'python': sys.version, 'platform': platform.platform(), 'device': str(target),
                    'device_name': torch.cuda.get_device_name(target) if target.type == 'cuda' else 'CPU',
                    'torch_cuda_build': torch.version.cuda,
                    'packages': {p: importlib.metadata.version(p) for p in
                                 ('numpy', 'torch', 'scipy', 'scikit-learn', 'matplotlib', 'threadpoolctl', 'tqdm')}}


def choose_initialization(x, variant, config):
    """Only observed X enters the period estimate and initialization rule."""
    if variant == 'fixed':
        return {'K': config['qpad']['simulation']['K'], 'initial_duration': 100.0,
                'period_estimate': None, 'period_estimation': None}
    if variant != 'input_estimated':
        raise ValueError('Unknown initialization variant.')
    p, info = estimate_period(x, config['baselines'])
    k = max(config['initialization']['minimum_K'], int(np.floor(len(x)/p+0.5)))
    return {'K': k, 'initial_duration': p*config['initialization']['duration_fraction'],
            'period_estimate': p, 'period_estimation': info}


def fit_config(config, task):
    result = copy.deepcopy(config)
    result['qpad']['simulation']['K'] = task['K']
    return result


def explicit_state(n, task):
    return {'t_k': np.arange(task['K'], dtype=float)*n/task['K'],
            'T_active': np.full(task['K'], task['initial_duration']),
            'M_independent': np.ones(task['K']), 'A': np.zeros(n)}


def load_reuse(folder, config):
    """Validate source code, configuration, inputs and fits before reuse."""
    if folder is None:
        return {}, None
    manifest = json.loads((folder/'manifest.json').read_text())
    source_config = manifest['config']
    if source_config['experiment'] != 'necessary_rebuttal_v1':
        raise ValueError('Reuse source must be a necessary_v1 run.')
    for key in ('learning_rate', 'loss_scale', 'trace_every', 'simulation'):
        if source_config['qpad'][key] != config['qpad'][key]:
            raise ValueError(f'Reuse rejected: QPAD {key} differs.')
    if source_config['evaluation']['multipliers'] != config['evaluation']['multipliers']:
        raise ValueError('Reuse rejected: threshold configuration differs.')
    if source_config['simulation']['length'] != config['simulation']['length']:
        raise ValueError('Reuse rejected: input length differs.')
    source_hashes = manifest['code_sha256']
    for name in ('models/simulation.py', 'datasets.py', 'evaluation.py', 'baselines.py'):
        if source_hashes.get(name) != digest(HERE/name):
            raise ValueError(f'Reuse rejected: source code differs for {name}.')
    if source_hashes.get('qpad_fit.py') not in (LEGACY_FITTER_SHA, digest(HERE/'qpad_fit.py')):
        raise ValueError('Reuse rejected: unrecognized QPAD fitting code.')
    period = source_config['simulation']['period']
    reuse = {}
    if period not in config['simulation']['periods']:
        raise ValueError('Reuse source has no matching generating period.')
    for seed in config['simulation']['seeds']:
        old_case_id = f'sim_nominal_s{seed}'
        old_task_id = old_case_id+'__QPAD__default'
        old_task = next((t for t in manifest['tasks'] if t['id'] == old_task_id), None)
        if old_task is None:
            continue
        row_path = folder/'fits'/f'{old_task_id}.json'
        if not row_path.is_file():
            continue
        row = json.loads(row_path.read_text())
        if row.get('status') != 'completed':
            continue
        input_path = folder/'inputs'/f'{old_case_id}.npz'
        expected = next(r for r in manifest['inputs'] if r['id'] == old_case_id)
        with np.load(input_path, allow_pickle=False) as z:
            if {k: array_digest(z[k]) for k in z.files} != expected['arrays']:
                raise ValueError('Reuse input snapshot hash mismatch.')
            arrays = {k: z[k].copy() for k in ARRAY_KEYS}
        generated = synthetic_case('nominal', seed, config['simulation']['length'], period)
        for key in ARRAY_KEYS:
            valid = np.array_equal(arrays[key], generated[key]) if key == 'labels' else np.allclose(arrays[key], generated[key], rtol=0, atol=1e-12)
            if not valid:
                raise ValueError(f'Reuse input differs from paired generator: {old_case_id}/{key}')
        validate_result(folder, old_task, row, source_config, arrays['X'].shape)
        fit_path = folder/'fits'/f'{old_task_id}.npz'
        with np.load(fit_path, allow_pickle=False) as z:
            metrics, _ = evaluate(z['A'], arrays['X'], arrays['labels'], config['evaluation']['multipliers'])
            if not close_values(metrics, row['metrics']):
                raise ValueError('Reuse metric verification failed.')
        case_id = f'nominal_p{period}_s{seed}'
        reuse[case_id] = {'arrays': arrays, 'row': row, 'fit_path': fit_path,
            'provenance': {'source_run': str(folder), 'source_task': old_task_id,
                           'source_commit': manifest['git_commit'],
                           'source_manifest_sha256': digest(folder/'manifest.json'),
                           'source_input_sha256': digest(input_path),
                           'source_record_sha256': digest(row_path),
                           'source_fit_sha256': digest(fit_path)}}
    if not reuse:
        raise ValueError('No valid matching defaults found. Omit --reuse-from to fit all tasks.')
    return reuse, manifest['environment']


def check_reuse_environment(old, new):
    if old is None:
        return
    for key in ('device_name', 'torch_cuda_build'):
        if old[key] != new[key]:
            raise ValueError(f'Reuse environment differs: {key}. Use the source environment or omit --reuse-from.')
    for key in ('numpy', 'torch'):
        if old['packages'][key] != new['packages'][key]:
            raise ValueError(f'Reuse environment differs: {key}. Use the source environment or omit --reuse-from.')


def prepare(config, reuse):
    cases, tasks = [], []
    sim = config['simulation']
    for period in sim['periods']:
        for seed in sim['seeds']:
            case_id = f'nominal_p{period}_s{seed}'
            arrays = reuse[case_id]['arrays'] if case_id in reuse else synthetic_case('nominal', seed, sim['length'], period)
            case = {'id': case_id, 'period': period, 'seed': seed, **arrays}
            cases.append(case)
            for variant in VARIANTS:
                source = reuse[case_id]['provenance'] if variant == 'fixed' and case_id in reuse else None
                tasks.append({'id': f'{case_id}__{variant}', 'case_id': case_id,
                              'method': 'QPAD', 'initialization_seed': None, 'variant': variant,
                              'true_period': period, 'seed': seed,
                              **choose_initialization(case['X'], variant, config), 'reuse': source})
    return cases, tasks


def make_manifest(config, cases, tasks, env):
    return {'experiment': config['experiment'], 'created_utc': datetime.now(timezone.utc).isoformat(),
            'git_commit': git('rev-parse', 'HEAD'), 'rebuttal_git_status': git('status', '--porcelain', '--', 'rebuttal'),
            'code_sha256': code_hashes(), 'config': config, 'environment': env,
            'inputs': [{'id': c['id'], 'period': c['period'], 'seed': c['seed'],
                        'arrays': {k: array_digest(c[k]) for k in ARRAY_KEYS}} for c in cases],
            'tasks': tasks,
            'timing_scope': 'model setup and fitting; input_estimated also includes period estimation; CUDA synchronized; excludes file I/O, evaluation and plotting'}


def run_task(task, case, config, device, output, reuse):
    import torch
    npz = output/'fits'/f'{task["id"]}.npz'
    if task['reuse'] is not None:
        source = reuse[case['id']]
        if digest(source['fit_path']) != task['reuse']['source_fit_sha256']:
            raise ValueError('Reuse source changed after preflight.')
        shutil.copyfile(source['fit_path'], npz)
        record = copy.deepcopy(source['row'])
        record.update(task)
        record['origin'] = 'reused'
        record['artifact_sha256'] = digest(npz)
        return record
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    chosen = choose_initialization(case['X'], task['variant'], config)
    if any(chosen[k] != task[k] for k in chosen):
        raise ValueError('Initialization differs from preflight.')
    initial = explicit_state(len(case['X']), task) if task['variant'] == 'input_estimated' else None
    p, a, meta, extra = fit_qpad(case['X'], 'simulation', fit_config(config, task), device,
                                initial_state=initial)
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    seconds = time.perf_counter()-started
    p, a = np.asarray(p, dtype=float), np.asarray(a, dtype=float)
    if p.shape != case['X'].shape or a.shape != p.shape or not np.isfinite(p).all() or not np.isfinite(a).all():
        raise FloatingPointError('Invalid QPAD output.')
    metrics, standardized = evaluate(a, case['X'], case['labels'], config['evaluation']['multipliers'])
    temp = npz.with_suffix('.tmp')
    with temp.open('wb') as stream:
        np.savez_compressed(stream, P=p, A=a, standardized=standardized, **extra)
    temp.replace(npz)
    return {**task, 'status': 'completed', 'origin': 'fitted', 'dataset': 'simulation',
            'scenario': 'nominal', 'group': f'seed_{case["seed"]}', 'seconds': seconds,
            'p_rmse': float(np.sqrt(np.mean((p-case['P_true'])**2))),
            'a_rmse': float(np.sqrt(np.mean((a-case['A_true'])**2))),
            'metadata': meta, 'metrics': metrics, 'artifact_sha256': digest(npz)}


def execute(output, manifest, cases, config, device, reuse, resume):
    import torch
    from threadpoolctl import threadpool_limits
    from period_prior_report import validate_saved, summarize
    if output.exists():
        if not resume:
            raise FileExistsError(f'{output} exists. Use --resume for this exact run.')
        old = json.loads((output/'manifest.json').read_text())
        for key in ('experiment', 'code_sha256', 'config', 'inputs', 'tasks'):
            if old[key] != manifest[key]:
                raise ValueError(f'Resume rejected: {key} differs.')
        for key in ('device', 'device_name', 'torch_cuda_build', 'packages'):
            if old['environment'][key] != manifest['environment'][key]:
                raise ValueError(f'Resume rejected: environment {key} differs.')
        for c, r in zip(cases, manifest['inputs']):
            with np.load(output/'inputs'/f'{c["id"]}.npz', allow_pickle=False) as z:
                if {k: array_digest(z[k]) for k in z.files} != r['arrays']:
                    raise ValueError('Resume rejected: input snapshot changed.')
    else:
        if resume:
            raise FileNotFoundError(f'Cannot resume absent output: {output}')
        (output/'inputs').mkdir(parents=True)
        (output/'fits').mkdir()
        write_json(output/'manifest.json', manifest)
        write_json(output/'config.json', config)
        for case in cases:
            np.savez_compressed(output/'inputs'/f'{case["id"]}.npz', **{k: case[k] for k in ARRAY_KEYS})
    by_id = {c['id']: c for c in cases}
    torch.set_num_threads(config['cpu_threads'])
    with threadpool_limits(limits=config['cpu_threads']):
        for index, task in enumerate(manifest['tasks'], 1):
            record_path = output/'fits'/f'{task["id"]}.json'
            if record_path.is_file():
                row = json.loads(record_path.read_text())
                if row.get('status') == 'completed':
                    validate_saved(output, task, row, config, by_id[task['case_id']])
                    print(f'[{index}/{len(manifest["tasks"])}] complete {task["id"]}', flush=True)
                    continue
            print(f'[{index}/{len(manifest["tasks"])}] {"reuse" if task["reuse"] else "fit"} {task["id"]}', flush=True)
            try:
                row = run_task(task, by_id[task['case_id']], config, device, output, reuse)
                validate_saved(output, task, row, config, by_id[task['case_id']])
            except Exception as error:
                row = {**task, 'status': 'failed', 'error': f'{type(error).__name__}: {error}',
                       'traceback': traceback.format_exc()}
                print(row['error'], flush=True)
            write_json(record_path, row)
            with (output/'events.jsonl').open('a') as f:
                f.write(json.dumps({'utc': datetime.now(timezone.utc).isoformat(), 'task': task['id'],
                                    'status': row['status'], 'error': row.get('error')})+'\n')
            write_json(output/'progress.json', {'last_task': task['id'], 'position': index,
                                               'planned': len(manifest['tasks'])})
    incomplete = summarize(output)
    print(f'Report: {output / "analysis/report.md"}; incomplete: {incomplete}', flush=True)
    return int(incomplete > 0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['check', 'run', 'summarize'])
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    parser.add_argument('--device', default='auto')
    parser.add_argument('--reuse-from', type=Path)
    parser.add_argument('--output', default='rebuttal/outputs/period_prior_v1')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    try:
        output = resolve_path(args.output)
        if args.command == 'summarize':
            from period_prior_report import summarize
            return int(summarize(output) > 0)
        config = json.loads(args.config.expanduser().resolve().read_text())
        validate_config(config)
        source = resolve_path(args.reuse_from) if args.reuse_from else None
        reuse, source_env = load_reuse(source, config)
        device, env = environment(args.device)
        check_reuse_environment(source_env, env)
        cases, tasks = prepare(config, reuse)
        manifest = make_manifest(config, cases, tasks, env)
        reused = sum(t['reuse'] is not None for t in tasks)
        print(json.dumps({'status': 'ready', 'inputs': len(cases), 'total_results': len(tasks),
                          'reused_results': reused, 'new_fits': len(tasks)-reused,
                          'device': str(device), 'output': str(output),
                          'initializations': [{'case': t['case_id'], 'variant': t['variant'],
                                               'p_hat': t['period_estimate'], 'K': t['K'],
                                               'T_initial': t['initial_duration']} for t in tasks]}, indent=2), flush=True)
        if args.command == 'check':
            return 0
        return execute(output, manifest, cases, config, device, reuse, args.resume)
    except Exception as error:
        print(f'{args.command} failed: {type(error).__name__}: {error}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())

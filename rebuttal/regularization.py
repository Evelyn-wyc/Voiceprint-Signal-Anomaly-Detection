"""Paired regularization sensitivity on saved revision inputs."""
import argparse
import copy
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sys
import traceback

import numpy as np
from evaluation import evaluate
from experiments import HERE, array_digest, digest, git, resolve_path, run_task, write_json
from period_prior import LEGACY_FITTER_SHA, check_reuse_environment, environment
from report import close_values, validate_result

DEFAULT_CONFIG = HERE/'configs/regularization.json'
GROUPS = {'lambda': ('lambda1', 'lambda2'), 'eta': ('eta1', 'eta2', 'eta3'), 'psi': ('psi',)}
CODE_FILES = ('regularization.py', 'regularization_report.py', 'qpad_fit.py',
              'models/simulation.py', 'models/real.py', 'evaluation.py', 'baselines.py',
              'experiments.py', 'report.py', 'period_prior.py')


def variants():
    return [{'variant': 'default', 'parameter_group': 'default', 'factor': 1.0}]+[
        {'variant': f'{group}_{label}', 'parameter_group': group, 'factor': factor}
        for group in GROUPS for label, factor in [('half', .5), ('double', 2.0)]]


def validate_config(config):
    if config['experiment'] != 'regularization_v1':
        raise ValueError('Expected regularization_v1 configuration.')
    selection = config['selection']
    seeds = selection['simulation_seeds']
    if not seeds or any(type(s) is not int or s < 0 for s in seeds) or len(set(seeds)) != len(seeds):
        raise ValueError('Simulation seeds must be unique nonnegative integers.')
    if selection['simulation_scenario'] != 'nominal' or selection['real_rule'] != 'first_segment_per_recording':
        raise ValueError('Expected nominal inputs and first segment per recording.')
    for value in (selection['expected_recordings'], config['cpu_threads'], config['qpad']['trace_every']):
        if type(value) is not int or value < 1:
            raise ValueError('Counts and trace interval must be positive integers.')
    for name in ('learning_rate', 'loss_scale'):
        if not np.isfinite(config['qpad'][name]) or config['qpad'][name] <= 0:
            raise ValueError(f'Invalid {name}.')
    for dataset in ('simulation', 'real'):
        q = config['qpad'][dataset]
        if type(q['K']) is not int or q['K'] < 3 or type(q['epochs']) is not int or q['epochs'] < 1:
            raise ValueError('Expected K >= 3 and a positive update budget.')
        if set(q['parameters']) != set(sum(GROUPS.values(), ())):
            raise ValueError('Unexpected regularization parameter names.')
        if any(not np.isfinite(v) or v <= 0 for v in q['parameters'].values()):
            raise ValueError('Reference penalties must be finite and positive.')
    ev = config['evaluation']
    if ev['primary_multiplier'] not in ev['multipliers'] or len(set(ev['multipliers'])) != len(ev['multipliers']) or any(not np.isfinite(v) or v <= 0 for v in ev['multipliers']):
        raise ValueError('Invalid prespecified thresholds.')


def select_inputs(records, config):
    """Selection uses dataset identifiers and segment indices only."""
    selected = []
    for seed in config['selection']['simulation_seeds']:
        matches = [r for r in records if r['dataset'] == 'simulation' and
                   r['scenario'] == config['selection']['simulation_scenario'] and r['seed'] == seed]
        if len(matches) != 1:
            raise ValueError(f'Expected one nominal input for seed {seed}.')
        selected += matches
    real = [r for r in records if r['dataset'] == 'real']
    groups = sorted({r['group'] for r in real})
    if len(groups) != config['selection']['expected_recordings']:
        raise ValueError('Source recording count differs from protocol.')
    for group in groups:
        items = [r for r in real if r['group'] == group]
        def segment(r):
            prefix, suffix = r['id'].rsplit('_', 1)
            if prefix != 'real_'+group or not suffix.isdigit():
                raise ValueError(f'Unrecognized segment identifier: {r["id"]}')
            return int(suffix)
        if len({segment(r) for r in items}) != len(items):
            raise ValueError('Duplicate segment indices.')
        selected.append(min(items, key=segment))
    return selected


def task_config(config, task):
    result = copy.deepcopy(config)
    params = result['qpad'][task['dataset']]['parameters']
    for key in GROUPS.get(task['parameter_group'], ()):
        params[key] *= task['factor']
    if params != task['parameters']:
        raise ValueError('Task parameters differ from the declared factor.')
    return result


def verify_fit(folder, task, row, config, case):
    validate_result(folder, task, row, config, case['X'].shape)
    if any(row[key] != case[key] for key in ('dataset', 'scenario', 'group', 'seed')):
        raise ValueError('Result dataset identity differs from input.')
    q = config['qpad'][case['dataset']]
    if row['metadata']['K'] != q['K'] or row['metadata']['iterations'] != q['epochs'] or row['metadata']['stopping_rule'] != 'fixed_iteration_budget':
        raise ValueError('Result settings differ from protocol.')
    with np.load(folder/'fits'/f'{task["id"]}.npz', allow_pickle=False) as z:
        metrics, standardized = evaluate(z['A'], case['X'], case['labels'], config['evaluation']['multipliers'])
        if not close_values(metrics, row['metrics']) or not np.allclose(standardized, z['standardized'], rtol=1e-9, atol=1e-10):
            raise ValueError('Metrics differ from saved components.')
        for key, output, truth in [('p_rmse', 'P', 'P_true'), ('a_rmse', 'A', 'A_true')]:
            value = float(np.sqrt(np.mean((z[output]-case[truth])**2))) if truth in case else None
            if not close_values(value, row[key]):
                raise ValueError('Component RMSE differs from saved arrays.')
        start, duration = (250., 150.) if case['dataset'] == 'real' else (0., 100.)
        initial = {'t_k': np.linspace(start, len(case['X']), q['K']+1)[:-1],
                   'T_active': np.full(q['K'], duration), 'M_independent': np.ones(q['K']),
                   'A': np.zeros(len(case['X']))}
        for key, value in initial.items():
            if not np.allclose(z['initial_'+key], value, rtol=0, atol=2e-4):
                raise ValueError('Initial state differs from the archived default.')


def validate_saved(folder, task, row, config, case):
    verify_fit(folder, task, row, task_config(config, task), case)
    if row['origin'] != ('reused' if task['reuse'] else 'fitted'):
        raise ValueError('Unexpected fit origin.')
    if task['reuse'] and row['artifact_sha256'] != task['source']['fit_sha256']:
        raise ValueError('Reused fit differs from its source.')


def prepare(source, config, refit_defaults=False):
    original = json.loads((source/'manifest.json').read_text())
    if original['experiment'] != 'necessary_rebuttal_v1':
        raise ValueError('Source must be the necessary_v1 run.')
    for key, value in config['qpad'].items():
        if original['config']['qpad'][key] != value:
            raise ValueError(f'Source QPAD {key} differs from the reference settings.')
    for key, value in config['evaluation'].items():
        if original['config']['evaluation'][key] != value:
            raise ValueError(f'Source evaluation {key} differs.')
    for name in ('models/simulation.py', 'models/real.py', 'evaluation.py', 'baselines.py'):
        if original['code_sha256'].get(name) != digest(HERE/name):
            raise ValueError(f'Source code differs: {name}.')
    if original['code_sha256'].get('qpad_fit.py') not in (LEGACY_FITTER_SHA, digest(HERE/'qpad_fit.py')):
        raise ValueError('Source uses an unrecognized fitting implementation.')
    selected = select_inputs(original['inputs'], config)
    cases, tasks, sources = [], [], {}
    for item in selected:
        path = source/'inputs'/f'{item["id"]}.npz'
        with np.load(path, allow_pickle=False) as z:
            if {k: array_digest(z[k]) for k in z.files} != item['arrays']:
                raise ValueError(f'Source input hash mismatch: {item["id"]}')
            case = {key: item[key] for key in ('id', 'dataset', 'scenario', 'group', 'seed')}
            case.update({k: z[k].copy() for k in z.files})
        if case['X'].shape != (item['length'],) or not np.isfinite(case['X']).all() or case['labels'].shape != case['X'].shape or not np.isin(case['labels'], [0, 1]).all():
            raise ValueError('Invalid source input or labels.')
        old_id = item['id']+'__QPAD__default'
        old_task = next(t for t in original['tasks'] if t['id'] == old_id)
        row_path = source/'fits'/f'{old_id}.json'
        row = json.loads(row_path.read_text())
        verify_fit(source, old_task, row, original['config'], case)
        fit_path = source/'fits'/f'{old_id}.npz'
        provenance = {'task': old_id, 'input_sha256': digest(path),
                      'record_sha256': digest(row_path), 'fit_sha256': digest(fit_path)}
        cases.append(case)
        sources[case['id']] = {'row': row, 'path': fit_path}
        for variant in variants():
            parameters = copy.deepcopy(config['qpad'][case['dataset']]['parameters'])
            for key in GROUPS.get(variant['parameter_group'], ()):
                parameters[key] *= variant['factor']
            tasks.append({'id': case['id']+'__'+variant['variant'], 'case_id': case['id'],
                          'method': 'QPAD', 'initialization_seed': None, 'dataset': case['dataset'],
                          **variant, 'parameters': parameters, 'source': provenance,
                          'reuse': variant['variant'] == 'default' and not refit_defaults})
    return cases, tasks, sources, original


def make_manifest(source, config, cases, tasks, original, env):
    return {'experiment': config['experiment'], 'created_utc': datetime.now(timezone.utc).isoformat(),
            'git_commit': git('rev-parse', 'HEAD'),
            'rebuttal_git_status': git('status', '--porcelain', '--untracked-files=all', '--', 'rebuttal'),
            'code_sha256': {name: digest(HERE/name) for name in CODE_FILES},
            'config': config, 'environment': env, 'tasks': tasks,
            'source': {'directory': str(source), 'manifest_sha256': digest(source/'manifest.json'),
                       'git_commit': original['git_commit'], 'environment': original['environment']},
            'inputs': [{**{k: c[k] for k in ('id', 'dataset', 'scenario', 'group', 'seed')},
                        'arrays': {k: array_digest(v) for k, v in c.items() if isinstance(v, np.ndarray)}} for c in cases],
            'timing_scope': 'model initialization and fixed-budget optimization; CUDA synchronized; excludes file I/O, evaluation and plotting'}


def execute(output, manifest, cases, sources, device, resume):
    import torch
    from threadpoolctl import threadpool_limits
    from regularization_report import summarize
    config = manifest['config']
    if output.exists():
        if not resume:
            raise FileExistsError('Output exists; use --resume for this exact run.')
        old = json.loads((output/'manifest.json').read_text())
        for key in ('experiment', 'code_sha256', 'config', 'inputs', 'tasks', 'source', 'environment'):
            if old[key] != manifest[key]:
                raise ValueError(f'Resume rejected: {key} differs.')
        for item in old['inputs']:
            with np.load(output/'inputs'/f'{item["id"]}.npz', allow_pickle=False) as z:
                if {k: array_digest(z[k]) for k in z.files} != item['arrays']:
                    raise ValueError('Resume input snapshot changed.')
    else:
        if resume:
            raise FileNotFoundError('Cannot resume an absent output directory.')
        (output/'inputs').mkdir(parents=True)
        (output/'fits').mkdir()
        write_json(output/'manifest.json', manifest)
        write_json(output/'config.json', config)
        for case in cases:
            np.savez_compressed(output/'inputs'/f'{case["id"]}.npz',
                                **{k: v for k, v in case.items() if isinstance(v, np.ndarray)})
    torch.set_num_threads(config['cpu_threads'])
    by_id = {c['id']: c for c in cases}
    with threadpool_limits(limits=config['cpu_threads']):
        for i, task in enumerate(manifest['tasks'], 1):
            path = output/'fits'/f'{task["id"]}.json'
            case = by_id[task['case_id']]
            if path.is_file():
                old = json.loads(path.read_text())
                if old.get('status') == 'completed':
                    validate_saved(output, task, old, config, case)
                    print(f'[{i}/{len(manifest["tasks"])}] complete {task["id"]}', flush=True)
                    continue
            print(f'[{i}/{len(manifest["tasks"])}] {"reuse" if task["reuse"] else "fit"} {task["id"]}', flush=True)
            try:
                if task['reuse']:
                    src = sources[task['case_id']]
                    if digest(src['path']) != task['source']['fit_sha256']:
                        raise ValueError('Source fit changed after preflight.')
                    shutil.copyfile(src['path'], output/'fits'/f'{task["id"]}.npz')
                    row = {**copy.deepcopy(src['row']), **task, 'origin': 'reused'}
                else:
                    row = run_task(task, case, task_config(config, task), device, output)
                    row['origin'] = 'fitted'
                validate_saved(output, task, row, config, case)
            except Exception as error:
                row = {**task, 'status': 'failed', 'error': f'{type(error).__name__}: {error}',
                       'traceback': traceback.format_exc()}
                print(row['error'], flush=True)
            write_json(path, row)
            with (output/'events.jsonl').open('a') as f:
                f.write(json.dumps({'utc': datetime.now(timezone.utc).isoformat(), 'task': task['id'],
                                    'status': row['status'], 'error': row.get('error')})+'\n')
            write_json(output/'progress.json', {'position': i, 'planned': len(manifest['tasks']), 'last_task': task['id']})
    return int(summarize(output) > 0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['check', 'run', 'summarize'])
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    parser.add_argument('--source', default='rebuttal/outputs/necessary_v1')
    parser.add_argument('--output', default='rebuttal/outputs/regularization_v1')
    parser.add_argument('--device', default='auto')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--refit-defaults', action='store_true')
    args = parser.parse_args()
    try:
        output = resolve_path(args.output)
        if args.command == 'summarize':
            from regularization_report import summarize
            return int(summarize(output) > 0)
        config = json.loads(args.config.expanduser().resolve().read_text())
        validate_config(config)
        source = resolve_path(args.source)
        cases, tasks, sources, original = prepare(source, config, args.refit_defaults)
        device, env = environment(args.device)
        if not args.refit_defaults:
            check_reuse_environment(original['environment'], env)
        manifest = make_manifest(source, config, cases, tasks, original, env)
        reused = sum(t['reuse'] for t in tasks)
        print(json.dumps({'status': 'ready', 'simulation_inputs': sum(c['dataset']=='simulation' for c in cases),
                          'real_inputs': sum(c['dataset']=='real' for c in cases), 'total_results': len(tasks),
                          'reused_results': reused, 'new_fits': len(tasks)-reused,
                          'variants': variants(), 'output': str(output), 'device': str(device)}, indent=2), flush=True)
        if args.command == 'check':
            return 0
        return execute(output, manifest, cases, sources, device, args.resume)
    except Exception as error:
        print(f'{args.command} failed: {type(error).__name__}: {error}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())

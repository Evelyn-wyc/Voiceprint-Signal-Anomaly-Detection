"""Validate and report paired period-initialization experiments."""
import json
from pathlib import Path
import numpy as np
from evaluation import evaluate
from experiments import array_digest, write_json
from report import close_values, fmt, validate_result, write_csv


def validate_saved(output, task, row, config, case):
    from period_prior import explicit_state, fit_config
    validate_result(output, task, row, fit_config(config, task), case['X'].shape)
    with np.load(output/'fits'/f'{task["id"]}.npz', allow_pickle=False) as z:
        metrics, score = evaluate(z['A'], case['X'], case['labels'], config['evaluation']['multipliers'])
        if not close_values(metrics, row['metrics']) or not np.allclose(score, z['standardized'], rtol=1e-9, atol=1e-10):
            raise ValueError('Saved metrics or scores differ from recomputed values.')
        for key, truth in [('p_rmse', 'P_true'), ('a_rmse', 'A_true')]:
            component = z['P' if key == 'p_rmse' else 'A']
            if not close_values(float(np.sqrt(np.mean((component-case[truth])**2))), row[key]):
                raise ValueError(f'Saved {key} differs from recomputed value.')
        expected = explicit_state(len(case['X']), task)
        for key, value in expected.items():
            if not np.allclose(z['initial_'+key], np.asarray(value, dtype=np.float32), rtol=0, atol=2e-4):
                raise ValueError(f'Saved initial {key} differs from planned initialization.')
    if row['metadata']['K'] != task['K'] or row['metadata']['iterations'] != config['qpad']['simulation']['epochs']:
        raise ValueError('Saved K or update count differs from plan.')
    if row.get('origin') != ('reused' if task['reuse'] else 'fitted'):
        raise ValueError('Result origin differs from plan.')
    if task['reuse'] and row['artifact_sha256'] != task['reuse']['source_fit_sha256']:
        raise ValueError('Reused array archive differs from its source.')


def stats(values):
    a = np.asarray(values, dtype=float)
    return {'mean': float(a.mean()), 'std': float(a.std(ddof=1)) if len(a) > 1 else None,
            'min': float(a.min()), 'max': float(a.max())}


def summarize(output):
    output = Path(output)
    manifest = json.loads((output/'manifest.json').read_text())
    config = manifest['config']
    primary = config['evaluation']['primary_multiplier']
    analysis = output/'analysis'
    analysis.mkdir(exist_ok=True)
    cases, input_errors = {}, {}
    for item in manifest['inputs']:
        try:
            with np.load(output/'inputs'/f'{item["id"]}.npz', allow_pickle=False) as z:
                if {k: array_digest(z[k]) for k in z.files} != item['arrays']:
                    raise ValueError('Input snapshot hash mismatch.')
                cases[item['id']] = {k: z[k].copy() for k in z.files}
        except Exception as error:
            input_errors[item['id']] = str(error)
    completed, failures, metrics_rows = [], [], []
    for task in manifest['tasks']:
        try:
            if task['case_id'] in input_errors:
                raise ValueError(input_errors[task['case_id']])
            row = json.loads((output/'fits'/f'{task["id"]}.json').read_text())
            validate_saved(output, task, row, config, cases[task['case_id']])
            metric = next(m for m in row['metrics'] if m['direction'] == 'positive' and m['rule'] == 'robust' and m['multiplier'] == primary)
            completed.append({'id': task['id'], 'case_id': task['case_id'],
                'period': task['true_period'], 'seed': task['seed'], 'variant': task['variant'],
                'origin': row['origin'], 'p_hat': task['period_estimate'], 'K': task['K'],
                'T_initial': task['initial_duration'], 'p_rmse': row['p_rmse'], 'a_rmse': row['a_rmse'],
                **{k: metric[k] for k in ('f1', 'ap', 'auroc', 'fpr')},
                'seconds': row['seconds'], 'objective': row['metadata']['final_objective'],
                'minimum_final_duration': row['metadata']['minimum_final_duration'],
                'stopping_rule': row['metadata']['stopping_rule']})
            metrics_rows += [{'id': task['id'], 'period': task['true_period'], 'seed': task['seed'],
                             'variant': task['variant'], **m} for m in row['metrics']]
        except Exception as error:
            failures.append({'id': task['id'], 'error': f'{type(error).__name__}: {error}'})
    groups, pairs, paired_summary = [], [], []
    keys = ('p_rmse', 'a_rmse', 'f1', 'ap', 'auroc', 'fpr', 'seconds')
    for period in config['simulation']['periods']:
        for variant in ('fixed', 'input_estimated'):
            selected = [r for r in completed if r['period'] == period and r['variant'] == variant]
            group = {'period': period, 'variant': variant, 'completed': len(selected),
                     'expected': len(config['simulation']['seeds'])}
            for key in keys:
                if selected:
                    group[key] = stats([r[key] for r in selected])
            groups.append(group)
        for seed in config['simulation']['seeds']:
            by_variant = {r['variant']: r for r in completed if r['period'] == period and r['seed'] == seed}
            if set(by_variant) == {'fixed', 'input_estimated'}:
                pairs.append({'period': period, 'seed': seed,
                              **{key: by_variant['input_estimated'][key]-by_variant['fixed'][key] for key in keys}})
        selected = [r for r in pairs if r['period'] == period]
        if selected:
            paired_summary.append({'period': period, 'pairs': len(selected),
                                   **{key: stats([r[key] for r in selected]) for key in keys}})
    summary = {'expected': len(manifest['tasks']), 'completed': len(completed), 'incomplete': len(failures),
               'reused': sum(r['origin'] == 'reused' for r in completed),
               'fitted': sum(r['origin'] == 'fitted' for r in completed),
               'primary_multiplier': primary, 'groups': groups, 'paired_differences': paired_summary}
    write_json(analysis/'summary.json', summary)
    write_csv(analysis/'per_fit.csv', completed)
    write_csv(analysis/'metrics.csv', metrics_rows)
    write_csv(analysis/'paired_per_seed.csv', pairs)
    write_csv(analysis/'failures.csv', failures)
    flat = [{k: v for k, v in r.items() if not isinstance(v, dict)} |
            {f'{key}_{stat}': value for key, entry in r.items() if isinstance(entry, dict) for stat, value in entry.items()}
            for r in groups]
    write_csv(analysis/'by_period.csv', flat)
    lines = ['# Period initialization comparison', '',
        f'Results: **{len(completed)}/{len(manifest["tasks"])}**; reused: {summary["reused"]}; newly fitted: {summary["fitted"]}; incomplete: {len(failures)}.', '',
        f'Generating periods: {config["simulation"]["periods"]}; seeds: {config["simulation"]["seeds"]}; updates per fit: {config["qpad"]["simulation"]["epochs"]}.',
        f'Positive detection uses c={primary:g}. RMSE measures component recovery. All fits stop at the fixed update budget.', '',
        '## Results by period', '',
        '| Period | Initialization | Completed | P RMSE | A RMSE | F1 | AP | Fit seconds |',
        '|---|---|---|---|---|---|---|---|']
    for r in groups:
        def cell(key):
            if key not in r:
                return '—'
            entry = r[key]
            return f'{fmt(entry["mean"])} ± {fmt(entry["std"])}'
        lines.append(f'| {r["period"]} | {r["variant"]} | {r["completed"]}/{r["expected"]} | '+
                     ' | '.join(cell(k) for k in ('p_rmse', 'a_rmse', 'f1', 'ap', 'seconds'))+' |')
    lines += ['', 'Entries are means ± sample standard deviations over completed seeds; per-seed values and missing results are retained.', '',
              '## Paired changes: input_estimated minus fixed', '',
              '| Period | Pairs | Δ P RMSE | Δ A RMSE | Δ F1 |', '|---|---|---|---|---|']
    for r in paired_summary:
        lines.append(f'| {r["period"]} | {r["pairs"]} | '+
                     ' | '.join(f'{fmt(r[k]["mean"])} ± {fmt(r[k]["std"])}' for k in ('p_rmse', 'a_rmse', 'f1'))+' |')
    lines += ['', f'A negative RMSE change indicates improved recovery; a positive F1 change indicates improved detection. The {len(config["simulation"]["seeds"])} seeds provide descriptive evidence over these inputs.', '',
              '## Initialization and timing', '',
              '- fixed: archived initialization, K fixed by configuration, T=100, equally spaced time shifts, amplitudes one, anomaly parameters zero.',
              '- input_estimated: current-input autocorrelation period p_hat; K=max(minimum_K, floor(N/p_hat+0.5)); t_k=kN/K; T=p_hat/2; amplitudes one; anomaly parameters zero.',
              '- Period search range and half-cycle ratio are fixed protocol assumptions. The joint initialization rule changes K and temporal scales together.',
              '- Timing includes initialization and optimization; input_estimated includes period estimation. Reused timings come from the source run, whose environment and files are recorded.',
              '- Full objective values correspond to each fitted K and are not used to select an initialization variant.', '']
    if completed:
        make_plots(output, analysis, completed, cases, config)
        lines += ['## Figures', '', '![Recovery and detection](figures/summary.png)', '',
                  'Points show individual seeds; bars show sample standard deviations.', '',
                  '![Components for the first configured seed](figures/components.png)', '',
                  'Component panels use the first configured seed for every period.', '',
                  '![Objective traces](figures/objectives.png)', '',
                  'Each panel shows a paired input at the fixed computation budget.', '']
    if failures:
        lines += ['## Incomplete results', '']+[f'- {r["id"]}: {r["error"]}' for r in failures]
    (analysis/'report.md').write_text('\n'.join(lines)+'\n')
    print(f'Validated {len(completed)}/{len(manifest["tasks"])} results. Report: {analysis / "report.md"}')
    return len(failures)


def make_plots(output, analysis, rows, cases, config):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False, 'pdf.fonttype': 42})
    dest = analysis/'figures'
    dest.mkdir(exist_ok=True)
    colors = {'fixed': '#536DA6', 'input_estimated': '#00796B'}
    periods, seeds = config['simulation']['periods'], config['simulation']['seeds']
    def save(fig, name):
        for ext in ('png', 'pdf'):
            fig.savefig(dest/f'{name}.{ext}', dpi=180, bbox_inches='tight')
        plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), layout='constrained')
    for ax, key, title in zip(axes.flat, ('p_rmse', 'a_rmse', 'f1', 'ap'),
                             ('Background RMSE', 'Anomaly RMSE', f'F1 at c={config["evaluation"]["primary_multiplier"]:g}', 'Average precision')):
        for variant, offset in [('fixed', -.12), ('input_estimated', .12)]:
            for i, period in enumerate(periods):
                values = [r[key] for r in rows if r['period'] == period and r['variant'] == variant]
                if not values:
                    continue
                ax.errorbar(i+offset, np.mean(values), yerr=np.std(values, ddof=1) if len(values)>1 else 0,
                            fmt='o', capsize=4, color=colors[variant], label=variant if i==0 else None)
                ax.scatter(i+offset+np.linspace(-.045, .045, len(values)), values,
                           s=19, alpha=.65, color=colors[variant])
        ax.set_xticks(range(len(periods)), periods)
        ax.set_xlabel('Generating period (grid points)'); ax.set_title(title); ax.grid(alpha=.15)
    axes[0, 0].legend(frameon=False)
    fig.suptitle('Period initialization: individual seeds and mean ± sample SD')
    save(fig, 'summary')
    fig, axes = plt.subplots(len(periods), 2, figsize=(12, 2.5*len(periods)), squeeze=False, layout='constrained')
    for i, period in enumerate(periods):
        case_id = f'nominal_p{period}_s{seeds[0]}'
        case = cases.get(case_id)
        if case is None:
            continue
        for ax, key, truth in zip(axes[i], ('P', 'A'), ('P_true', 'A_true')):
            ax.plot(case[truth], color='#777777', lw=1.1, label='True '+key)
            for r in rows:
                if r['case_id'] == case_id:
                    with np.load(output/'fits'/f'{r["id"]}.npz', allow_pickle=False) as z:
                        ax.plot(z[key], color=colors[r['variant']], lw=1, alpha=.85, label=r['variant'])
            ax.set_title(f'Period {period}, seed {seeds[0]}: {key}'); ax.set_xlabel('Sample index')
    axes[0, 0].legend(frameon=False, fontsize=8)
    save(fig, 'components')
    fig, axes = plt.subplots(len(periods), len(seeds), figsize=(4*len(seeds), 2.5*len(periods)), squeeze=False, layout='constrained')
    for i, period in enumerate(periods):
        for j, seed in enumerate(seeds):
            ax = axes[i, j]
            for r in rows:
                if r['period'] == period and r['seed'] == seed:
                    with np.load(output/'fits'/f'{r["id"]}.npz', allow_pickle=False) as z:
                        ax.plot(z['trace'][:, 0], z['trace'][:, 1], color=colors[r['variant']], label=r['variant'])
            ax.set_title(f'Period {period}, seed {seed}'); ax.set_xlabel('Updates'); ax.set_ylabel('Objective')
    axes[0, 0].legend(frameon=False, fontsize=8)
    save(fig, 'objectives')

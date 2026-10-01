"""Validate regularization runs and report every paired setting."""
import json
from pathlib import Path
import numpy as np
from experiments import array_digest, write_json
from report import fmt, write_csv

METRICS = ('p_rmse', 'a_rmse', 'f1', 'ap', 'auroc', 'fpr', 'seconds')


def statistics(values):
    values = np.asarray([v for v in values if v is not None], dtype=float)
    return {'n': len(values), 'mean': float(values.mean()) if len(values) else None,
            'std': float(values.std(ddof=1)) if len(values) > 1 else None,
            'min': float(values.min()) if len(values) else None,
            'max': float(values.max()) if len(values) else None}


def summarize(output):
    from regularization import validate_config, validate_saved, variants
    output = Path(output)
    manifest = json.loads((output/'manifest.json').read_text())
    config = manifest['config']
    validate_config(config)
    if config != json.loads((output/'config.json').read_text()):
        raise ValueError('Saved configuration differs from manifest.')
    expected_variants = {v['variant']: v for v in variants()}
    expected = {(r['id'], v) for r in manifest['inputs'] for v in expected_variants}
    if len(manifest['tasks']) != len(expected) or {(t['case_id'], t['variant']) for t in manifest['tasks']} != expected:
        raise ValueError('Manifest does not contain one result per input and setting.')
    analysis = output/'analysis'
    analysis.mkdir(exist_ok=True)
    cases, errors = {}, {}
    for item in manifest['inputs']:
        try:
            with np.load(output/'inputs'/f'{item["id"]}.npz', allow_pickle=False) as z:
                if {k: array_digest(z[k]) for k in z.files} != item['arrays']:
                    raise ValueError('Input snapshot hash mismatch.')
                cases[item['id']] = {**item, **{k: z[k].copy() for k in z.files}}
        except Exception as error:
            errors[item['id']] = str(error)
    rows, failures, metrics = [], [], []
    for task in manifest['tasks']:
        try:
            if any(task[k] != v for k, v in expected_variants[task['variant']].items()):
                raise ValueError('Task factor/group differs from declared variant.')
            if task['case_id'] in errors:
                raise ValueError(errors[task['case_id']])
            record = json.loads((output/'fits'/f'{task["id"]}.json').read_text())
            validate_saved(output, task, record, config, cases[task['case_id']])
            primary = next(m for m in record['metrics'] if m['direction'] == 'positive' and m['rule'] == 'robust' and m['multiplier'] == config['evaluation']['primary_multiplier'])
            rows.append({**{k: task[k] for k in ('id', 'case_id', 'variant', 'parameter_group', 'factor', 'dataset')},
                         **{k: record[k] for k in ('group', 'seed', 'origin', 'seconds', 'p_rmse', 'a_rmse')},
                         **{k: primary[k] for k in ('positive_points', 'f1', 'ap', 'auroc', 'fpr')},
                         **task['parameters'], 'objective': record['metadata']['final_objective']})
            metrics += [{'id': task['id'], 'dataset': task['dataset'], 'variant': task['variant'], **m} for m in record['metrics']]
        except Exception as error:
            failures.append({'id': task['id'], 'error': f'{type(error).__name__}: {error}'})
    pairs = []
    defaults = {r['case_id']: r for r in rows if r['variant'] == 'default'}
    for row in rows:
        base = defaults.get(row['case_id'])
        if row['variant'] == 'default' or base is None:
            continue
        pair = {k: row[k] for k in ('case_id', 'dataset', 'group', 'seed', 'variant', 'positive_points')}
        for key in METRICS:
            pair[key] = row[key]-base[key] if row[key] is not None and base[key] is not None else None
        with np.load(output/'fits'/f'{row["id"]}.npz', allow_pickle=False) as z, np.load(output/'fits'/f'{base["id"]}.npz', allow_pickle=False) as b:
            for key in ('P', 'A'):
                pair[key+'_output_change_rmse'] = float(np.sqrt(np.mean((z[key]-b[key])**2)))
        pairs.append(pair)
    groups, paired_groups = [], []
    for dataset in ('simulation', 'real'):
        expected_count = sum(r['dataset'] == dataset for r in manifest['inputs'])
        for variant in expected_variants:
            selected = [r for r in rows if r['dataset'] == dataset and r['variant'] == variant]
            group = {'dataset': dataset, 'variant': variant, 'expected': expected_count, 'completed': len(selected),
                     'positive_inputs': sum(r['positive_points'] > 0 for r in selected)}
            for key in METRICS:
                # Match the main report: positive-label inputs for detection averages;
                # retain all input-specific results and all-input false-positive rates.
                group[key] = statistics([r[key] for r in selected if key not in ('f1', 'ap', 'auroc') or r['positive_points'] > 0])
            group['normal_fpr'] = statistics([r['fpr'] for r in selected if r['positive_points'] == 0])
            groups.append(group)
            if variant != 'default':
                paired = [r for r in pairs if r['dataset'] == dataset and r['variant'] == variant]
                paired_groups.append({'dataset': dataset, 'variant': variant, 'pairs': len(paired),
                    **{key: statistics([r[key] for r in paired if key not in ('f1', 'ap', 'auroc') or r['positive_points'] > 0])
                       for key in (*METRICS, 'P_output_change_rmse', 'A_output_change_rmse')}})
    summary = {'expected': len(manifest['tasks']), 'completed': len(rows), 'incomplete': len(failures),
               'reused': sum(r['origin'] == 'reused' for r in rows),
               'fitted': sum(r['origin'] == 'fitted' for r in rows), 'groups': groups, 'paired_differences': paired_groups,
               'new_fit_seconds': sum(r['seconds'] for r in rows if r['origin'] == 'fitted')}
    write_json(analysis/'summary.json', summary)
    write_csv(analysis/'per_fit.csv', rows)
    write_csv(analysis/'paired_per_input.csv', pairs)
    write_csv(analysis/'metrics.csv', metrics)
    write_csv(analysis/'failures.csv', failures)
    flat = [{k: v for k, v in g.items() if not isinstance(v, dict)} |
            {f'{key}_{stat}': v for key, value in g.items() if isinstance(value, dict) for stat, v in value.items()} for g in groups]
    write_csv(analysis/'by_setting.csv', flat)
    lines = ['# Regularization sensitivity', '',
             f'Results: **{len(rows)}/{len(manifest["tasks"])}**; reused: {summary["reused"]}; newly fitted: {summary["fitted"]}; incomplete: {len(failures)}.', '',
             'Each setting changes one parameter group. Lambda scales (lambda1, lambda2); eta scales (eta1, eta2, eta3); psi scales the similarity penalty. Factors are 0.5, 1 and 2; factor 1 is shared by all groups.', '',
             'Simulation inputs use the configured nominal seeds. Real inputs are the first segment of every source recording, selected by numeric segment index. Selection uses identifiers; labels enter evaluation.', '',
             f'F1 uses positive detection at c={config["evaluation"]["primary_multiplier"]:g}. AP/AUROC use continuous scores. Means and sample standard deviations describe the completed inputs. Detection means use positive-label inputs; FPR retains all inputs. Each metric records its defined count in summary.json.', '',
             '## Results', '', '| Dataset | Setting | Complete | Positive inputs | P RMSE | A RMSE | F1 | AP | FPR |',
             '|---|---|---|---|---|---|---|---|---|']
    def cell(entry):
        return '—' if entry['mean'] is None else f'{fmt(entry["mean"])} ± {fmt(entry["std"])}'
    for g in groups:
        lines.append(f'| {g["dataset"]} | {g["variant"]} | {g["completed"]}/{g["expected"]} | {g["positive_inputs"]} | '+
                     ' | '.join(cell(g[k]) for k in ('p_rmse', 'a_rmse', 'f1', 'ap', 'fpr'))+' |')
    lines += ['', '## Paired differences from the default', '',
              '| Dataset | Setting | Pairs | Delta A RMSE | Delta F1 | Delta AP |', '|---|---|---|---|---|---|']
    for g in paired_groups:
        lines.append(f'| {g["dataset"]} | {g["variant"]} | {g["pairs"]} | '+
                     ' | '.join(cell(g[k]) for k in ('a_rmse', 'f1', 'ap'))+' |')
    lines += ['', 'Negative RMSE changes indicate improved recovery; positive F1/AP changes indicate improved detection. All per-input values and paired changes are retained.', '',
              '## Interpretation and timing', '',
              '- This experiment measures response to grouped coefficient changes at fixed initialization, thresholds and update budgets. The simulated ground truth supports component-error measurements; the real subset supports within-recording paired detection comparisons.',
              '- All settings are reported. The protocol retains the reference coefficients and does not select a new configuration by these evaluation labels.',
              '- The tested group factors describe a finite range. Individual-coefficient and cross-group interactions require separate evidence.',
              '- Objective values use different regularization weights and are recorded per fit, without using them to rank settings.',
              f'- New fitting time: {summary["new_fit_seconds"]:.2f} seconds. Timing covers model setup and optimization. Reused default times retain their recorded source environment.',
              '- Historical parameter-selection provenance and data-acquisition/annotation details remain separate factual questions.', '']
    if rows:
        make_plots(analysis, rows, config)
        lines += ['## Figures', '', '![Simulation parameter response](figures/simulation.png)', '',
                  '![Real subset parameter response](figures/real.png)', '',
                  'Thin curves show every input; blue points and bars show means and sample standard deviations. Each panel includes the same default at factor 1. Simulation panels report anomaly RMSE and F1; real panels report F1 and AP. The real subset contains one segment per recording.', '']
    if failures:
        lines += ['## Incomplete results', '']+[f'- {r["id"]}: {r["error"]}' for r in failures]
    (analysis/'report.md').write_text('\n'.join(lines)+'\n')
    print(f'Validated {len(rows)}/{len(manifest["tasks"])} results. Report: {analysis / "report.md"}', flush=True)
    return len(failures)


def make_plots(analysis, rows, config):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False, 'pdf.fonttype': 42})
    dest = analysis/'figures'
    dest.mkdir(exist_ok=True)
    for dataset, keys in [('simulation', ('a_rmse', 'f1')), ('real', ('f1', 'ap'))]:
        fig, axes = plt.subplots(2, 3, figsize=(11, 6), layout='constrained', squeeze=False)
        selected = [r for r in rows if r['dataset'] == dataset]
        for col, group in enumerate(('lambda', 'eta', 'psi')):
            variants = (group+'_half', 'default', group+'_double')
            for ax, key in zip(axes[:, col], keys):
                sets = []
                for case in sorted({r['case_id'] for r in selected}):
                    chosen = {r['variant']: r for r in selected if r['case_id'] == case}
                    vals = [chosen[v][key] if v in chosen and (key == 'a_rmse' or chosen[v]['positive_points'] > 0) else None for v in variants]
                    vals = [np.nan if v is None else v for v in vals]
                    ax.plot([.5, 1, 2], vals, color='#8896A8', alpha=.45, lw=.9, marker='.', ms=4)
                    sets.append(vals)
                if sets:
                    array = np.asarray(sets)
                    for j, factor in enumerate((.5, 1, 2)):
                        values = array[:, j][np.isfinite(array[:, j])]
                        if len(values):
                            ax.errorbar(factor, np.mean(values), yerr=np.std(values, ddof=1) if len(values)>1 else 0,
                                        fmt='o', color='#245A81', capsize=4)
                ax.set_xscale('log', base=2); ax.set_xticks([.5, 1, 2], ['0.5', '1', '2'])
                ax.set_xlabel(group+' multiplier'); ax.grid(alpha=.15)
                label = {'a_rmse': 'Anomaly RMSE', 'f1': f'F1 at c={config["evaluation"]["primary_multiplier"]:g}', 'ap': 'Average precision'}[key]
                ax.set_title(label)
        fig.suptitle(f'{dataset.capitalize()} regularization response: inputs and mean ± sample SD')
        for ext in ('png', 'pdf'):
            fig.savefig(dest/f'{dataset}.{ext}', dpi=180, bbox_inches='tight')
        plt.close(fig)

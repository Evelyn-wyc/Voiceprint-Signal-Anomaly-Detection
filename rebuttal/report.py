"""Validate returned artifacts and summarize paired revision experiments."""
import csv
import hashlib
import json
import math
from pathlib import Path
import zipfile
import numpy as np
from evaluation import evaluate


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def array_hash(a):
    a = np.ascontiguousarray(a)
    return hashlib.sha256(str(a.dtype).encode()+str(a.shape).encode()+a.tobytes()).hexdigest()


def write_csv(path, rows):
    fields = list(dict.fromkeys(k for row in rows for k in row)) or ['status']
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def fmt(x):
    return f'{x:.4g}' if isinstance(x, (float, int)) and math.isfinite(x) else '—'


def mean(values):
    values = [x for x in values if x is not None]
    return float(np.mean(values)) if values else None


def close_values(a, b):
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(close_values(a[k], b[k]) for k in a)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(close_values(x, y) for x, y in zip(a, b))
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return math.isclose(a, b, rel_tol=1e-7, abs_tol=1e-10)
    return a == b


def validate_result(run, task, row, config, shape):
    if any(row.get(k) != v for k, v in task.items()):
        raise ValueError('Result identity does not match the planned task.')
    if row.get('status') != 'completed':
        raise ValueError(row.get('error', 'Run is not complete.'))
    path = run/'fits'/f'{task["id"]}.npz'
    if sha(path) != row['artifact_sha256']:
        raise ValueError('Fitted artifact hash mismatch.')
    if not math.isfinite(row['seconds']) or row['seconds'] < 0:
        raise ValueError('Invalid recorded timing.')
    with np.load(path, allow_pickle=False) as data:
        for k in ['P', 'A', 'standardized']:
            if data[k].shape != shape or not np.isfinite(data[k]).all():
                raise ValueError(f'Invalid {k} component shape or values.')
        if task['method'] == 'QPAD':
            trace = data['trace']
            q = config['qpad']['real' if row['dataset'] == 'real' else 'simulation']
            if trace.ndim != 2 or trace.shape[1] != 2 or not np.isfinite(trace).all():
                raise ValueError('Invalid QPAD objective trace.')
            if trace[0, 0] != 0 or trace[-1, 0] != q['epochs'] or not (np.diff(trace[:, 0]) > 0).all():
                raise ValueError('Incomplete QPAD iteration trace.')
            if not math.isclose(float(trace[-1, 1]), row['metadata']['final_objective'], rel_tol=1e-9):
                raise ValueError('Objective does not match final trace value.')
            for k in ['initial_t_k', 'final_t_k', 'initial_T_active', 'final_T_active',
                      'initial_M_independent', 'final_M_independent']:
                if data[k].shape != (q['K'],) or not np.isfinite(data[k]).all():
                    raise ValueError(f'Invalid QPAD parameter artifact: {k}')
            for k in ['initial_A', 'final_A']:
                if data[k].shape != shape or not np.isfinite(data[k]).all():
                    raise ValueError(f'Invalid QPAD anomaly parameter artifact: {k}')


def aggregate(rows, fits, primary):
    out = []
    for dataset, scenario in sorted({(r['dataset'], r['scenario']) for r in fits}):
        methods = sorted({r['method'] for r in fits if r['dataset'] == dataset and r['scenario'] == scenario})
        for method in methods:
            expected = [r for r in fits if r['dataset'] == dataset and r['scenario'] == scenario and r['method'] == method and r['initialization_seed'] is None]
            for direction, rule, multiplier in [('positive', 'zero', 0.0)]+[(d, 'robust', t) for d in ['positive', 'two_sided'] for t in primary]:
                chosen = [r for r in rows if r['dataset'] == dataset and r['scenario'] == scenario and r['method'] == method and r['initialization_seed'] is None and (r['direction'], r['rule'], r['multiplier']) == (direction, rule, multiplier)]
                if not expected:
                    continue
                anomalous = [r for r in chosen if r['positive_points'] > 0]
                normal = [r for r in chosen if r['positive_points'] == 0]
                neg = sum(r['fp']+r['tn'] for r in chosen)
                normal_neg = sum(r['fp']+r['tn'] for r in normal)
                out.append({'dataset': dataset, 'scenario': scenario, 'method': method,
                            'direction': direction, 'rule': rule, 'multiplier': multiplier,
                            'expected': len(expected), 'completed': len(chosen), 'incomplete': len(expected)-len(chosen),
                            'with_positive_labels': len(anomalous), 'normal_sequences': len(normal),
                            'f1_macro_with_anomaly': mean([r['f1'] for r in anomalous]),
                            'precision_macro_with_anomaly': mean([r['precision'] for r in anomalous]),
                            'recall_macro_with_anomaly': mean([r['recall'] for r in anomalous]),
                            'ap_macro': mean([r['ap'] for r in chosen]),
                            'auroc_macro': mean([r['auroc'] for r in chosen]),
                            'fpr_pooled': sum(r['fp'] for r in chosen)/neg if neg else None,
                            'fpr_normal': sum(r['fp'] for r in normal)/normal_neg if normal_neg else None,
                            'p_rmse_mean': mean([r['p_rmse'] for r in chosen]),
                            'a_rmse_mean': mean([r['a_rmse'] for r in chosen]),
                            'seconds_mean': mean([r['seconds'] for r in chosen])})
    return out


def paired_comparisons(rows, config):
    primary = config['evaluation']['primary_multiplier']
    rows = [r for r in rows if r['initialization_seed'] is None and r['rule'] == 'robust' and r['multiplier'] == primary]
    out = []
    rng = np.random.default_rng(8301)
    for dataset, scenario, direction in sorted({(r['dataset'], r['scenario'], r['direction']) for r in rows}):
        group_rows = [r for r in rows if (r['dataset'], r['scenario'], r['direction']) == (dataset, scenario, direction)]
        q = {r['case_id']: r for r in group_rows if r['method'] == 'QPAD'}
        for method in [m for m in config['methods'] if m != 'QPAD']:
            other = {r['case_id']: r for r in group_rows if r['method'] == method}
            for metric in ['f1', 'ap', 'auroc', 'fpr', 'p_rmse', 'a_rmse']:
                pairs = [case for case in sorted(q.keys() & other.keys())
                         if q[case].get(metric) is not None and other[case].get(metric) is not None
                         and (metric != 'f1' or q[case]['positive_points'] > 0)]
                if not pairs:
                    continue
                # Recordings are resampling units for real data; seeds for simulations.
                blocks = {}
                for case in pairs:
                    blocks.setdefault(q[case]['group'], []).append(q[case][metric]-other[case][metric])
                values = np.array([np.mean(v) for _, v in sorted(blocks.items())])
                boot = np.mean(values[rng.integers(0, len(values), (config['evaluation']['bootstrap_repeats'], len(values)))], axis=1)
                lo, hi = np.quantile(boot, [0.025, 0.975]) if len(values) > 1 else (None, None)
                out.append({'dataset': dataset, 'scenario': scenario, 'direction': direction,
                            'baseline': method, 'metric': metric, 'matched_cases': len(pairs),
                            'resampling_units': len(values), 'delta_qpad_minus_baseline': float(values.mean()),
                            'ci95_low': None if lo is None else float(lo), 'ci95_high': None if hi is None else float(hi),
                            'aggregation': 'equal recording weight' if dataset == 'real' else 'equal seed weight',
                            'qpad_only_cases': len(q.keys()-other.keys()), 'baseline_only_cases': len(other.keys()-q.keys()),
                            'matched_case_ids': ';'.join(pairs)})
    return out


def initialization_summary(run, records, config):
    out = []
    primary = config['evaluation']['primary_multiplier']
    for name in config['initialization']['signals']:
        case = 'sim_'+name
        selected = [r for r in records if r['case_id'] == case and r['method'] == 'QPAD' and r['status'] == 'completed']
        default = next((r for r in selected if r['initialization_seed'] is None), None)
        random = [r for r in selected if r['initialization_seed'] is not None]
        if not selected:
            continue
        row = {'case_id': case, 'random_expected': len(config['initialization']['seeds']),
               'random_completed': len(random), 'default_available': default is not None}
        for metric in ['objective', 'a_rmse', 'p_rmse', 'f1_positive', 'f1_two_sided']:
            def value(record):
                if metric == 'objective':
                    return record['metadata']['final_objective']
                if metric.startswith('f1_'):
                    direction = metric.removeprefix('f1_')
                    return next(m['f1'] for m in record['metrics'] if m['direction'] == direction and m['rule'] == 'robust' and m['multiplier'] == primary)
                return record[metric]
            numbers = [value(r) for r in random]
            row[f'{metric}_default'] = value(default) if default else None
            row[f'{metric}_random_mean'] = mean(numbers)
            row[f'{metric}_random_std'] = float(np.std(numbers, ddof=1)) if len(numbers) > 1 else None
            row[f'{metric}_random_min'] = min(numbers) if numbers else None
            row[f'{metric}_random_max'] = max(numbers) if numbers else None
        if default and random:
            with np.load(run/'fits'/f'{default["id"]}.npz', allow_pickle=False) as base:
                p0, a0 = base['P'], base['A']
            dp, da = [], []
            for r in random:
                with np.load(run/'fits'/f'{r["id"]}.npz', allow_pickle=False) as data:
                    dp.append(float(np.sqrt(np.mean((data['P']-p0)**2))))
                    da.append(float(np.sqrt(np.mean((data['A']-a0)**2))))
            row['p_difference_rmse_mean'] = mean(dp)
            row['a_difference_rmse_mean'] = mean(da)
        out.append(row)
    return out


def summarize(run):
    run = Path(run)
    manifest = json.loads((run/'manifest.json').read_text())
    config = json.loads((run/'config.json').read_text())
    if config != manifest['config'] or config['experiment'] != 'necessary_rebuttal_v1':
        raise ValueError('Run configuration differs from its manifest.')
    if len({t['id'] for t in manifest['tasks']}) != len(manifest['tasks']):
        raise ValueError('Duplicate planned tasks.')
    inputs = {}
    for record in manifest['inputs']:
        with np.load(run/'inputs'/f'{record["id"]}.npz', allow_pickle=False) as data:
            arrays = {k: data[k].copy() for k in data.files}
        if {k: array_hash(v) for k, v in arrays.items()} != record['arrays']:
            raise ValueError(f'Input snapshot integrity failure: {record["id"]}')
        if arrays['X'].shape != (record['length'],) or arrays['labels'].shape != arrays['X'].shape:
            raise ValueError('Invalid saved input/label shape.')
        inputs[record['id']] = (record, arrays)
    records, metrics, fits = [], [], []
    for task in manifest['tasks']:
        record, arrays = inputs[task['case_id']]
        path = run/'fits'/f'{task["id"]}.json'
        row = {**task, 'status': 'missing', 'error': 'Result file absent.'}
        if path.exists():
            try:
                row = json.loads(path.read_text())
                if not isinstance(row, dict) or any(row.get(k) != v for k, v in task.items()):
                    raise ValueError('Result identity differs from manifest.')
                if row.get('status') not in ['completed', 'failed']:
                    raise ValueError('Unknown result status.')
                if row.get('status') == 'completed':
                    validate_result(run, task, row, config, arrays['X'].shape)
                    with np.load(run/'fits'/f'{task["id"]}.npz', allow_pickle=False) as data:
                        recomputed, z = evaluate(data['A'], arrays['X'], arrays['labels'], config['evaluation']['multipliers'])
                        if not close_values(recomputed, row['metrics']) or not np.allclose(z, data['standardized'], rtol=1e-8, atol=1e-10):
                            raise ValueError('Saved metrics/scores differ from their arrays.')
                        for name, key in [('p_rmse', 'P'), ('a_rmse', 'A')]:
                            truth = key+'_true'
                            expected = float(np.sqrt(np.mean((data[key]-arrays[truth])**2))) if truth in arrays else None
                            if not close_values(expected, row[name]):
                                raise ValueError(f'{name} differs from saved components/truth.')
            except (OSError, ValueError, KeyError, TypeError, EOFError, zipfile.BadZipFile) as error:
                row = {**task, 'status': 'incomplete_artifacts', 'error': f'{type(error).__name__}: {error}'}
        row.update({k: record[k] for k in ['dataset', 'scenario', 'group', 'seed']})
        records.append(row)
        fit = {k: row.get(k) for k in [*task.keys(), 'dataset', 'scenario', 'group', 'seed', 'status', 'seconds', 'a_rmse', 'p_rmse', 'error']}
        fit['solver_status'] = row.get('metadata', {}).get('solver_status')
        fit['metadata_json'] = json.dumps(row.get('metadata', {}), ensure_ascii=False)
        fits.append(fit)
        if row['status'] == 'completed':
            common = {k: row[k] for k in [*task.keys(), 'dataset', 'scenario', 'group', 'seed', 'seconds', 'p_rmse', 'a_rmse']}
            metrics.extend([{**common, **m} for m in row['metrics']])
    grouped = aggregate(metrics, fits, config['evaluation']['multipliers'])
    paired = paired_comparisons(metrics, config)
    initial = initialization_summary(run, records, config) if 'C' in manifest['parts'] else []
    out = run/'analysis'
    out.mkdir(exist_ok=True)
    for name, rows in [('fits', fits), ('metrics', metrics), ('metrics_by_group', grouped),
                       ('paired_comparisons', paired), ('initialization', initial)]:
        write_csv(out/f'{name}.csv', rows)
    incomplete = [r for r in records if r['status'] != 'completed']
    write_csv(out/'failures.csv', [{k: r.get(k) for k in ['id', 'status', 'error']} for r in incomplete])
    primary = config['evaluation']['primary_multiplier']
    text = ['# QPAD 必要返修实验汇总', '',
            f'计划 {len(records)} 次拟合；结果齐备 {len(records)-len(incomplete)} 次；失败或缺失 {len(incomplete)} 次。', '',
            f'- 代码：`{manifest.get("git_commit")}`；范围：{", ".join(manifest["parts"])}。',
            f'- 设备：{manifest["environment"]["device_name"]}；QPAD 使用指定设备，其他方法使用 CPU。',
            f'- 主阈值倍数：{primary:g}；2/3/4 倍等预定阈值共用每次分解。实际列表见 config.json。',
            '- QPAD 保留归档模型、初始化和正则参数；固定迭代预算不代表达到收敛条件。',
            '- F1 汇总包含所有有异常序列的有效零分；无异常序列单独报告误报。AUROC/AP 未定义处留空。', '',
            '## A：真实数据公平比较（主单向规则）', '',
            '| 方法 | 完成/预期 | F1 | AP | AUROC | 全部负标签点 FPR | 无正标签片段 FPR | 平均秒 |',
            '|---|---:|---:|---:|---:|---:|---:|---:|']
    for g in grouped:
        if g['dataset'] == 'real' and (g['direction'], g['rule'], g['multiplier']) == ('positive', 'robust', primary):
            text.append('| '+' | '.join([g['method'], f'{g["completed"]}/{g["expected"]}',
                        *[fmt(g[k]) for k in ['f1_macro_with_anomaly', 'ap_macro', 'auroc_macro', 'fpr_pooled', 'fpr_normal', 'seconds_mean']]])+' |')
    text += ['', '## B：仿真比较（共用 A 的比较协议）', '',
             '| 场景 | 规则 | 方法 | 完成/预期 | F1 | AP | A RMSE | P RMSE | FPR |',
             '|---|---|---|---:|---:|---:|---:|---:|---:|']
    for g in grouped:
        directions = ['positive', 'two_sided'] if g['scenario'] in ['negative', 'bidirectional'] else ['positive']
        if g['dataset'] == 'simulation' and g['direction'] in directions and (g['rule'], g['multiplier']) == ('robust', primary):
            text.append('| '+' | '.join([g['scenario'], g['direction'], g['method'], f'{g["completed"]}/{g["expected"]}',
                        *[fmt(g[k]) for k in ['f1_macro_with_anomaly', 'ap_macro', 'a_rmse_mean', 'p_rmse_mean', 'fpr_pooled']]])+' |')
    text += ['', '所有场景的单向/双向规则、零阈值和阈值敏感性见 metrics_by_group.csv。双向规则使用 |A−median(A)| 的标准化分数；其表现需要根据结果判断。', '',
             '## C：随机初值（固定输入）', '',
             '| 信号 | 随机完成/预期 | 默认 A RMSE | 随机 A RMSE 均值 ± SD | 默认 F | 随机 F 均值 ± SD |',
             '|---|---:|---:|---:|---:|---:|']
    for row in initial:
        text.append(f'| {row["case_id"]} | {row["random_completed"]}/{row["random_expected"]} | {fmt(row["a_rmse_default"])} | {fmt(row["a_rmse_random_mean"])} ± {fmt(row["a_rmse_random_std"])} | {fmt(row["objective_default"])} | {fmt(row["objective_random_mean"])} ± {fmt(row["objective_random_std"])} |')
    text += ['', '随机初值均值、样本标准差和范围只在预定随机扰动运行上计算，默认初值单列。每次初始/最终参数及目标曲线在 fits/*.npz。', '',
             '## 配对比较与材料', '',
             '- paired_comparisons.csv：同一输入下 QPAD 减基线的差值与 95% bootstrap 区间。',
             '- 真实数据按原始录音分组重采样、录音等权；仿真按种子重采样。仅 5 个种子时区间精度有限。',
             '- 配对表保留匹配数量和缺失数量；故障或缺失不会被记成零分。',
             '- inputs/*.npz 保存配套 X、标签及仿真 P/A/noise 真值；manifest.json 保存代码、数据哈希和环境。',
             '- fits.csv 保存全部状态、计算成本及求解器停止信息；metrics.csv 保存逐信号完整评价。',
             '- failures.csv 列出失败与不完整条目。数值完成、满足算法容差及恢复准确是不同判断。', '',
             '## 对应审稿意见', '',
             '- A：R1.3 阈值、R1.5 比较公平性；运行参数与计时支持 R2.5。',
             '- B：R1.1 包络适用范围、R2.3 负向异常、R2.4 失配/噪声与统计比较。',
             '- C：R1.2 初值敏感性；不能由有限次数直接推断联合分解唯一。',
             '- R1.4 理论与实现一致性、数据来源及文献说明仍需在稿件中单独处理。', '']
    if incomplete:
        text += ['## 失败或缺失', '']
        text += [f'- {r["id"]}: {r.get("error", r["status"])}' for r in incomplete]
    (out/'report.md').write_text('\n'.join(text))
    (out/'summary.json').write_text(json.dumps({'expected': len(records), 'completed': len(records)-len(incomplete),
                                               'incomplete': len(incomplete), 'groups': grouped,
                                               'initialization': initial}, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    return len(incomplete)

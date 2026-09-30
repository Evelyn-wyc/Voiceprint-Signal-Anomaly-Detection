"""Summarize one returned phase-diagnostic run without loading PyTorch.

Usage: python rebuttal/summarize.py PATH_TO_PHASE_DIRECTORY
Requires NumPy. Reads the saved run and writes analysis/*.csv and report.md.
"""
import argparse
import csv
import json
import math
from pathlib import Path
import sys
import zipfile

import numpy as np

METRICS = ['iterations', 'final_objective', 'a_rmse_to_saved_ground_truth',
           'max_onset_movement_samples', 'max_duration_movement_samples', 'seconds']


def write_csv(path, rows, fields):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


def format_number(value):
    return f'{value:.6g}' if isinstance(value, (float, int)) and math.isfinite(value) else '—'


def summarize(run_dir, output_dir=None):
    config = json.loads((run_dir / 'config.json').read_text())
    manifest = json.loads((run_dir / 'manifest.json').read_text())
    if config.get('experiment') != 'original_phase_probe':
        raise ValueError('This summary entry point expects an original_phase_probe run.')
    if manifest.get('resolved_config') != config:
        raise ValueError('config.json differs from the resolved configuration in manifest.json.')
    expected = [(name, float(shift)) for name in config['signals'] for shift in config['phase_shifts']]
    if len(set(expected)) != len(expected):
        raise ValueError('Configuration contains duplicate cases.')
    summary_path = run_dir / 'summary.json'
    source_rows = json.loads(summary_path.read_text())['runs'] if summary_path.exists() else []
    found = {}
    for row in source_rows:
        key = (row['signal'], float(row['phase_shift_samples']))
        if key not in expected or key in found:
            raise ValueError(f'Unexpected or duplicate run in summary.json: {key}')
        found[key] = row
    lengths = {row['signal']: row['length'] for row in manifest['inputs']}
    rows, components = [], {}
    for name, shift in expected:
        key = (name, shift)
        row = dict(found.get(key, {'signal': name, 'phase_shift_samples': shift,
                                   'status': 'missing', 'error': 'Expected run absent from summary.json.'}))
        if row['status'] == 'completed_fixed_budget':
            try:
                if row['iterations'] != config['epochs']:
                    raise ValueError('Iteration count differs from configuration.')
                if any(not math.isfinite(row[k]) for k in METRICS):
                    raise ValueError('Summary contains a nonfinite metric.')
                prefix = f'{name}_phase{shift:g}'
                with (run_dir / f'{prefix}_trace.csv').open(newline='') as stream:
                    trace = list(csv.DictReader(stream))
                if not trace or int(trace[-1]['step']) != config['epochs']:
                    raise ValueError('Loss trace does not reach the configured iteration budget.')
                values = [float(item['objective']) for item in trace]
                if not all(math.isfinite(v) for v in values) or not math.isclose(values[-1], row['final_objective'], rel_tol=1e-9, abs_tol=1e-12):
                    raise ValueError('Loss trace and final objective are inconsistent.')
                with np.load(run_dir / f'{prefix}.npz', allow_pickle=False) as data:
                    p, a = data['P'].astype(float), data['A'].astype(float)
                if p.shape != (lengths[name],) or a.shape != p.shape:
                    raise ValueError('Component shape differs from input length.')
                if not np.isfinite(p).all() or not np.isfinite(a).all():
                    raise ValueError('Component contains a nonfinite value.')
                components[key] = (p, a)
            except (OSError, ValueError, KeyError, TypeError, EOFError, zipfile.BadZipFile) as error:
                row['status'] = 'incomplete_artifacts'
                row['error'] = f'{type(error).__name__}: {error}'
        rows.append(row)
    by_key = {(r['signal'], float(r['phase_shift_samples'])): r for r in rows}
    comparisons = []
    for name, shift in expected:
        if shift == 0:
            continue
        pair = {'signal': name, 'phase_shift_samples': shift}
        base, changed = (name, 0.0), (name, shift)
        if base not in components or changed not in components:
            pair.update(status='unavailable', reason='Both zero-shift and shifted components are required.')
        else:
            p0, a0 = components[base]
            p1, a1 = components[changed]
            pair.update(status='available',
                        delta_objective=by_key[changed]['final_objective'] - by_key[base]['final_objective'],
                        delta_a_rmse=by_key[changed]['a_rmse_to_saved_ground_truth'] - by_key[base]['a_rmse_to_saved_ground_truth'],
                        p_rmse_between_initializations=float(np.sqrt(np.mean((p1-p0)**2))),
                        a_rmse_between_initializations=float(np.sqrt(np.mean((a1-a0)**2))))
        comparisons.append(pair)
    out = output_dir if output_dir is not None else run_dir / 'analysis'
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / 'runs.csv', rows, ['signal', 'phase_shift_samples', 'status', *METRICS, 'error'])
    write_csv(out / 'phase_comparison.csv', comparisons,
              ['signal', 'phase_shift_samples', 'status', 'delta_objective', 'delta_a_rmse',
               'p_rmse_between_initializations', 'a_rmse_between_initializations', 'reason'])
    complete = sum(r['status'] == 'completed_fixed_budget' for r in rows)
    text = ['# 原模型相位诊断汇总', '',
            f'预期 {len(expected)} 次运行；完成固定预算且输出齐备 {complete} 次；其余 {len(expected)-complete} 次。', '',
            f"- 代码提交：`{manifest.get('git_commit')}`",
            f"- 原模型 SHA-256：`{manifest.get('source_sha256')}`",
            f"- 设备：`{manifest.get('device')}`（{manifest.get('device_name')}）",
            f"- 每次迭代预算：{config['epochs']}。",
            '', '## 每次运行', '',
            '| 信号 | 起点偏移/点 | 状态 | 最终目标 | A 真值 RMSE | 秒 |',
            '|---|---:|---|---:|---:|---:|']
    for r in rows:
        text.append('| ' + ' | '.join([r['signal'], format_number(r['phase_shift_samples']), r['status'],
                    format_number(r.get('final_objective')), format_number(r.get('a_rmse_to_saved_ground_truth')),
                    format_number(r.get('seconds'))]) + ' |')
    text += ['', '## 相对零偏移的变化', '',
             '| 信号 | 偏移/点 | Δ目标 | ΔA 真值 RMSE | P 分量间 RMSE | A 分量间 RMSE |',
             '|---|---:|---:|---:|---:|---:|']
    for r in comparisons:
        text.append('| ' + ' | '.join([r['signal'], format_number(r['phase_shift_samples']),
                    *[format_number(r.get(k)) for k in ['delta_objective', 'delta_a_rmse',
                      'p_rmse_between_initializations', 'a_rmse_between_initializations']]]) + ' |')
    errors = [r for r in rows if r['status'] != 'completed_fixed_budget']
    if errors:
        text += ['', '## 失败与缺失', '']
        for r in errors:
            message = str(r.get('error', r['status'])).replace('\n', ' ')
            text.append(f"- {r['signal']} / 偏移 {r['phase_shift_samples']}：{message}")
    text += ['', '## 解释范围', '',
             'Δ 表示偏移初值减去零偏移初值。P/A 分量间 RMSE 衡量同一输入下不同初值的分解差异；P 的这一数值不是对真实背景的恢复误差。A 真值 RMSE 来自运行时保存的评价记录。', '',
             '达到固定迭代预算不等于已收敛。本诊断检验原模型的相位敏感性，不能据此估计随机初值总体方差、证明修改后模型稳定，或推断 F1/AUROC/AP 的变化。', '']
    (out / 'report.md').write_text('\n'.join(text))
    return out, len(expected) - complete


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_dir', type=Path)
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args()
    try:
        out, incomplete = summarize(args.run_dir.expanduser().resolve(),
                                    args.output_dir.expanduser().resolve() if args.output_dir else None)
    except (OSError, ValueError, KeyError, TypeError, EOFError, zipfile.BadZipFile) as error:
        print(f'Summary failed: {type(error).__name__}: {error}', file=sys.stderr)
        return 2
    print(f'Analysis: {out}\nIncomplete or failed runs: {incomplete}')
    return 1 if incomplete else 0


if __name__ == '__main__':
    raise SystemExit(main())

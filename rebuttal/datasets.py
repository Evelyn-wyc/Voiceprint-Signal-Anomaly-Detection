"""Versioned synthetic cases and interval annotations for revision experiments."""
import csv
from pathlib import Path
import numpy as np

SCENARIOS = ('nominal', 'asymmetric', 'double_peak', 'high_noise',
             'negative', 'bidirectional', 'normal')


def synthetic_case(scenario, seed, n=2000, period=200):
    """Return independent P/A/noise and event-support labels; no file side effects."""
    if scenario not in SCENARIOS or n < 4 * period:
        raise ValueError('Unknown scenario or fewer than four cycles.')
    bg = np.random.default_rng(seed)
    events = np.random.default_rng(10000 + seed)
    noise_rng = np.random.default_rng(20000 + seed)
    p = np.zeros(n, dtype=float)
    event_windows = []
    for k, onset in enumerate(range(0, n, period)):
        start = max(0, onset + int(bg.integers(-period // 20, period // 20 + 1)))
        duration = max(10, int(period * 0.5 * bg.uniform(0.92, 1.08)))
        end = min(n, start + duration)
        u = (np.arange(start, end) - start) / duration
        amplitude = 2.0 * bg.uniform(0.85, 1.15)
        if scenario == 'asymmetric':
            shape = np.sin(np.pi * u**2.2)
        elif scenario == 'double_peak':
            shape = np.sin(np.pi * u) * (0.55 + 0.45 * np.cos(4 * np.pi * (u - 0.25)))
            shape /= max(float(shape.max()), 1e-12)
        else:
            shape = np.sin(np.pi * u)
        p[start:end] += amplitude * shape
        event_windows.append((start, end))
    a = np.zeros(n)
    labels = np.zeros(n, dtype=bool)
    # Two separated disturbances; support labels are set before amplitude generation.
    positions = [int(n * 0.28 + events.integers(-period // 4, period // 4)),
                 int(n * 0.72 + events.integers(-period // 4, period // 4))]
    if scenario != 'normal':
        for j, center in enumerate(positions):
            negative = scenario == 'negative' or (scenario == 'bidirectional' and j == 1)
            if negative:
                # A local energy drop inside a normal event, with P+A remaining nonnegative.
                k = min(len(event_windows) - 1, max(0, center // period))
                lo, hi = event_windows[k]
                start, end = lo + (hi-lo)//4, lo + 3*(hi-lo)//4
                a[start:end] -= 0.7 * p[start:end]
            else:
                width = int(events.integers(max(8, period//5), max(9, period//3)))
                start, end = max(0, center-width//2), min(n, center+width//2)
                u = (np.arange(end-start) + 0.5) / (end-start)
                a[start:end] += events.uniform(1.5, 2.5) * np.sin(np.pi*u)**2
                # A short spike within the second positive event changes anomaly shape.
                if j == 1:
                    a[center-2:center+3] += 1.5
            labels[start:end] = True
    sigma = 0.4 if scenario == 'high_noise' else 0.2
    noise = noise_rng.normal(0.0, sigma, n)
    return {'X': p+a+noise, 'P_true': p, 'A_true': a, 'noise': noise, 'labels': labels}


def load_real_labels(path, n):
    y = np.zeros(n, dtype=bool)
    if path.suffix == '.npy':
        raw = np.load(path, allow_pickle=False)
        if raw.shape != (n,) or not np.isin(raw, [0, 1]).all():
            raise ValueError(f'Expected binary point labels of length {n}: {path}')
        return raw.astype(bool)
    with path.open(newline='', encoding='utf-8-sig') as handle:
        header = handle.readline()
        handle.seek(0)
        delimiter = '\t' if header.count('\t') > header.count(',') else ','
        reader = csv.DictReader(handle, delimiter=delimiter)
        fields = {k.strip() for k in (reader.fieldnames or [])}
        if not ({'X', 'Width'} <= fields or {'start', 'end'} <= fields):
            raise ValueError(f'Expected X/Width or start/end columns: {path}')
        for row in reader:
            row = {k.strip(): v for k, v in row.items()}
            left = float(row['X']) if 'X' in row else float(row['start'])
            right = left + float(row['Width']) if 'X' in row else float(row['end'])
            if not np.isfinite([left, right]).all() or not 0 <= left <= right <= n:
                raise ValueError(f'Invalid annotation interval {left}, {right}: {path}')
            y[int(left):int(right)] = True
    return y


def find_real_paths(root, signal_dir=None, label_dir=None):
    def choose(explicit, choices):
        if explicit:
            p = Path(explicit).expanduser().resolve()
            if not p.is_dir():
                raise FileNotFoundError(p)
            return p
        for choice in choices:
            p = root / choice
            if p.is_dir():
                return p
        raise FileNotFoundError(f'None found under {root}: {choices}')
    signals = choose(signal_dir, ['data/processed/241230_vector_npy', '241230_vector_npy'])
    labels = choose(label_dir, ['data/annotations/241230_gt', '241230_gt',
                               'data/annotations/241230_gt_xwidth', '241230_gt_xwidth'])
    return signals, labels


def real_cases(root, expected_count, expected_length, signal_dir=None, label_dir=None):
    signals, labels = find_real_paths(root, signal_dir, label_dir)
    paths = sorted(signals.glob('*.npy'))
    if len(paths) != expected_count:
        raise ValueError(f'Expected {expected_count} real signals, found {len(paths)} in {signals}')
    cases = []
    for path in paths:
        x = np.load(path, allow_pickle=False)
        if x.shape != (expected_length,) or np.iscomplexobj(x) or not np.isfinite(x).all():
            raise ValueError(f'Invalid real signal: {path}')
        options = [labels / f'Overlay Elements of {path.stem}.csv',
                   labels / f'{path.stem}_xwidth.npy', labels / f'{path.stem}_xwidth.csv',
                   labels / f'Overlay Elements of {path.stem}_xwidth.csv']
        label = next((p for p in options if p.is_file()), None)
        if label is None:
            raise FileNotFoundError(f'No interval/point labels for {path.stem} in {labels}')
        y = load_real_labels(label, len(x))
        cases.append({'id': path.stem, 'dataset': 'real', 'scenario': 'real',
                      'group': path.stem.rsplit('_', 1)[0], 'seed': None,
                      'X': x.astype(float), 'labels': y,
                      'source_paths': [str(path), str(label)]})
    return cases

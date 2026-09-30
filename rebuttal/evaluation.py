"""Pointwise metrics and a fixed label-free robust threshold protocol."""
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
from baselines import mad_scale


def standardized_scores(raw, x):
    raw, x = np.asarray(raw, dtype=float), np.asarray(x, dtype=float)
    if raw.shape != x.shape or not np.isfinite(raw).all() or not np.isfinite(x).all():
        raise ValueError('Finite equally sized scores and inputs required.')
    center = float(np.median(raw))
    noise = mad_scale(np.diff(x))/np.sqrt(2)
    floor = 1e-12 * max(1.0, float(np.max(np.abs(x))))
    scale = max(mad_scale(raw), noise, floor)
    return (raw-center)/scale, {'score_center': center, 'score_scale': scale,
                               'input_noise_scale': noise, 'raw_score_mad': mad_scale(raw)}


def binary_metrics(labels, scores, predictions):
    y, pred = np.asarray(labels, dtype=bool), np.asarray(predictions, dtype=bool)
    tp, fp = int(np.sum(y & pred)), int(np.sum(~y & pred))
    fn, tn = int(np.sum(y & ~pred)), int(np.sum(~y & ~pred))
    # Retain true zero scores. AUROC requires two classes, AP requires positives.
    return {'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn, 'positive_points': int(y.sum()),
            'precision': tp/(tp+fp) if tp+fp else 0.0,
            'recall': tp/(tp+fn) if tp+fn else None,
            'f1': 2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.0,
            'fpr': fp/(fp+tn) if fp+tn else None,
            'auroc': float(roc_auc_score(y, scores)) if y.any() and (~y).any() else None,
            'ap': float(average_precision_score(y, scores)) if y.any() else None}


def evaluate(raw, x, labels, multipliers):
    if np.asarray(labels).shape != np.asarray(x).shape:
        raise ValueError('Label shape differs from input.')
    z, params = standardized_scores(raw, x)
    rows = []
    for direction in ['positive', 'two_sided']:
        score = z if direction == 'positive' else np.abs(z)
        for multiplier in multipliers:
            rows.append({'direction': direction, 'rule': 'robust', 'multiplier': multiplier,
                         **params, **binary_metrics(labels, score, score > multiplier)})
    rows.append({'direction': 'positive', 'rule': 'zero', 'multiplier': 0.0,
                 **params, **binary_metrics(labels, raw, raw > 0)})
    return rows, z

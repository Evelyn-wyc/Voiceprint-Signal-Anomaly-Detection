"""Small synthetic functional/numerical checks; no paper-data fitting."""
import copy
import json
from pathlib import Path
import sys
import unittest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from baselines import QPGPCovariance, cycle_rpca, estimate_period, fit_baseline
from datasets import synthetic_case
from evaluation import evaluate
from experiments import DEFAULT_CONFIG, validate_config


class ProtocolTests(unittest.TestCase):
    def setUp(self):
        self.config = json.loads(DEFAULT_CONFIG.read_text())

    def test_qpgp_matches_dense_for_complete_and_partial_cycles(self):
        rng = np.random.default_rng(3)
        for n in [48, 53]:
            p, delta, theta, omega = 12, 0.35, 1.4, 0.8
            i = np.arange(n)
            k = np.exp(-(theta*np.sin(np.pi*(i[:, None]-i)/p))**2) * omega**np.abs(i[:, None]//p-i//p)
            c = k+delta**2*np.eye(n)
            y = rng.normal(size=n)
            solver = QPGPCovariance(n, p, delta, theta, omega)
            np.testing.assert_allclose(solver.solve(y), np.linalg.solve(c, y), rtol=1e-9, atol=1e-9)
            self.assertAlmostEqual(solver.logdet, np.linalg.slogdet(c)[1], places=8)
            nll, beta, sigma2, alpha = solver.likelihood(y)
            np.testing.assert_allclose(y-delta**2*alpha, beta+k@np.linalg.solve(c, y-beta), atol=1e-9)
            q = np.linalg.solve(c, y-beta)
            dense = .5*(np.linalg.slogdet(c)[1]+n*(np.log((y-beta)@q/n)+1+np.log(2*np.pi)))
            self.assertAlmostEqual(nll, dense, places=7)

    def test_synthetic_truth_and_signed_labels(self):
        for scenario in self.config['simulation']['scenarios']:
            a = synthetic_case(scenario, 2)
            b = synthetic_case(scenario, 2)
            np.testing.assert_array_equal(a['X'], b['X'])
            np.testing.assert_allclose(a['X'], a['P_true']+a['A_true']+a['noise'])
            self.assertFalse(np.any((np.abs(a['A_true']) > 0) & ~a['labels']))
            if scenario == 'negative':
                self.assertLess(a['A_true'].min(), 0)
                self.assertEqual(a['A_true'].max(), 0)
            if scenario == 'bidirectional':
                self.assertLess(a['A_true'].min(), 0)
                self.assertGreater(a['A_true'].max(), 0)
            if scenario == 'normal':
                self.assertFalse(a['labels'].any())

    def test_threshold_scores_keep_zeros_and_undefined_metrics(self):
        x = np.sin(np.arange(40))*.1
        labels = np.zeros(40, bool)
        labels[20] = True
        raw = np.zeros(40)
        raw[20] = -2
        rows, _ = evaluate(raw, x, labels, [2., 3., 4.])
        pos = next(r for r in rows if r['direction']=='positive' and r['multiplier']==3)
        two = next(r for r in rows if r['direction']=='two_sided' and r['multiplier']==3)
        self.assertEqual(pos['f1'], 0)
        self.assertEqual(two['f1'], 1)
        self.assertEqual(two['ap'], 1)
        normal, _ = evaluate(raw, x, np.zeros(40, bool), [3.])
        self.assertIsNone(normal[0]['auroc'])
        self.assertIsNone(normal[0]['ap'])
        self.assertEqual(normal[0]['f1'], 0)
        scaled, _ = evaluate(raw*4, x*4, labels, [2., 3., 4.])
        self.assertEqual([r['f1'] for r in rows], [r['f1'] for r in scaled])

    def test_period_sample_units(self):
        x = np.sin(2*np.pi*np.arange(800)/80)
        period, _ = estimate_period(x, self.config['baselines'])
        self.assertLessEqual(abs(period-80), 2)

    def test_rpca_reconstruction_and_sparse_recovery(self):
        p = np.tile(np.sin(2*np.pi*np.arange(20)/20), 12)
        x = p.copy()
        x[[31, 110, 202]] += 5
        low, sparse, meta = cycle_rpca(x, 20, 500, 1e-7)
        self.assertLess(np.linalg.norm(x-low-sparse)/np.linalg.norm(x), 1e-6)
        self.assertLess(np.sqrt(np.mean((low-p)**2)), .1)
        self.assertGreater(sparse[31], 4)
        self.assertEqual(meta['solver_status'], 'converged')

    def test_baselines_return_continuous_full_length_outputs(self):
        x = np.sin(2*np.pi*np.arange(97)/16)+.1*np.cos(np.arange(97))
        settings = copy.deepcopy(self.config['baselines'])
        settings.update(qpgp_period_factors=[1.], qpgp_maxiter=2, rpca_maxiter=20)
        for method in ['STL', 'VMD', 'QPGP', 'RPCA']:
            p, a, info = fit_baseline(method, x, settings)
            self.assertEqual(p.shape, x.shape)
            self.assertEqual(a.shape, x.shape)
            self.assertTrue(np.isfinite(a).all())
            self.assertGreater(np.ptp(p), .01)
            self.assertGreater(np.count_nonzero(a), len(x)//2)

    def test_configuration_validation(self):
        validate_config(self.config)
        for path, value in [(('qpad', 'simulation', 'epochs'), 0), (('initialization', 'duration_fraction'), 2),
                            (('evaluation', 'multipliers'), [3., 3.])]:
            c = copy.deepcopy(self.config)
            target = c
            for key in path[:-1]:
                target = target[key]
            target[path[-1]] = value
            with self.assertRaises(ValueError):
                validate_config(c)


if __name__ == '__main__':
    unittest.main()

"""Numerical, selection and artifact tests for coefficient sensitivity."""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from datasets import synthetic_case
from experiments import HERE, make_manifest, run_task, tasks_for, write_json
from period_prior import environment
from regularization import DEFAULT_CONFIG, GROUPS, prepare, select_inputs, task_config, validate_config, variants


class RegularizationTests(unittest.TestCase):
    def test_group_scaling_changes_the_expected_loss_terms(self):
        import torch
        from models import simulation
        cfg = json.loads(DEFAULT_CONFIG.read_text())
        validate_config(cfg)
        model = simulation.Periodic(3, 360)
        with torch.no_grad():
            model.t_k.copy_(torch.tensor([0., 120., 240.]))
            model.T_active.copy_(torch.tensor([70., 80., 60.]))
            model.M_independent.copy_(torch.tensor([1., 2., .5]))
            model.A.copy_(torch.linspace(-1, 1, 360))
        opt = simulation.JointOptimizer.__new__(simulation.JointOptimizer)
        opt.periodic_model, opt.X = model, torch.ones(360)
        base = cfg['qpad']['simulation']['parameters']
        def loss(params):
            for key, value in params.items():
                setattr(opt, key, value)
            return float(opt.compute_loss().detach())
        original_loss = loss(base)
        self.assertEqual(len(variants()), 7)
        for variant in variants():
            params = base.copy()
            for key in GROUPS.get(variant['parameter_group'], ()):
                params[key] *= variant['factor']
            task = {'dataset': 'simulation', **variant, 'parameters': params}
            scaled = task_config(cfg, task)['qpad']['simulation']['parameters']
            self.assertEqual(scaled, params)
            if variant['parameter_group'] != 'default':
                removed = base.copy()
                for key in GROUPS[variant['parameter_group']]:
                    removed[key] = 0
                expected = original_loss+(variant['factor']-1)*(original_loss-loss(removed))
                self.assertAlmostEqual(loss(scaled), expected, places=5)
        self.assertEqual(cfg, json.loads(DEFAULT_CONFIG.read_text()))
        bad = copy.deepcopy(cfg); bad['qpad']['real']['parameters']['psi'] = -1
        with self.assertRaises(ValueError):
            validate_config(bad)

    def test_segment_selection_uses_numeric_order_and_identifiers(self):
        cfg = json.loads(DEFAULT_CONFIG.read_text())
        cfg['selection'].update(simulation_seeds=[0], expected_recordings=1)
        records = [{'id': 'sim_nominal_s0', 'dataset': 'simulation', 'scenario': 'nominal', 'seed': 0, 'group': 'seed_0'},
                   *[{'id': f'real_recording_{i}', 'dataset': 'real', 'scenario': 'real', 'seed': None,
                      'group': 'recording', 'labels': [i == 10], 'f1': i/10} for i in (10, 2)]]
        selected = select_inputs(records, cfg)
        self.assertEqual([r['id'] for r in selected], ['sim_nominal_s0', 'real_recording_2'])
        for row in records:
            row.update(labels=[True], f1=1.)
        self.assertEqual([r['id'] for r in select_inputs(records[::-1], cfg)], ['sim_nominal_s0', 'real_recording_2'])

    def test_run_reuse_resume_metrics_and_corruption(self):
        import torch
        legacy = json.loads((HERE/'configs/necessary.json').read_text())
        cfg = json.loads(DEFAULT_CONFIG.read_text())
        cfg['selection'].update(simulation_seeds=[0], expected_recordings=1)
        for config in (legacy, cfg):
            config['qpad']['simulation'].update(epochs=2, K=3)
            config['qpad']['real'].update(epochs=2, K=3)
        legacy['simulation'].update(length=128, period=32, seeds=[0], scenarios=['nominal'])
        sim = {'id': 'sim_nominal_s0', 'dataset': 'simulation', 'scenario': 'nominal',
               'seed': 0, 'group': 'seed_0', 'source_paths': [], **synthetic_case('nominal', 0, 128, 32)}
        real = {'id': 'real_recording_1', 'dataset': 'real', 'scenario': 'real',
                'seed': None, 'group': 'recording', 'source_paths': [],
                'X': 1+np.sin(np.arange(512)/20), 'labels': np.zeros(512, dtype=bool)}
        cases = [sim, real]
        env_vars = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MPLCONFIGDIR='/tmp/qpad-regularization-test-mpl')
        def command(args, expected=0):
            result = subprocess.run([sys.executable, str(HERE/'regularization.py'), *args],
                                    env=env_vars, text=True, capture_output=True)
            self.assertEqual(result.returncode, expected, result.stdout+result.stderr)
            return result
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); source, output = root/'source', root/'output'
            (source/'inputs').mkdir(parents=True); (source/'fits').mkdir()
            _, env = environment('cpu')
            manifest = make_manifest(legacy, ['B'], cases, [], env)
            for case in cases:
                np.savez_compressed(source/'inputs'/f'{case["id"]}.npz',
                                    **{k: v for k, v in case.items() if isinstance(v, np.ndarray)})
                task = {'id': case['id']+'__QPAD__default', 'case_id': case['id'], 'method': 'QPAD', 'initialization_seed': None}
                manifest['tasks'].append(task)
                row = run_task(task, case, legacy, torch.device('cpu'), source)
                write_json(source/'fits'/f'{task["id"]}.json', row)
            write_json(source/'manifest.json', manifest)
            path = root/'config.json'; write_json(path, cfg)
            args = ['--source', str(source), '--output', str(output), '--config', str(path), '--device', 'cpu']
            command(['check', *args])
            command(['run', *args])
            summary = json.loads((output/'analysis/summary.json').read_text())
            self.assertEqual((summary['completed'], summary['reused'], summary['fitted']), (14, 2, 12))
            real_group = next(r for r in summary['groups'] if r['dataset']=='real' and r['variant']=='default')
            self.assertEqual(real_group['ap']['n'], 0)
            self.assertEqual(real_group['normal_fpr']['n'], 1)
            self.assertTrue((output/'analysis/figures/real.pdf').is_file())
            self.assertEqual((source/'fits/sim_nominal_s0__QPAD__default.npz').read_bytes(),
                             (output/'fits/sim_nominal_s0__default.npz').read_bytes())
            events = (output/'events.jsonl').read_bytes()
            command(['run', *args, '--resume'])
            self.assertEqual(events, (output/'events.jsonl').read_bytes())
            command(['run', *args], expected=2)
            corrupt = output/'fits/sim_nominal_s0__lambda_half.json'
            original = corrupt.read_text(); row=json.loads(original)
            row['metrics'][0]['f1'] = .123456789
            write_json(corrupt, row)
            command(['summarize', '--output', str(output)], expected=1)
            command(['run', *args, '--resume'], expected=2)
            corrupt.write_text(original)
            command(['summarize', '--output', str(output)])
            row=json.loads(original); row.update(status='failed', error='test interruption')
            write_json(corrupt, row)
            command(['run', *args, '--resume'])
            self.assertEqual(len((output/'events.jsonl').read_text().splitlines()), 15)
            changed = copy.deepcopy(cfg); changed['qpad']['real']['parameters']['psi'] *= 2
            with self.assertRaises(ValueError):
                prepare(source, changed)
            bad_input = source/'inputs/sim_nominal_s0.npz'
            with np.load(bad_input) as z:
                values = {k:z[k].copy() for k in z.files}
            values['X'][0] += 1
            np.savez_compressed(bad_input, **values)
            with self.assertRaises(ValueError):
                prepare(source, cfg)


if __name__ == '__main__':
    unittest.main()

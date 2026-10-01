"""Small numerical and artifact checks for the period-initialization study."""
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
from experiments import DEFAULT_CONFIG as LEGACY_CONFIG, HERE
from period_prior import DEFAULT_CONFIG, choose_initialization, explicit_state, fit_config, prepare, validate_config
from qpad_fit import fit_qpad


class PeriodPriorTests(unittest.TestCase):
    def test_plan_and_observation_based_initialization(self):
        config = json.loads(DEFAULT_CONFIG.read_text())
        validate_config(config)
        cases, tasks = prepare(config, {})
        self.assertEqual(len(cases), 9)
        self.assertEqual(len(tasks), 18)
        self.assertEqual({c['seed'] for c in cases}, {0, 1, 2})
        x = np.sin(2*np.pi*np.arange(800)/80)
        choice = choose_initialization(x, 'input_estimated', config)
        self.assertLessEqual(abs(choice['period_estimate']-80), 2)
        self.assertEqual(choice['K'], 10)
        self.assertLessEqual(abs(choice['initial_duration']-40), 1)
        self.assertEqual(choose_initialization(x, 'fixed', config)['initial_duration'], 100)
        invalid = copy.deepcopy(config)
        invalid['simulation']['seeds'] = [0, 0]
        with self.assertRaises(ValueError):
            validate_config(invalid)

    def test_explicit_initialization_preserves_archived_adam_update(self):
        import torch
        from models import simulation
        config = json.loads(DEFAULT_CONFIG.read_text())
        config['qpad']['simulation']['epochs'] = 1
        task = {'K': 5, 'initial_duration': 9.5}
        cfg = fit_config(config, task)
        x = np.sin(np.arange(96)/5)+1
        state = explicit_state(len(x), task)
        model = simulation.Periodic(5, len(x))
        with torch.no_grad():
            for key, value in state.items():
                getattr(model, key).copy_(torch.tensor(value, dtype=torch.float32))
        original = simulation.JointOptimizer.__new__(simulation.JointOptimizer)
        original.periodic_model, original.X = model, torch.tensor(x, dtype=torch.float32)
        for key, value in cfg['qpad']['simulation']['parameters'].items():
            setattr(original, key, value)
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg['qpad']['learning_rate'])
        optimizer.zero_grad(); (original.compute_loss()*cfg['qpad']['loss_scale']).backward(); optimizer.step()
        p, a, meta, arrays = fit_qpad(x, 'simulation', cfg, torch.device('cpu'), initial_state=state)
        with torch.no_grad():
            old_p, old_a = model()
        np.testing.assert_array_equal(p, old_p.numpy())
        np.testing.assert_array_equal(a, old_a.numpy())
        for key, value in state.items():
            np.testing.assert_array_equal(arrays['initial_'+key], np.asarray(value, np.float32))
        with self.assertRaises(ValueError):
            fit_qpad(x, 'simulation', cfg, torch.device('cpu'), 101, initial_state=state)

    def test_reuse_resume_reporting_and_corruption(self):
        legacy = json.loads(LEGACY_CONFIG.read_text())
        legacy['simulation'].update(length=128, period=32, seeds=[0], scenarios=['nominal'])
        legacy['qpad']['simulation'].update(K=4, epochs=2)
        legacy['baselines'].update(qpgp_period_factors=[1.0], qpgp_maxiter=2, rpca_maxiter=20)
        legacy['evaluation']['bootstrap_repeats'] = 10
        config = json.loads(DEFAULT_CONFIG.read_text())
        config['simulation'].update(length=128, periods=[24, 32], seeds=[0])
        config['qpad'] = {key: copy.deepcopy(legacy['qpad'][key]) for key in config['qpad']}
        env = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MPLCONFIGDIR='/tmp/qpad-period-test-mpl')
        def run(command, expected=0):
            result = subprocess.run(command, capture_output=True, text=True, env=env)
            self.assertEqual(result.returncode, expected, result.stdout+result.stderr)
            return result
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            old_config, cfg = base/'old.json', base/'new.json'
            old_config.write_text(json.dumps(legacy)); cfg.write_text(json.dumps(config))
            source, output = base/'source', base/'period'
            run([sys.executable, str(HERE/'experiments.py'), 'run', '--config', str(old_config),
                 '--parts', 'B', '--device', 'cpu', '--output', str(source)])
            args = ['--config', str(cfg), '--device', 'cpu', '--reuse-from', str(source), '--output', str(output)]
            run([sys.executable, str(HERE/'period_prior.py'), 'check', *args])
            command = [sys.executable, str(HERE/'period_prior.py'), 'run', *args]
            run(command)
            summary = json.loads((output/'analysis/summary.json').read_text())
            self.assertEqual((summary['completed'], summary['reused'], summary['fitted']), (4, 1, 3))
            self.assertTrue((output/'analysis/figures/summary.pdf').is_file())
            self.assertEqual((source/'fits/sim_nominal_s0__QPAD__default.npz').read_bytes(),
                             (output/'fits/nominal_p32_s0__fixed.npz').read_bytes())
            events = (output/'events.jsonl').read_bytes()
            run(command+['--resume'])
            self.assertEqual(events, (output/'events.jsonl').read_bytes())
            bad_source_config = json.loads((source/'manifest.json').read_text())
            bad_source_config['config']['qpad']['learning_rate'] = .02
            source_manifest = (source/'manifest.json').read_text()
            (source/'manifest.json').write_text(json.dumps(bad_source_config))
            result = run([sys.executable, str(HERE/'period_prior.py'), 'check', *args], 2)
            self.assertIn('Reuse rejected', result.stderr)
            (source/'manifest.json').write_text(source_manifest)
            config['evaluation']['multipliers'] = [1., 2., 3., 4.]
            cfg.write_text(json.dumps(config))
            result = run(command+['--resume'], 2)
            self.assertIn('Reuse rejected', result.stderr)
            (output/'fits/nominal_p24_s0__input_estimated.npz').write_bytes(b'broken')
            run([sys.executable, str(HERE/'period_prior.py'), 'summarize', '--output', str(output)], 1)
            self.assertEqual(json.loads((output/'analysis/summary.json').read_text())['incomplete'], 1)


if __name__ == '__main__':
    unittest.main()

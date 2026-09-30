"""Artifact, resume, label parsing, and original-update fidelity checks."""
import copy
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from datasets import load_real_labels
from experiments import DEFAULT_CONFIG, HERE, ROOT
from qpad_fit import fit_qpad


class RunnerTests(unittest.TestCase):
    def test_interval_delimiters_empty_and_bounds(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'labels.csv'
            path.write_text('Index\tX\tWidth\n')
            self.assertFalse(load_real_labels(path, 20).any())
            path.write_text('Index, X, Width\n1,2.2,3.5\n')
            y = load_real_labels(path, 20)
            np.testing.assert_array_equal(np.flatnonzero(y), [2, 3, 4])
            path.write_text('X,Width\n19,4\n')
            with self.assertRaises(ValueError):
                load_real_labels(path, 20)

    def test_qpad_one_step_matches_frozen_original_update(self):
        import torch
        from models import simulation, real
        config = json.loads(DEFAULT_CONFIG.read_text())
        x = np.sin(np.arange(96)/10.) + 1
        torch.set_num_threads(1)
        for kind, module in [('simulation', simulation), ('real', real)]:
            config['qpad'][kind]['epochs'] = 1
            config['qpad'][kind]['K'] = 4
            model = module.Periodic(4, len(x))
            original = module.JointOptimizer.__new__(module.JointOptimizer)
            original.periodic_model = model
            original.device = torch.device('cpu')
            original.X = torch.tensor(x, dtype=torch.float32)
            for key, value in config['qpad'][kind]['parameters'].items():
                setattr(original, key, value)
            optimizer = torch.optim.Adam(model.parameters(), lr=.01)
            optimizer.zero_grad()
            (original.compute_loss()*10).backward()
            optimizer.step()
            p, a, meta, arrays = fit_qpad(x, kind, config, torch.device('cpu'))
            with torch.no_grad():
                old_p, old_a = model()
            np.testing.assert_array_equal(p, old_p.numpy())
            np.testing.assert_array_equal(a, old_a.numpy())
            self.assertEqual(arrays['trace'][-1, 0], 1)

    def test_run_resume_and_corruption_detection(self):
        config = json.loads(DEFAULT_CONFIG.read_text())
        config['simulation'].update(length=96, period=24, seeds=[0], scenarios=['nominal', 'asymmetric'])
        config['initialization']['seeds'] = [101]
        config['qpad']['simulation'].update(epochs=2, K=4)
        config['baselines'].update(qpgp_period_factors=[1.0], qpgp_maxiter=2, rpca_maxiter=20)
        config['evaluation']['bootstrap_repeats'] = 20
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            cfg = folder/'test.json'
            cfg.write_text(json.dumps(config))
            output = folder/'result'
            command = [sys.executable, str(HERE/'experiments.py'), 'run', '--config', str(cfg),
                       '--parts', 'B', 'C', '--device', 'cpu', '--output', str(output)]
            env = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
            first = subprocess.run(command, capture_output=True, text=True, env=env)
            self.assertEqual(first.returncode, 0, first.stdout+first.stderr)
            summary = json.loads((output/'analysis/summary.json').read_text())
            self.assertEqual(summary['expected'], 12)
            self.assertEqual(summary['completed'], 12)
            events = (output/'events.jsonl').read_bytes()
            with (output/'analysis/initialization.csv').open() as stream:
                initial = list(csv.DictReader(stream))
            self.assertEqual(len(initial), 2)
            second = subprocess.run(command+['--resume'], capture_output=True, text=True, env=env)
            self.assertEqual(second.returncode, 0, second.stdout+second.stderr)
            self.assertEqual(events, (output/'events.jsonl').read_bytes())
            config['qpad']['simulation']['epochs'] = 3
            cfg.write_text(json.dumps(config))
            changed = subprocess.run(command+['--resume'], capture_output=True, text=True, env=env)
            self.assertNotEqual(changed.returncode, 0)
            self.assertIn('Resume rejected', changed.stderr)
            npz = next((output/'fits').glob('*.npz'))
            npz.write_bytes(b'corrupt')
            summary_command = [sys.executable, str(HERE/'experiments.py'), 'summarize', '--output', str(output)]
            broken = subprocess.run(summary_command, capture_output=True, text=True, env=env)
            self.assertEqual(broken.returncode, 1, broken.stdout+broken.stderr)
            self.assertEqual(json.loads((output/'analysis/summary.json').read_text())['incomplete'], 1)


if __name__ == '__main__':
    unittest.main()

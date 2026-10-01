"""Numerical checks for the constrained model and experiment artifacts."""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import torch

HERE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(HERE))
from theory_model import Problem, ProximalQP, differences, fit_theory, isotonic, local_envelope
from theory_experiments import validate_config


def small_config():
    c=json.loads((HERE/'configs/theory_probe.json').read_text())
    c['simulation'].update(length=64,period=16,seeds=[12345],scenarios=['nominal'])
    c['real']['enabled']=False
    c['qpad']['simulation'].update(K=4,epochs=3,initial_duration=8)
    c['initialization'].update(signal='sim_nominal_s12345',seeds=[],modes=[])
    c['model']['alignment_points']=8
    c['optimization'].update(check_every=1,consecutive_checks=2)
    return c


class TheoryNumerics(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.c=small_config()

    def test_envelope_support_and_c1_joins(self):
        w=self.c['model']['boundary_fraction']
        for boundary in [0.,w,1-w,1.]:
            u=torch.tensor([boundary-1e-8,boundary+1e-8],dtype=torch.float64,requires_grad=True)
            y=local_envelope(u,w);g=torch.autograd.grad(y.sum(),u)[0]
            self.assertLess(abs(float((y[1]-y[0]).detach())),1e-6)
            self.assertLess(abs(float(g[1]-g[0])),1e-5)
        u=torch.tensor([-10.,-.1,.25,.5,.75,1.1,10.],dtype=torch.float64)
        y=local_envelope(u,w).numpy()
        np.testing.assert_array_equal(y[[0,1,5,6]],0.)
        np.testing.assert_allclose(y[2:5],.5*np.sin(np.pi*u.numpy()[2:5]),atol=1e-15)

    def test_prox_closed_forms_and_isotonic(self):
        p=self.c['proximal'];y=np.array([-3.,-.4,.2,2.])
        qp=ProximalQP(4,[],[],np.ones(4),np.zeros(4),np.full(4,10.),False,p)
        z,_=qp.solve(y,.5)
        np.testing.assert_allclose(z,np.maximum(y-.5,0),atol=1e-7)
        qp=ProximalQP(2,[differences(2)],[np.array([1.])],np.zeros(2),np.full(2,-10.),np.full(2,10.),False,p)
        z,_=qp.solve(np.array([0.,3.]),.5)
        np.testing.assert_allclose(z,[.5,2.5],atol=1e-7)
        qp=ProximalQP(3,[],[],np.zeros(3),np.full(3,-10.),np.full(3,10.),True,p)
        z,_=qp.solve(np.array([3.,1.,2.]),1.)
        np.testing.assert_allclose(z,[2.,2.,2.],atol=1e-7)
        np.testing.assert_allclose(isotonic([3.,1.,2.],[1.,3.,1.]),[1.5,1.5,2.])

    def test_prox_nonexpansiveness_with_tv_and_second_difference(self):
        rng=np.random.default_rng(4);n=9;p=self.c['proximal']
        qp=ProximalQP(n,[differences(n),differences(n,2)],[np.full(n-1,.3),np.full(n-2,.15)],np.zeros(n),np.full(n,-2.),np.full(n,2.),False,p)
        a,b=rng.normal(size=(2,n));pa,_=qp.solve(a,1.);pb,_=qp.solve(b,1.)
        self.assertLessEqual(np.linalg.norm(pa-pb),np.linalg.norm(a-b)+1e-7)
        self.assertLessEqual(float((pa-pb)@(pa-pb)),float((pa-pb)@(a-b))+1e-7)

    def test_smooth_gradient_includes_moving_alignment(self):
        problem=Problem(np.linspace(-.1,2,64),'simulation',self.c,torch.device('cpu'))
        state=problem.initial('local',9)
        for b in ['M','t','T']:
            analytic=torch.autograd.grad(problem.smooth(state),state[b])[0].detach().numpy()
            for j in [0,2]:
                values=[]
                for sign in [-1,1]:
                    trial=dict(state);trial[b]=state[b].detach().clone();trial[b][j]+=sign*1e-6
                    values.append(float(problem.smooth(trial).detach()))
                self.assertAlmostEqual(analytic[j],(values[1]-values[0])/2e-6,delta=2e-5)

    def test_descent_bounds_and_finite_budget(self):
        from datasets import synthetic_case
        x=synthetic_case('nominal',12345,64,16)['X']
        for mode in ['default','feasible_random']:
            p,a,meta,arrays=fit_theory(x,'simulation',self.c,'cpu','palm',mode,101)
            self.assertTrue(np.isfinite(p).all())
            self.assertTrue(np.all(a>=0))
            self.assertTrue(np.all(np.diff(arrays['final_t'])>=0))
            self.assertEqual(meta['termination'],'iteration_budget')
            self.assertTrue(np.all(np.diff(arrays['objective_history'])<1e-8))
            audit=arrays['block_audit']
            self.assertTrue(np.all(audit[:,3]<=audit[:,2]-audit[:,8]+audit[:,7]+1e-12))
        p,a,meta,arrays=fit_theory(x,'simulation',self.c,'cpu','amsgrad')
        self.assertEqual(meta['iterations'],3)
        self.assertEqual(meta['final_diagnostics']['constraint_violation'],0.)

    def test_config_rejects_invalid_theory_settings(self):
        validate_config(self.c)
        for section,key,value in [('model','duration_bounds_in_scale',[0,1]),('model','duration_bounds_in_scale',[.1,float('inf')]),('model','boundary_fraction',0),('optimization','amsgrad_decay_power',.4)]:
            c=copy.deepcopy(self.c);c[section][key]=value
            with self.assertRaises(ValueError):validate_config(c)

    def test_cli_resume_and_corruption(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);cfg=root/'config.json';cfg.write_text(json.dumps(self.c));output=root/'output'
            command=[sys.executable,str(HERE/'theory_experiments.py'),'run','--config',str(cfg),'--device','cpu','--output',str(output)]
            env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
            first=subprocess.run(command,capture_output=True,text=True,env=env)
            self.assertEqual(first.returncode,0,first.stdout+first.stderr)
            summary=json.loads((output/'analysis/summary.json').read_text())
            self.assertEqual(summary['completed'],2)
            events=(output/'events.jsonl').read_bytes()
            repeat=subprocess.run(command+['--resume'],capture_output=True,text=True,env=env)
            self.assertEqual(repeat.returncode,0,repeat.stdout+repeat.stderr)
            self.assertEqual(events,(output/'events.jsonl').read_bytes())
            changed=subprocess.run(command+['--resume','--max-iterations','4'],capture_output=True,text=True,env=env)
            self.assertEqual(changed.returncode,2)
            self.assertIn('Resume rejected',changed.stderr)
            next((output/'fits').glob('*.npz')).write_bytes(b'broken')
            broken=subprocess.run([sys.executable,str(HERE/'theory_experiments.py'),'summarize','--output',str(output)],capture_output=True,text=True,env=env)
            self.assertEqual(broken.returncode,1,broken.stdout+broken.stderr)
            self.assertEqual(json.loads((output/'analysis/summary.json').read_text())['failed_or_missing'],1)


if __name__=='__main__':unittest.main()

"""Constrained QPAD with C1 local envelopes and explicit composite optimization.

The loss is f + h: f contains reconstruction and aligned profile consistency;
h contains the original L1/TV penalties and convex constraints. PALM uses
checked proximal block steps. Projected AMSGrad is an empirical comparison.
"""
from dataclasses import dataclass
import math
import numpy as np
import scipy.sparse as sp
import torch

BLOCKS = ('M', 't', 'T', 'A')


def local_envelope(u, width=0.05):
    """Half sine on [width, 1-width], C1 cubic joins to zero at both ends."""
    s, d = 0.5*math.sin(math.pi*width), 0.5*math.pi*math.cos(math.pi*width)
    a = (3*s-width*d)/width**2
    b = (width*d-2*s)/width**3
    left = a*u**2+b*u**3
    v = 1-u
    right = a*v**2+b*v**3
    middle = 0.5*torch.sin(math.pi*u)
    inside = torch.where(u < width, left, torch.where(u > 1-width, right, middle))
    return torch.where((u >= 0) & (u <= 1), inside, torch.zeros_like(u))


def isotonic(y, weights=None):
    """Weighted Euclidean projection onto nondecreasing sequences (PAVA)."""
    weights = np.ones(len(y)) if weights is None else np.asarray(weights, float)
    levels, masses, counts = [], [], []
    for value, weight in zip(y, weights):
        levels.append(float(value)); masses.append(float(weight)); counts.append(1)
        while len(levels) > 1 and levels[-2] > levels[-1]:
            mass = masses[-2]+masses[-1]
            level = (levels[-2]*masses[-2]+levels[-1]*masses[-1])/mass
            count = counts[-2]+counts[-1]
            levels[-2:] = [level]; masses[-2:] = [mass]; counts[-2:] = [count]
    return np.repeat(levels, counts)


def differences(n, order=1):
    if order == 1:
        return sp.diags([-np.ones(n-1), np.ones(n-1)], [0, 1], shape=(n-1, n), format='csc')
    return sp.diags([np.ones(n-2), -2*np.ones(n-2), np.ones(n-2)], [0, 1, 2], shape=(n-2, n), format='csc')


class ProximalQP:
    """prox(alpha*h)(y), with h = linear*x + sum(w*|B*x|) + I_C.

    OSQP solves the sparse epigraph QP. Every returned solve must meet the
    declared primal/dual and feasibility tolerances. The caller records
    residuals for accepted updates and stationarity checks.
    """
    def __init__(self, n, operators, weights, linear, lower, upper, ordered, settings):
        import osqp
        self.n = n
        self.lower, self.upper = np.asarray(lower), np.asarray(upper)
        self.ordered = ordered
        self.linear = np.asarray(linear, float)
        self.B = sp.vstack(operators, format='csc') if operators else sp.csc_matrix((0, n))
        self.weights = np.concatenate(weights) if weights else np.empty(0)
        m = self.B.shape[0]
        pieces = [sp.hstack([sp.eye(n, format='csc'), sp.csc_matrix((n, m))], format='csc')]
        lows, highs = [self.lower], [self.upper]
        if m:
            eye = sp.eye(m, format='csc')
            pieces += [sp.hstack([self.B, -eye], format='csc'), sp.hstack([-self.B, -eye], format='csc')]
            lows += [np.full(m, -np.inf)]*2; highs += [np.zeros(m)]*2
        if ordered:
            pieces.append(sp.hstack([differences(n), sp.csc_matrix((n-1, m))], format='csc'))
            lows.append(np.zeros(n-1)); highs.append(np.full(n-1, np.inf))
        self.constraint = sp.vstack(pieces, format='csc')
        self.lo, self.hi = np.concatenate(lows), np.concatenate(highs)
        self.settings = settings
        self.solver = osqp.OSQP()
        self.solver.setup(P=sp.diags(np.r_[np.ones(n), np.zeros(m)], format='csc'),
                          q=np.zeros(n+m), A=self.constraint, l=self.lo, u=self.hi,
                          verbose=False, eps_abs=settings['eps_abs'], eps_rel=settings['eps_rel'],
                          max_iter=settings['max_iter'], polishing=True, warm_starting=True,
                          adaptive_rho=True, check_termination=25)

    def solve(self, y, alpha):
        q = np.r_[-np.asarray(y)+alpha*self.linear, alpha*self.weights]
        self.solver.update(q=q)
        result = self.solver.solve(raise_error=False)
        if result.info.status_val != 1 or result.x is None or not np.isfinite(result.x).all():
            raise RuntimeError(f'Proximal QP: {result.info.status}, iterations={result.info.iter}')
        ax = self.constraint@result.x
        primal = max(float(np.maximum(self.lo-ax, 0).max()), float(np.maximum(ax-self.hi, 0).max()))
        px = np.r_[result.x[:self.n], np.zeros(len(result.x)-self.n)]
        dual = float(np.max(np.abs(px+q+self.constraint.T@result.y)))
        scale = max(1., float(np.max(np.abs(result.x))), float(np.max(np.abs(q))),
                    float(np.max(np.abs(self.constraint.T@result.y))))
        tolerance = self.settings['residual_multiplier']*(self.settings['eps_abs']+self.settings['eps_rel']*scale)
        if max(primal, dual) > tolerance:
            raise RuntimeError(f'Proximal QP residual {max(primal,dual):.3g} > {tolerance:.3g}')
        raw = result.x[:self.n]
        z = np.clip(raw, self.lower, self.upper)
        if self.ordered:
            z = np.maximum.accumulate(z)  # Only tolerance-size feasibility repair after QP.
        repair = float(np.max(np.abs(z-raw)))
        if repair > tolerance:
            raise RuntimeError('Proximal feasibility repair exceeds tolerance.')
        return z, {'primal_residual': primal, 'dual_residual': dual,
                   'feasibility_repair': repair, 'tolerance': tolerance,
                   'iterations': int(result.info.iter)}


@dataclass
class Problem:
    x: np.ndarray
    kind: str
    config: dict
    device: torch.device

    def __post_init__(self):
        self.n = len(self.x)
        q = self.config['qpad'][self.kind]
        self.k = q['K']; self.penalties = q['parameters']
        self.scale = self.n/self.k  # Fixed coordinate transform, not an estimated period.
        self.dtype = torch.float64
        self.X = torch.as_tensor(self.x, dtype=self.dtype, device=self.device)
        self.grid = torch.arange(self.n, dtype=self.dtype, device=self.device)/self.scale
        model = self.config['model']
        self.width = model['boundary_fraction']
        self.phase = (torch.arange(model['alignment_points'], dtype=self.dtype, device=self.device)+0.5)/model['alignment_points']
        self.level = max(float(np.max(np.abs(self.x))), 1.)
        mmax = model['amplitude_bound_factor']*self.level
        amax = model['anomaly_bound_factor']*self.level
        tmin, tmax = model['duration_bounds_in_scale']
        self.bounds = {'M': (np.zeros(self.k), np.full(self.k, mmax)),
                       't': (np.full(self.k, -tmax), np.full(self.k, self.n/self.scale)),
                       'T': (np.full(self.k, tmin), np.full(self.k, tmax)),
                       'A': (np.full(self.n, 0. if model['anomaly_domain']=='nonnegative' else -amax), np.full(self.n, amax))}
        p = self.penalties
        d1, d2 = differences(self.k), differences(self.k, 2)
        spec = {'M': ([d1, d2], [np.full(self.k-1,p['eta1']/self.n), np.full(self.k-2,p['eta2']/self.n)], np.zeros(self.k)),
                't': ([], [], np.zeros(self.k)),
                'T': ([d1], [np.full(self.k-1,p['eta3']*self.scale/self.n)], np.zeros(self.k))}
        if model['anomaly_domain'] == 'nonnegative':
            spec['A'] = ([differences(self.n)], [np.full(self.n-1,p['lambda2']/self.n)], np.full(self.n,p['lambda1']/self.n))
        else:
            spec['A'] = ([sp.eye(self.n,format='csc'),differences(self.n)],
                         [np.full(self.n,p['lambda1']/self.n),np.full(self.n-1,p['lambda2']/self.n)],np.zeros(self.n))
        self.prox = {b: ProximalQP(len(self.bounds[b][0]), *spec[b], *self.bounds[b], b=='t', self.config['proximal']) for b in BLOCKS}
        self.reference_steps = {'M': 1., 't': 0.05, 'T': 0.05, 'A': 0.45*self.n}

    def tensor_state(self, values):
        return {b: torch.tensor(values[b], dtype=self.dtype, device=self.device, requires_grad=True) for b in BLOCKS}

    def initial(self, mode, seed):
        onset = np.arange(self.k)*self.n/self.k if self.kind=='simulation' else 250+np.arange(self.k)*(self.n-250)/self.k
        duration = 100. if self.kind=='simulation' else 150.
        # Test configurations and explicitly changed lengths use the same relative scale.
        duration = self.config['qpad'][self.kind].get('initial_duration', duration)
        values = {'M': np.ones(self.k), 't': onset/self.scale,
                  'T': np.full(self.k,duration/self.scale), 'A': np.zeros(self.n)}
        rng = np.random.default_rng(seed)
        if mode == 'local':
            values['t'] += rng.uniform(-0.1,0.1,self.k)
            values['T'] *= rng.uniform(0.9,1.1,self.k)
            values['M'] *= rng.uniform(0.9,1.1,self.k)
        elif mode == 'feasible_random':
            for b in ('M','t','T'):
                lo, hi = self.bounds[b]
                values[b] = rng.uniform(lo, hi)
            values['t'].sort()
        elif mode != 'default':
            raise ValueError(f'Unknown initialization {mode}')
        for b in BLOCKS:
            values[b] = np.clip(values[b], *self.bounds[b])
        values['t'] = isotonic(values['t'])
        return self.tensor_state(values)

    def background(self, state, positions=None):
        positions = self.grid if positions is None else positions
        u = (positions.reshape(1,-1)-state['t'][:,None])/state['T'][:,None]
        return (state['M'][:,None]*local_envelope(u,self.width)).sum(dim=0).reshape(positions.shape)

    def smooth(self, state):
        p = self.background(state)
        data = (self.X-p-state['A']).square().sum()
        positions = state['t'][:,None]+state['T'][:,None]*self.phase[None,:]
        profiles = self.background(state,positions)
        similarity = self.penalties['psi']*(self.scale/2)*torch.diff(profiles,dim=0).square().mean(dim=1).sum()
        return (data+similarity)/self.n

    def regularizer(self, state):
        p = self.penalties
        return (p['lambda1']*state['A'].abs().sum()+p['lambda2']*torch.diff(state['A']).abs().sum()
                +p['eta1']*torch.diff(state['M']).abs().sum()+p['eta2']*torch.diff(state['M'],n=2).abs().sum()
                +p['eta3']*self.scale*torch.diff(state['T']).abs().sum())/self.n

    def objective(self, state):
        return self.smooth(state)+self.regularizer(state)

    def violation(self, state):
        maximum = 0.
        for b in BLOCKS:
            v = state[b].detach().cpu().numpy(); lo,hi = self.bounds[b]
            maximum=max(maximum,float(np.maximum(lo-v,0).max()),float(np.maximum(v-hi,0).max()))
        maximum=max(maximum,float(np.maximum(-np.diff(state['t'].detach().cpu().numpy()),0).max()))
        return maximum

    def stationarity(self,state):
        f = self.smooth(state)
        grads = torch.autograd.grad(f,tuple(state[b] for b in BLOCKS))
        mapping, moves, infos = [], [], []
        for b,g in zip(BLOCKS,grads):
            x = state[b].detach().cpu().numpy(); alpha = self.reference_steps[b]
            z, info = self.prox[b].solve(x-alpha*g.detach().cpu().numpy(),alpha)
            mapping.append((x-z)/alpha); moves.append(float(np.linalg.norm(x-z)/max(1.,np.linalg.norm(x))))
            infos.append(info)
        vector=np.concatenate(mapping)
        return {'mapping_inf':float(np.max(np.abs(vector))), 'mapping_l2':float(np.linalg.norm(vector)),
                'relative_fixed_point_residual':max(moves), 'constraint_violation':self.violation(state),
                'prox_primal_max':max(i['primal_residual'] for i in infos),
                'prox_dual_max':max(i['dual_residual'] for i in infos),
                'prox_feasibility_repair_max':max(i['feasibility_repair'] for i in infos)}


def fit_theory(x, kind, config, device, solver, initialization='default', seed=0, progress=None):
    problem=Problem(np.asarray(x,float),kind,config,torch.device(device))
    state=problem.initial(initialization,seed)
    initial={b:state[b].detach().cpu().numpy().copy() for b in BLOCKS}
    setup=config['optimization']; max_steps=config['qpad'][kind]['epochs']
    steps=dict(problem.reference_steps)
    trace=[]; history=[]; block_audit=[]; satisfactory=0; stop='iteration_budget'
    if solver=='amsgrad':
        optimizer=torch.optim.Adam(list(state.values()),lr=setup['amsgrad_learning_rate'],amsgrad=True)
    elif solver!='palm':
        raise ValueError(solver)
    first=problem.stationarity(state)
    objective=float(problem.objective(state).detach())
    trace.append({'iteration':0,'objective':objective,**first}); history.append(objective)
    if progress: progress(trace[-1])
    total_backtracks=0; worst_prox_primal=0.; worst_prox_dual=0.
    for iteration in range(1,max_steps+1):
        if solver=='palm':
            for b in BLOCKS:
                f=problem.smooth(state); fold=float(f.detach())
                grad=torch.autograd.grad(f,state[b])[0].detach().cpu().numpy()
                old=state[b].detach().cpu().numpy().copy()
                fullold=float(problem.objective(state).detach())
                alpha=min(steps[b]*(setup['step_growth'] if b!='A' else 1.),setup['step_max'])
                if b=='A': alpha=min(alpha,problem.reference_steps['A'])
                accepted=False
                for bt in range(setup['max_backtracks']):
                    candidate,info=problem.prox[b].solve(old-alpha*grad,alpha)
                    trial=dict(state)
                    trial[b]=torch.tensor(candidate,dtype=problem.dtype,device=problem.device,requires_grad=True)
                    d=candidate-old; d2=float(d@d)
                    fnew=float(problem.smooth(trial).detach()); fullnew=float(problem.objective(trial).detach())
                    slack=setup['descent_tolerance']*max(1.,abs(fullold))
                    upper=fold+float(grad@d)+(1-setup['descent_fraction'])*d2/(2*alpha)
                    margin=setup['descent_fraction']*d2/(2*alpha)
                    if fnew<=upper+slack and fullnew<=fullold-margin+slack:
                        state=trial; accepted=True; steps[b]=alpha
                        block_audit.append([iteration,BLOCKS.index(b),fullold,fullnew,alpha,bt,d2,slack,margin,
                                            info['primal_residual'],info['dual_residual']])
                        total_backtracks+=bt
                        worst_prox_primal=max(worst_prox_primal,info['primal_residual'])
                        worst_prox_dual=max(worst_prox_dual,info['dual_residual'])
                        break
                    alpha*=setup['backtrack_factor']
                    if alpha<setup['step_min']: break
                if not accepted:
                    raise RuntimeError(f'Line search failed at iteration {iteration}, block {b}; no accepted step.')
        else:
            lr=setup['amsgrad_learning_rate']/(1+(iteration-1)/setup['amsgrad_decay_scale'])**setup['amsgrad_decay_power']
            for group in optimizer.param_groups:group['lr']=lr
            optimizer.zero_grad(); loss=problem.objective(state);loss.backward();optimizer.step()
            with torch.no_grad():
                for b in BLOCKS:
                    v=state[b].cpu().numpy().copy()
                    if b=='t':
                        optstate=optimizer.state[state[b]]
                        denom=np.sqrt(optstate['max_exp_avg_sq'].cpu().numpy()/(1-0.999**iteration))+1e-8
                        v=isotonic(v,denom)
                    v=np.clip(v,*problem.bounds[b])
                    state[b].copy_(torch.as_tensor(v,dtype=problem.dtype,device=problem.device))
        objective=float(problem.objective(state).detach())
        if not np.isfinite(objective):raise FloatingPointError('Non-finite objective')
        history.append(objective)
        if iteration%setup['check_every']==0 or iteration==max_steps:
            diagnostics=problem.stationarity(state)
            row={'iteration':iteration,'objective':objective,**diagnostics}
            trace.append(row)
            if progress:progress(row)
            meets=(diagnostics['mapping_inf']<=setup['stationarity_tolerance'] and
                   diagnostics['relative_fixed_point_residual']<=setup['fixed_point_tolerance'] and
                   diagnostics['constraint_violation']<=1e-10)
            satisfactory=satisfactory+1 if meets else 0
            if satisfactory>=setup['consecutive_checks']:
                stop='stationarity_tolerance';break
    final={b:state[b].detach().cpu().numpy().copy() for b in BLOCKS}
    with torch.no_grad():p=problem.background(state).cpu().numpy();a=state['A'].cpu().numpy()
    meta={'solver':solver,'termination':stop,'iterations':iteration,'final_diagnostics':trace[-1],
          'coordinate_scale':problem.scale,'bounds':{b:[lo.tolist(),hi.tolist()] for b,(lo,hi) in problem.bounds.items() if b!='A'},
          'anomaly_bounds':[float(problem.bounds['A'][0][0]),float(problem.bounds['A'][1][0])],
          'penalties':problem.penalties,'total_backtracks':total_backtracks,
          'prox_primal_max':worst_prox_primal,'prox_dual_max':worst_prox_dual,
          'initialization':initialization,'seed':seed,
          'objective_increases':sum(history[i]>history[i-1]+setup['descent_tolerance']*max(1.,abs(history[i-1])) for i in range(1,len(history)))}
    arrays={'objective_history':np.asarray(history),'trace_columns':np.asarray(list(trace[0])),
            'trace':np.asarray([[r[k] for k in trace[0]] for r in trace]),
            'block_audit_columns':np.asarray(['iteration','block','old_objective','new_objective','step','backtracks','step_squared_norm','slack','required_decrease','prox_primal','prox_dual']),
            'block_audit':np.asarray(block_audit).reshape(-1,11)}
    for b in BLOCKS:
        arrays['initial_'+b]=initial[b];arrays['final_'+b]=final[b]
        if b in ('t','T'):
            arrays['initial_'+b+'_samples']=initial[b]*problem.scale
            arrays['final_'+b+'_samples']=final[b]*problem.scale
    return p,a,meta,arrays

"""Run frozen QPAD definitions with recorded initial states and explicit devices."""
import importlib
import numpy as np


def fit_qpad(x, dataset, config, device, initialization_seed=None):
    import torch
    definitions = importlib.import_module('models.real' if dataset == 'real' else 'models.simulation')
    options = config['qpad']['real' if dataset == 'real' else 'simulation']
    torch.manual_seed(0 if initialization_seed is None else initialization_seed)
    model = definitions.Periodic(options['K'], len(x)).to(device)
    opt = definitions.JointOptimizer.__new__(definitions.JointOptimizer)
    opt.device, opt.periodic_model = device, model
    opt.X = torch.tensor(x, dtype=torch.float32, device=device)
    for key, value in options['parameters'].items():
        setattr(opt, key, value)
    if initialization_seed is not None:
        rng = np.random.default_rng(initialization_seed)
        perturb = config['initialization']
        period = len(x)/options['K']
        with torch.no_grad():
            model.t_k.add_(torch.as_tensor(rng.uniform(-1, 1, options['K']) *
                                          perturb['phase_fraction']*period, device=device, dtype=torch.float32))
            model.T_active.mul_(torch.as_tensor(1+rng.uniform(-1, 1, options['K']) *
                                               perturb['duration_fraction'], device=device, dtype=torch.float32))
            model.M_independent.mul_(torch.as_tensor(1+rng.uniform(-1, 1, options['K']) *
                                                    perturb['amplitude_fraction'], device=device, dtype=torch.float32))
    def state():
        return {key: value.detach().cpu().numpy().copy()
                for key, value in model.named_parameters() if key != 'M'}
    initial = state()
    adam = torch.optim.Adam(model.parameters(), lr=config['qpad']['learning_rate'])
    trace = []
    for step in range(options['epochs']):
        adam.zero_grad()
        objective = opt.compute_loss()
        if not torch.isfinite(objective):
            raise FloatingPointError(f'Nonfinite QPAD objective before update {step+1}.')
        if step % config['qpad']['trace_every'] == 0:
            trace.append([step, float(objective.detach().cpu())])
        (objective*config['qpad']['loss_scale']).backward()
        if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
            raise FloatingPointError(f'Nonfinite QPAD gradient at update {step+1}.')
        adam.step()
    with torch.no_grad():
        p, a = model()
        final_objective = float(opt.compute_loss().cpu())
    final = state()
    trace.append([options['epochs'], final_objective])
    if not np.isfinite(final_objective) or any(not np.isfinite(v).all() for v in final.values()):
        raise FloatingPointError('Nonfinite QPAD final state.')
    arrays = {'trace': np.asarray(trace, dtype=float)}
    arrays.update({f'initial_{k}': v for k, v in initial.items()})
    arrays.update({f'final_{k}': v for k, v in final.items()})
    meta = {'iterations': options['epochs'], 'final_objective': final_objective,
            'stopping_rule': 'fixed_iteration_budget', 'solver_status': 'completed_fixed_budget',
            'initialization_seed': initialization_seed, 'K': options['K'],
            'minimum_final_duration': float(final['T_active'].min()),
            'ordered_final_onsets': bool(np.all(np.diff(final['t_k']) > 0)),
            'max_onset_movement': float(np.max(np.abs(final['t_k']-initial['t_k'])))}
    return p.detach().cpu().numpy(), a.detach().cpu().numpy(), meta, arrays

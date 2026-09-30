"""Continuous-score baselines. Mathematical sources and adaptations: PROTOCOL.md."""
import numpy as np
from scipy.linalg import cho_factor, cho_solve, toeplitz
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import minimize
from scipy.signal import find_peaks


def mad_scale(x):
    x = np.asarray(x, dtype=float)
    return float(1.482602218505602 * np.median(np.abs(x-np.median(x))))


def estimate_period(x, settings):
    """Clipped, smoothed autocorrelation; returns periods in sample units."""
    n = len(x)
    lo = max(4, int(n * settings['period_min_fraction']))
    hi = min(n//2, int(n * settings['period_max_fraction']))
    center = np.median(x)
    scale = max(mad_scale(x), np.std(x)*0.1, 1e-12)
    z = gaussian_filter1d(np.clip(x-center, -3*scale, 3*scale), 1.5)
    z -= z.mean()
    f = np.fft.rfft(z, n=2*n)
    acf = np.fft.irfft(f*np.conj(f), n=2*n)[:n]
    if acf[0] <= 1e-20:
        return max(lo, min(hi, n//10)), {'status': 'flat_signal_fallback'}
    acf /= acf[0]
    peaks = find_peaks(acf[lo:hi+1])[0] + lo
    if not len(peaks):
        p = int(lo + np.argmax(acf[lo:hi+1]))
    else:
        best = np.max(acf[peaks])
        strong = peaks[acf[peaks] >= max(0.0, 0.85*best)]
        p = int(strong[0] if len(strong) else peaks[np.argmax(acf[peaks])])
    return p, {'status': 'estimated', 'acf_at_period': float(acf[p]), 'search_samples': [lo, hi]}


class QPGPCovariance:
    """Exact observed-grid covariance solver, including a final partial cycle.

    C_ij = exp[-theta^2 sin^2(pi(i-j)/p)] * omega^|floor(i/p)-floor(j/p)|
           + delta^2 I_ij.
    Complete cycles use FFT/eigendecomposition, the tail a Schur complement.
    """
    def __init__(self, n, period, delta, theta, omega):
        self.n, self.p = n, int(period)
        self.k, self.m = divmod(n, self.p)
        if self.k < 1 or delta <= 0 or theta <= 0 or not 0 < omega < 1:
            raise ValueError('Invalid QPGP dimensions or covariance parameters.')
        self.delta = delta
        self.r = np.exp(-(theta*np.sin(np.pi*np.arange(self.p)/self.p))**2)
        self.sr = np.maximum(np.fft.fft(self.r).real, 0)
        w, self.u = np.linalg.eigh(toeplitz(omega**np.arange(self.k)))
        self.den = self.sr[:, None]*np.maximum(w, 0)[None, :] + delta**2
        self.logdet = float(np.log(self.den).sum())
        self.cross = omega**np.arange(self.k, 0, -1)
        if self.m:
            v = self.u.T @ self.cross
            first = self.r.copy()
            first[0] += delta**2
            first -= np.fft.ifft(self.sr**2 * ((1/self.den) @ (v*v))).real
            self.tail_factor = cho_factor(toeplitz(first[:self.m]), lower=True, check_finite=False)
            self.logdet += float(2*np.log(np.diag(self.tail_factor[0])).sum())

    def _full_solve(self, b):
        # Column-major phase/cycle layout agrees with contiguous time ordering.
        mat = b.reshape(self.k, self.p).T
        freq = np.fft.fft(mat @ self.u, axis=0) / self.den
        return (np.fft.ifft(freq, axis=0).real @ self.u.T).T.reshape(-1)

    def _cross_to_tail(self, full):
        mat = full.reshape(self.k, self.p).T
        return np.fft.ifft(self.sr * np.fft.fft(mat @ self.cross)).real[:self.m]

    def _cross_to_full(self, tail):
        padded = np.zeros(self.p)
        padded[:self.m] = tail
        phase = np.fft.ifft(self.sr * np.fft.fft(padded)).real
        return (phase[:, None] * self.cross[None, :]).T.reshape(-1)

    def solve(self, b):
        b = np.asarray(b, dtype=float)
        full = self._full_solve(b[:self.k*self.p])
        if not self.m:
            return full
        tail = cho_solve(self.tail_factor, b[self.k*self.p:] - self._cross_to_tail(full), check_finite=False)
        full -= self._full_solve(self._cross_to_full(tail))
        return np.concatenate([full, tail])

    def likelihood(self, y):
        qy, q1 = self.solve(y), self.solve(np.ones(self.n))
        beta = float(qy.sum()/q1.sum())
        residual = y-beta
        alpha = qy-beta*q1
        sigma2 = max(float(residual @ alpha / self.n), 1e-15)
        nll = 0.5*(self.logdet + self.n*(np.log(sigma2)+1+np.log(2*np.pi)))
        return float(nll), beta, sigma2, alpha


def fit_qpgp(x, period, settings):
    # Center/scale improves conditioning; parameter selection uses likelihood only.
    center, scale = float(np.mean(x)), max(float(np.std(x)), 1e-12)
    y = (x-center)/scale
    candidates = sorted(set(max(4, min(len(x)//2, int(round(period*f))))
                            for f in settings['qpgp_period_factors']))
    starts = [np.array([np.log(d), np.log(t), w])
              for d, t, w in [(0.2, 1.0, 0.5), (0.5, 3.0, 0.9),
                              (1.0, 1.0, 0.9), (0.2, 5.0, 0.5)]]
    bounds = [(np.log(0.03), np.log(3.0)), (np.log(0.2), np.log(10.0)), (0.01, 0.999)]
    trials = []
    for p in candidates:
        def objective(v):
            try:
                model = QPGPCovariance(len(y), p, np.exp(v[0]), np.exp(v[1]), v[2])
                return model.likelihood(y)[0]
            except (ValueError, np.linalg.LinAlgError, FloatingPointError):
                return 1e100
        start = min(starts, key=objective)
        initial_value = objective(start)
        fit = minimize(objective, start, method='L-BFGS-B', bounds=bounds,
                       options={'maxiter': settings['qpgp_maxiter'], 'ftol': 1e-8})
        candidate = fit.x if np.isfinite(fit.fun) and fit.fun <= initial_value else start
        value = objective(candidate)
        trials.append({'period': p, 'nll': float(value), 'parameters': candidate.tolist(),
                       'optimizer_success': bool(fit.success), 'optimizer_message': str(fit.message),
                       'iterations': int(fit.nit)})
    best = min(trials, key=lambda row: row['nll'])
    if best['nll'] >= 1e99:
        raise RuntimeError('All QPGP covariance fits failed.')
    d, t, w = best['parameters']
    model = QPGPCovariance(len(y), best['period'], np.exp(d), np.exp(t), w)
    _, beta, sigma2, alpha = model.likelihood(y)
    # Posterior mean of the latent signal: beta + K C^-1 (y-beta).
    latent = y - model.delta**2 * alpha
    p = center+scale*latent
    return p, x-p, {'selected_period': best['period'], 'delta': float(np.exp(d)),
                    'theta': float(np.exp(t)), 'omega': float(w), 'beta': beta,
                    'sigma2_normalized': sigma2, 'period_trials': trials,
                    'solver_status': 'converged' if best['optimizer_success'] else 'budget_or_line_search_stop'}


def cycle_rpca(x, period, maxiter=500, tolerance=1e-6):
    """Principal component pursuit on a phase-by-cycle matrix, IALM updates.

    Pad the final cycle from the beginning of X, then crop on reconstruction.
    """
    n = len(x)
    cols = int(np.ceil(n/period))
    matrix = np.resize(x, cols*period).reshape(cols, period).T.copy()
    lam = 1/np.sqrt(max(matrix.shape))
    norm = np.linalg.norm(matrix, 'fro')
    if norm == 0:
        return x.copy(), np.zeros_like(x), {'iterations': 0, 'relative_residual': 0.0, 'solver_status': 'converged'}
    spectral = np.linalg.norm(matrix, 2)
    dual = matrix / max(spectral, np.max(np.abs(matrix))/lam)
    mu = 1.25/max(spectral, 1e-12)
    cap = mu*1e7
    low, sparse = np.zeros_like(matrix), np.zeros_like(matrix)
    residual = float('inf')
    for step in range(1, maxiter+1):
        u, s, vt = np.linalg.svd(matrix-sparse+dual/mu, full_matrices=False)
        low = (u*np.maximum(s-1/mu, 0)) @ vt
        z = matrix-low+dual/mu
        sparse = np.sign(z)*np.maximum(np.abs(z)-lam/mu, 0)
        error = matrix-low-sparse
        residual = float(np.linalg.norm(error, 'fro')/norm)
        if residual < tolerance:
            break
        dual += mu*error
        mu = min(mu*1.5, cap)
    p = low.T.reshape(-1)[:n]
    a = sparse.T.reshape(-1)[:n]
    return p, a, {'lambda': float(lam), 'iterations': step, 'relative_residual': residual,
                  'padding_points': cols*period-n, 'period': period,
                  'solver_status': 'converged' if residual < tolerance else 'iteration_budget'}


def fit_baseline(method, x, settings):
    period, period_info = estimate_period(x, settings)
    if method == 'STL':
        from statsmodels.tsa.seasonal import STL
        fit = STL(x, period=period, robust=True).fit()
        p = fit.trend+fit.seasonal
        a, info = x-p, {'robust': True, 'period': period}
    elif method == 'VMD':
        from vmdpy import VMD
        # vmdpy requires even length; pad one point and crop when needed.
        data = x if len(x)%2 == 0 else np.r_[x, x[-1]]
        u, _, omega = VMD(data, settings['vmd_alpha'], 0.0,
                          settings['vmd_modes'], False, 1, settings['vmd_tolerance'])
        chosen = np.argsort(omega[-1])[:settings['vmd_background_modes']]
        p = np.sum(u[chosen], axis=0)[:len(x)]
        a = x-p
        info = {'selected_modes': chosen.tolist(), 'final_frequencies': omega[-1].tolist(),
                'alpha': settings['vmd_alpha'], 'iterations': len(omega),
                'solver_status': 'iteration_budget' if len(omega) >= 499 else 'stopped'}
    elif method == 'QPGP':
        p, a, info = fit_qpgp(x, period, settings)
    elif method == 'RPCA':
        p, a, info = cycle_rpca(x, period, settings['rpca_maxiter'], settings['rpca_tolerance'])
    else:
        raise ValueError(f'Unknown baseline: {method}')
    return p, a, {'period_estimate': period, 'period_estimation': period_info, **info}

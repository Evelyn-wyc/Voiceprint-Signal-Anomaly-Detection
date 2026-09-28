'''
Method 3: Statistical Model (Non-Parametric): Yongxiang Li's work (QPGP).
Quasi-Periodic Gaussian Process Modeling of Pseudo-Periodic Signals
'''

import numpy as np
from scipy.linalg import toeplitz, cholesky, solve_toeplitz
from scipy.fft import fft, ifft
from scipy.optimize import minimize
import os
from matplotlib import pyplot as plt
from tqdm import tqdm


def periodic_corr(delta, theta, p, t1, t2):
    """
    Example periodic correlation function (kernel).
    This is a common choice for periodic Gaussian Processes.

    Args:
        delta (float): The amplitude of the correlation.
        theta (float): The length-scale parameter.
        p (int): The period.
        t1 (np.ndarray): First set of time points.
        t2 (np.ndarray): Second set of time points.

    Returns:
        np.ndarray: The correlation matrix.
    """
    dist = np.abs(t1[:, np.newaxis] - t2)
    return delta**2 * np.exp(-2 * (np.sin(np.pi * dist / p) / theta)**2)

def constant_regr(t):
    """
    Example constant regression function (mean function).
    Assumes a constant mean value across the process.

    Args:
        t (np.ndarray): Time points.

    Returns:
        np.ndarray: The regression matrix (design matrix).
    """
    return np.ones((len(t), 1))

def kmseig(k, omega):
    """
    Computes eigenvalues and eigenvectors for a stationary correlation matrix.
    Translated from the SMT (Surrogate Modeling Toolbox) `kmseig.m`.

    Args:
        k (int): The dimension of the Toeplitz matrix.
        omega (float): The correlation parameter.

    Returns:
        tuple[np.ndarray, np.ndarray]: A tuple containing the eigenvalues (1D array) and the Uo matrix.
    """
    # Create the first row of the Toeplitz matrix
    c = omega ** np.arange(k)
    C = toeplitz(c)
    
    # Eigendecomposition. S_eig is a 1D array of eigenvalues.
    S_eig, V = np.linalg.eigh(C)

    # Ensure eigenvalues are positive for numerical stability
    S_eig[S_eig <= 1e-9] = 1e-9

    # Uo is needed for later calculations
    Uo = V @ np.diag(1. / np.sqrt(S_eig))
    
    # Return the 1D array of eigenvalues and the Uo matrix.
    # The original error was caused by returning the inverse of C instead of its eigenvalues.
    return S_eig, Uo


def objfunc(para, data):
    """
    Objective function: calculates the negative log-likelihood.
    This function is minimized to find the best model parameters.

    Args:
        para (list or np.ndarray): List of parameters [delta, theta, omega].
        data (dict): A dictionary containing the data and model settings.

    Returns:
        tuple[float, dict]: The minimum negative log-likelihood and a fit dictionary.
    """
    delta, theta, omega = para
    Y_data = data['Y']
    P_periods = data['P']
    corr_func = data['corr']
    regr_func = data['regr']

    n = len(Y_data)
    q = regr_func(np.array([0])).shape[1]

    likelihoods = np.full(len(P_periods), np.nan)
    sigmas = np.full(len(P_periods), np.nan)
    betas = np.full((len(P_periods), q), np.nan)

    for i, p in enumerate(P_periods):
        p = int(p)
        k = n // p
        p1 = n - p * k

        if k == 0: # Cannot proceed if the period is longer than the data
            continue

        Y = np.zeros((p, k))
        Y.flat[:k * p] = Y_data[:k * p]
        Ys = Y_data[k * p:]

        Gamma = regr_func(np.arange(1, p + 1))
        Gs = Gamma[:p1, :]

        So_eig, Uo = kmseig(k, omega)
        
        t_range = np.arange(1, p + 1)
        r0 = corr_func(delta, theta, p, t_range, 1).flatten()
        r0[0] += 1e-8
        
        Sr = np.real(fft(r0))
        Sr[Sr < 0] = 0

        S = Sr[:, np.newaxis] * So_eig + delta**2
        Lambda = 1. / S

        UrGamma = fft(Gamma, axis=0) / np.sqrt(p)
        UoOnes = Uo.T @ np.ones(k)
        UrYUo = fft(Y @ Uo, axis=0) / np.sqrt(p)

        ChiYY = np.real(np.sum(np.conj(UrYUo) * (UrYUo * Lambda)))
        ChiFF_term = (Lambda @ (UoOnes**2))
        ChiFF = np.real(np.conj(UrGamma).T @ (ChiFF_term[:, np.newaxis] * UrGamma))
        ChiFY = np.real(np.conj(UrGamma).T @ ((UrYUo * Lambda) @ UoOnes))

        if p1 == 0:
            try:
                Beta = np.linalg.solve(ChiFF, ChiFY)
                sigma2 = (ChiYY - np.conj(Beta).T @ ChiFF @ Beta) / n
            except np.linalg.LinAlgError:
                continue
        else:
            r = r0.copy()
            r[0] += delta**2
            
            UoW = Uo.T @ (omega ** np.arange(k, 0, -1))
            
            # FIX: Ensure element-wise multiplication for vectors.
            # The previous use of Sr[:, np.newaxis] caused an incorrect broadcast to a matrix.
            # This ensures we have (p,) * (p,) element-wise multiplication.
            Yd_term_matmul = (UrYUo * Lambda) @ UoW
            Yd_term = Sr * Yd_term_matmul
            Yd = ifft(Yd_term, axis=0) * np.sqrt(p)
            Yd = np.real(Ys - Yd[:p1].flatten())

            # Apply the same fix for the Fd calculation
            Fd_term_matmul = Lambda @ (UoW * UoOnes)
            Fd_term = Sr * Fd_term_matmul
            Fd_fft = Fd_term[:, np.newaxis] * fft(Gamma, axis=0)
            Fd = ifft(Fd_fft, axis=0)
            Fd = np.real(Gs - Fd[:p1, :])

            try:
                L = cholesky(toeplitz(r[:p1]), lower=True)
                Fdl = np.linalg.solve(L, Fd)
                FPiF = Fdl.T @ Fdl
                Ydl = np.linalg.solve(L, Yd)
                YPiY = Ydl.T @ Ydl
                FPiY = Fdl.T @ Ydl
                
                Beta = np.linalg.solve(ChiFF + FPiF, ChiFY + FPiY)
                sigma2_num = (ChiYY + YPiY) - np.conj(Beta).T @ (ChiFF + FPiF) @ Beta
                sigma2 = sigma2_num / n
                
                log_det_term = np.sum(2 * np.log(np.diag(L)))
                likelihoods[i] = (n * np.log(sigma2) + np.sum(np.log(np.real(S))) + log_det_term + n + n * np.log(2 * np.pi)) / 2
            except np.linalg.LinAlgError:
                continue 

        if 'sigma2' in locals() and sigma2 > 0:
            sigmas[i] = np.sqrt(sigma2)
            betas[i, :] = Beta.flatten()
            if p1 == 0:
                 likelihoods[i] = (n * np.log(sigma2) + np.sum(np.log(np.real(S))) + n + n * np.log(2 * np.pi)) / 2
    
    valid_idx = ~np.isnan(likelihoods)
    if not np.any(valid_idx):
        obj = np.inf
    else:
        obj = np.min(likelihoods[valid_idx])

    fit = {
        'P': P_periods,
        'Y': Y_data,
        'sigma': sigmas,
        'beta': betas,
        'corr': corr_func,
        'regr': regr_func,
        'likelihood': -likelihoods
    }
    return obj, fit


def fit_qpgp(P, Y, regr, corr, lob, upb, theta0=None):
    """
    Main function to fit the Quasi-Periodic Gaussian Process model.

    Args:
        P (np.ndarray): Array of periods to test.
        Y (np.ndarray): The time series data.
        regr (callable): The regression (mean) function.
        corr (callable): The correlation (kernel) function.
        lob (list or np.ndarray): Lower bounds for the parameters [delta, theta, omega].
        upb (list or np.ndarray): Upper bounds for the parameters.
        theta0 (list or np.ndarray, optional): Initial guess for the parameters. Defaults to None.

    Returns:
        dict: A dictionary containing the fitted model and results.
    """
    data = {'corr': corr, 'regr': regr, 'P': P, 'Y': Y}
    bounds = list(zip(lob, upb))

    if theta0 is not None:
        res = minimize(lambda t: objfunc(t, data)[0], theta0, method='L-BFGS-B', bounds=bounds)
        theta_hat = res.x
        _, fit = objfunc(theta_hat, data)

    else:
        u = np.linspace(0, 1, 6)
        grid_points = u[1:-1]
        x1, x2, x3 = np.meshgrid(grid_points, grid_points, grid_points)
        
        thetas = np.array(lob) + (np.array(upb) - np.array(lob)) * np.vstack([x1.ravel(), x2.ravel(), x3.ravel()]).T
        
        objs = np.full(thetas.shape[0], np.inf)
        
        for i, t in enumerate(thetas):
            objs[i], _ = objfunc(t, data)

        best_start_index = np.nanargmin(objs)
        theta0 = thetas[best_start_index, :]
        
        res = minimize(lambda t: objfunc(t, data)[0], theta0, method='L-BFGS-B', bounds=bounds)
        theta_hat = res.x
        _, fit = objfunc(theta_hat, data)

    fit['theta0'] = theta0
    fit['thetahat'] = theta_hat
    
    valid_likelihoods = fit['likelihood'][~np.isnan(fit['likelihood'])]
    if len(valid_likelihoods) > 0:
        best_idx = np.nanargmax(fit['likelihood'])
        fit['betahat'] = fit['beta'][best_idx, :]
        fit['sigmahat'] = fit['sigma'][best_idx]
        fit['period'] = fit['P'][best_idx]
        fit['Gamma'] = data['regr'](np.arange(1, int(fit['period']) + 1))
    else:
        fit['betahat'] = None
        fit['sigmahat'] = None
        fit['period'] = None
        fit['Gamma'] = None

    return fit

if __name__ == '__main__':
    
    # 1. Read the data
    
    # # simulation
    # dirpath = '../synthetic_signal/'
    # savepath_rst = './simulation/method3/result/'
    # savepath_npy = './simulation/method3/npy/'

    # case
    dirpath = '../241230_vector_npy/'
    savepath_rst = './case/method3/result/'
    savepath_npy = './case/method3/npy/'
    files = os.listdir(dirpath)
    files.sort()

    for file in tqdm(files):
        if file.endswith('.npy'):
            Y_data = np.load(os.path.join(dirpath, file))
            n_points = len(Y_data)
            time = np.linspace(0, 100, n_points)  # 时间轴

            # 2. Define the search space for the model
            P_periods = np.arange(6, 12, 1) 
            lower_bounds = [1e-2, 1e-2, 1e-2]
            upper_bounds = [np.std(Y_data) * 3, 10.0, 0.99]

            # 3. Run the fitting function
            print("Fitting Quasi-Periodic Gaussian Process model...")
            fit_results = fit_qpgp(
                P=P_periods, 
                Y=Y_data, 
                regr=constant_regr, 
                corr=periodic_corr, 
                lob=lower_bounds, 
                upb=upper_bounds
            )

            # 4. Print the results
            print("\n--- Fit Results ---")
            print(f"Optimal Period: {fit_results['period']:.2f}")
            print(f"Optimal Sigma (noise std dev): {fit_results['sigmahat']:.4f}")
            print(f"Optimal Beta (mean level): {fit_results['betahat'][0]:.4f}")
            print(f"Optimal Theta (parameters): {fit_results['thetahat']}")
            
            # --- 5. Saving and plotting in unified 4-subplot style ---
            periodic_component = np.tile(fit_results['Gamma'] @ fit_results['betahat'],
                                        (n_points // int(fit_results['period'])) + 1)[:n_points]
            residuals = Y_data - periodic_component
            anomaly_threshold = 3 * fit_results['sigmahat']
            A_qpgp = np.where(np.abs(residuals) > anomaly_threshold, residuals, 0.0)  # anomaly component

            # 保存 npy 文件
            np.save(os.path.join(savepath_npy, file[:-4] + '_P.npy'), periodic_component)
            np.save(os.path.join(savepath_npy, file[:-4] + '_A.npy'), A_qpgp)

            # 全局字体设置（非常重要）
            plt.rcParams.update({
            "font.size": 18,          # 基础字体
            "axes.titlesize": 20,     # 子图标题
            "axes.labelsize": 16,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
            "legend.fontsize": 14
            })
            plt.figure(figsize=(14, 9), dpi=300)

            plt.subplot(4, 1, 1)
            plt.plot(time, Y_data, label='Original Signal', linewidth=1.5)
            plt.title('Original Signal')

            plt.subplot(4, 1, 2)
            plt.plot(time, periodic_component, label='Periodic', color='green', linewidth=1.5)
            plt.title('Periodic')

            plt.subplot(4, 1, 3)
            plt.plot(time, A_qpgp, label='Anomaly', color='red', linewidth=1.5)
            plt.title('Anomaly')

            plt.subplot(4, 1, 4)
            plt.plot(time, Y_data - periodic_component - A_qpgp, label='Residual', color='orange', linewidth=1.5)
            plt.title('Residual')

            plt.tight_layout()
            plt.subplots_adjust(hspace=0.6)
            plt.savefig(os.path.join(savepath_rst, file[:-4] + '_QRGP_result.png'), bbox_inches='tight')
            plt.close()

'''
Method 2: Time-Frequency: Variational Mode Decomposition (VMD).
'''

import numpy as np
from vmdpy import VMD
import os
import seaborn as sns
from matplotlib import pyplot as plt
from pandas.plotting import register_matplotlib_converters

# 设置数据目录和保存目录
# simulation
dirpath = '../synthetic_signal/'
savepath_rst = './simulation/method2/result/'
savepath_npy = './simulation/method2/npy/'

# # case
# dirpath = '../241230_vector_npy/'
# savepath_rst = './case/method2/result/'
# savepath_npy = './case/method2/npy/'
files = os.listdir(dirpath)
files.sort()

# --- VMD Parameters ---
alpha = 2000  # Moderate bandwidth constraint
tau = 0.      # Noise-slack (0 means no noise)
K = 4         # Total number of modes to be decomposed
DC = False    # No DC part imposed
init = 1      # Uniform initialization
tol = 1e-7    # Tolerance
num_periodic_modes = 2 # Number of lowest-frequency modes to sum for the periodic component


for file in files:
    # if file.endswith('.npy') and file.startswith('cpl'):
    if file.endswith('.npy'):
        print(f"Processing file: {file}")
        signal = np.load(os.path.join(dirpath, file))
        time = np.linspace(0, 100, len(signal))  # 时间轴
        u, u_hat, omega = VMD(signal, alpha, tau, K, DC, init, tol)
        P_vmd = np.sum(u[:num_periodic_modes], axis=0)  # Sum the first num_periodic_modes for periodic component

        # 基于残差识别异常点 (使用 3-sigma 法则)
        resid = signal - P_vmd
        resid_mean = resid.mean()
        resid_std = resid.std()
        anomaly_threshold_upper = resid_mean + 3 * resid_std
        anomaly_threshold_lower = resid_mean - 3 * resid_std
        A_vmd = np.where((resid < anomaly_threshold_lower) | (resid > anomaly_threshold_upper), resid, 0.0)

        # 将周期成分和异常成分存储为 .npy 文件
        base_filename = file[:-4]  # 去掉 .npy 后缀
        np.save(f'{savepath_npy}{base_filename}_P.npy', P_vmd)
        np.save(f'{savepath_npy}{base_filename}_A.npy', A_vmd)

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
        plt.plot(time, signal, label='Original Signal', linewidth=1.5)
        plt.title('Original Signal')
        
        plt.subplot(4, 1, 2)
        plt.plot(time, P_vmd, label='Periodic', color='green', linewidth=1.5)
        plt.title('Periodic')
        
        plt.subplot(4, 1, 3)
        plt.plot(time, A_vmd, label='Anomaly', color='red', linewidth=1.5)
        plt.title('Anomaly')
        
        plt.subplot(4, 1, 4)
        plt.plot(time, signal - P_vmd - A_vmd, label='Residual', color='orange', linewidth=1.5)
        plt.title('Residual')
        
        plt.tight_layout()
        plt.subplots_adjust(hspace=0.5)
        plt.savefig(f'{savepath_rst}{base_filename}_vmd_result.png',
            bbox_inches='tight')
        plt.close()
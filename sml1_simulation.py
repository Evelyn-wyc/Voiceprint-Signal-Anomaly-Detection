'''
simulate a synthetic signal with pseudo-periodic component P(t),
wide anomalies A(t) with hill shape and jitter, and noise N(t).
P(t) has two types: cpl and non-cpl. Check the filedir and name for details.
cpl: complex periodic with amplitude drift, period jitter, baseline drift
non-cpl: simple periodic with fixed amplitude (period jitter).
'''

import numpy as np
import matplotlib.pyplot as plt
import os

savedir = './synthetic_signal/'
savedir_gt = './synthetic_gtanomaly/'
if not os.path.exists(savedir):
    os.makedirs(savedir)

# 时间轴
t = np.linspace(0, 100, 2000)  # 0-100秒，2000个点

# 基础周期参数
M_base = 2.0
T_active_base = 5.0 # T_active = 100
T_rest_base = 5.0 # T_rest = 100
T_base = T_active_base + T_rest_base

def generate_pseudo_periodic(t, M, T_active_base, T_rest_base, jitter_std=0.2):
    P = np.zeros_like(t)
    t_now = 0.0
    while t_now < t[-1]:
        T_active = T_active_base + np.random.normal(0, jitter_std)
        T_rest = T_rest_base + np.random.normal(0, jitter_std)
        T_total = T_active + T_rest
        
        mask = (t >= t_now) & (t < t_now + T_active)
        P[mask] = M * np.sin(np.pi * (t[mask] - t_now) / T_active)
        
        t_now += T_total
    return P

# P_true = generate_pseudo_periodic(t, M_base, T_active_base, T_rest_base)

def generate_complex_periodic(
    t, 
    M_base=2.0, M_drift=0.2, M_fluctuation_std=0.1,
    T_active_base=5.0, T_rest_base=5.0, T_jitter_std=0.5,
    asymmetry_factor=1.2, 
    baseline_amplitude=0.05, baseline_freq=0.05
):
    """
    - t: 时间向量
    - M_base: 基础振幅
    - M_drift: 整个信号期间，振幅的变化范围 (+/-)
    - M_fluctuation_std: 每个周期的随机振幅波动标准差
    - T_active_base, T_rest_base: 基础的活动/休息时长
    - T_jitter_std: 时长的随机抖动标准差
    - asymmetry_factor: 波形不对称因子 (>1 左偏, <1 右偏)
    - baseline_amplitude, baseline_freq: 基线漂移的振幅和频率
    """
    P = np.zeros_like(t)
    t_now = 0.0
    
    # 1. 生成振幅序列
    K_approx = int(t[-1] / (T_active_base + T_rest_base))
    amplitudes_drift = np.linspace(M_base - M_drift/2, M_base + M_drift/2, K_approx) 
    amplitudes_noise = np.random.normal(0, M_fluctuation_std, K_approx)
    M_sequence = amplitudes_drift + amplitudes_noise
    
    k = 0
    while t_now < t[-1]:
        # 2. 周期波动
        T_active = T_active_base + np.random.normal(0, T_jitter_std)
        T_rest = T_rest_base + np.random.normal(0, T_jitter_std)
        # 保证时长为正
        T_active = max(T_active, T_active_base / 2)
        T_total = T_active + T_rest
        
        # 获取当前周期的独立振幅
        current_M = M_sequence[k] if k < len(M_sequence) else M_sequence[-1]
        
        # 找到当前活动区间的mask
        mask = (t >= t_now) & (t < t_now + T_active)
        if not np.any(mask):
            t_now += T_total
            k += 1
            continue

        # 3. 生成非对称波形
        relative_t_warped = ((t[mask] - t_now) / T_active) ** asymmetry_factor
        phase = np.pi * relative_t_warped
        
        P[mask] = current_M * np.sin(phase)
        
        t_now += T_total
        k += 1
    
    # 4. 添加基线漂移
    baseline_offset = (baseline_amplitude + 0.01) * np.sin(np.pi * baseline_freq * t)  # 0.05 Hz的基线漂移
    P += baseline_offset
    
    return P

P_true = generate_complex_periodic(t, M_base, M_drift=0.2, M_fluctuation_std=0.1,
                                    T_active_base=T_active_base, T_rest_base=T_rest_base,
                                    T_jitter_std=0.5, asymmetry_factor=1.2,
                                    baseline_amplitude=0.1, baseline_freq=0.05)

# 生成复杂异常 A(t)
# hill shape anomalies with jitter
A_true = np.zeros_like(t)
A_true_hill = np.zeros_like(t)
random_seed = 2
np.random.seed(random_seed)  # 固定随机种子以便复现

for hill_num in range(4):
    for spike_cluster in range(4): # 6 for anomaly, 4 for complex period
        A_true = np.zeros_like(t)  # 重置异常信号

        # hill shape anomalies
        for _ in range(hill_num):
            center_idx = np.random.randint(0, len(t))
            width = np.random.randint(30, 100)  # 宽度30到100个点
            start_idx = max(center_idx - width//2, 0)
            end_idx = min(center_idx + width//2, len(t))
            span = end_idx - start_idx
            # peak_height = np.random.uniform(4, 8)
            peak_height = np.random.uniform(1, 3)
            
            # 创建hill shape
            local_t = np.linspace(-1, 1, span)
            hill_shape = peak_height * (1 - local_t**2)
            
            # 加上小抖动
            jitter = np.random.normal(0, 0.2, size=span)
            hill_with_jitter = hill_shape + jitter
            
            A_true[start_idx:end_idx] += hill_with_jitter

        # spike shape anomalies
        for _ in range(spike_cluster):
            spike_num = np.random.randint(50, 200)  # 每个异常尖峰簇中的尖峰数量
            spike_idx = np.random.randint(50, len(t))  # 每个尖峰簇的中心位置
            for spike in range(spike_num):
                # 在中心位置附近随机生成尖峰
                offset = np.random.randint(-50, 50)
                spike_idx_offset = max(0, min(spike_idx + offset, len(t) - 1))  # 确保尖峰不会超出边界
                spike_height = np.random.uniform(0.5, 1)  # 尖峰高度
                A_true[spike_idx_offset] += spike_height



        # 生成噪声 N(t)
        N_true = np.random.normal(0, 0.2, size=len(t))

        # 合成观测信号
        X = P_true + A_true + N_true

        # # 储存信号
        # np.save(f'{savedir}h{hill_num}_s{spike_cluster}.npy', X)
        # np.save(f'{savedir_gt}h{hill_num}_s{spike_cluster}_A.npy', A_true)
        # print(f'Saved synthetic signal with hill_num={hill_num}, spike_cluster={spike_cluster} to {savedir}h{hill_num}_s{spike_cluster}.npy')

        # # ---- 绘制结果 ----
        # plt.figure(figsize=(15,5))
        # plt.plot(t, X, label='Observed X (P + A + N)', color='black', linewidth=1)
        # plt.plot(t, P_true, label='True P (Pseudo-periodic)', color='blue', linestyle='--')
        # plt.plot(t, A_true, label='True A (Wide Anomalies with Jitter)', color='red', linestyle='-.')
        # plt.title('Synthetic Signal with Pseudo-periodicity, Tilted and Jittered Anomalies')
        # plt.xlabel('Time')
        # plt.ylabel('Signal Amplitude')
        # plt.legend()
        # plt.grid(True)
        # plt.savefig(f'{savedir}h{hill_num}_s{spike_cluster}.png', dpi=300)
        # plt.close()

        # 储存信号
        np.save(f'{savedir}cpl_h{hill_num}_s{spike_cluster}.npy', X)
        np.save(f'{savedir_gt}cpl_h{hill_num}_s{spike_cluster}_A.npy', A_true)
        print(f'Saved synthetic signal with hill_num={hill_num}, spike_cluster={spike_cluster} to {savedir}cpl_h{hill_num}_s{spike_cluster}.npy')

        # ---- 绘制结果 ----
        plt.figure(figsize=(15,5))
        plt.plot(t, X, label='Observed X (P + A + N)', color='black', linewidth=1)
        plt.plot(t, P_true, label='True P (Pseudo-periodic)', color='blue', linestyle='--')
        plt.plot(t, A_true, label='True A (Wide Anomalies with Jitter)', color='red', linestyle='-.')
        plt.title('Synthetic Signal with Pseudo-periodicity, Tilted and Jittered Anomalies')
        plt.xlabel('Time')
        plt.ylabel('Signal Amplitude')
        plt.legend()
        plt.grid(True)
        plt.savefig(f'{savedir}cpl_h{hill_num}_s{spike_cluster}.png', dpi=300)
        plt.close()
'''
Method 1: Time Series: Seasonal Trend Decomposition using LOESS (STL).
'''
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
import os
from statsmodels.tsa.seasonal import STL
from pandas.plotting import register_matplotlib_converters

# 设置数据目录和保存目录
# simulation
dirpath = '../synthetic_signal/'
savepath_rst = './simulation/method1/result/'
savepath_npy = './simulation/method1/npy/'

# # case
# dirpath = '../241230_vector_npy/'
# savepath_rst = './case/method1/result/'
# savepath_npy = './case/method1/npy/'
files = os.listdir(dirpath)
files.sort()

for file in files:
    # if file.endswith('.npy') and file.startswith('cpl'):
    if file.endswith('.npy'):
        print(f"Processing file: {file}")

        # 获取数据
        sample = os.path.join(dirpath, file)
        df = pd.DataFrame(np.load(sample), columns=['value'])
        X = df['value'].values
        time = np.linspace(0, 100, len(X))  # 时间轴
        stl = STL(df['value'], period = 10, robust = True)
        res = stl.fit()
        trend = res.trend
        season = res.seasonal
        residuals = res.resid

        # 基于残差识别异常点 (使用 3-sigma 法则)
        resid_mean = residuals.mean()
        resid_std = residuals.std()
        anomaly_threshold_upper = resid_mean + 3 * resid_std
        anomaly_threshold_lower = resid_mean - 3 * resid_std
        # anomaly 在非0的时候储存为数值，在=0的时候储存为0，长度和trend，season一致，npy文件
        anomaly = np.where((residuals < anomaly_threshold_lower) | (residuals > anomaly_threshold_upper), residuals, 0.0)
        
        # 将trend, season和异常点存储为 .npy 文件
        np.save(f'{savepath_npy}{file[:-4]}_T.npy', trend)
        np.save(f'{savepath_npy}{file[:-4]}_S.npy', season)
        np.save(f'{savepath_npy}{file[:-4]}_P.npy', trend + season)
        np.save(f'{savepath_npy}{file[:-4]}_A.npy', anomaly)
        np.save(f'{savepath_npy}{file[:-4]}_A_idx.npy', np.where(anomaly != 0)[0])

        # 全局字体设置（非常重要）
        plt.rcParams.update({
        "font.size": 26,          # 基础字体
        "axes.titlesize": 26,     # 子图标题
        "axes.labelsize": 22,
        "xtick.labelsize": 22,
        "ytick.labelsize": 22,
        "legend.fontsize": 22
        })

        # 可视化四行图
        plt.figure(figsize=(14, 9), dpi=300)

        plt.subplot(4, 1, 1)
        plt.plot(time, X, linewidth=1.5)
        plt.title('Original Signal')

        plt.subplot(4, 1, 2)
        plt.plot(time, trend + season, color='green', linewidth=1.5)
        plt.title('Periodic')

        plt.subplot(4, 1, 3)
        plt.plot(time, anomaly, color='red', linewidth=1.5)
        plt.title('Anomaly')

        plt.subplot(4, 1, 4)
        plt.plot(time, X - trend - season - anomaly, color='orange', linewidth=1.5)
        plt.title('Residual')
        plt.xlabel('Time')

        plt.tight_layout()
        plt.subplots_adjust(hspace=0.6)

        plt.savefig(
            f'{savepath_rst}{file[:-4]}_stl_result.png',
            bbox_inches='tight'
        )
        plt.close()
import numpy as np
import matplotlib.pyplot as plt
import os
import torch
from tqdm import tqdm

if __name__ == "__main__":

    # simulation
    # 读取信号文件
    dirpath = './synthetic_signal/'
    filenames = os.listdir(dirpath)
    filenames.sort()  # 确保文件按字母顺序排序

    # 储存图像分解结果文件
    savedir = './synthetic_result/'

    # 储存分解结果的numpy数组
    savedir_npy = './synthetic_signal_npy/'


    # # case
    # # 读取信号文件
    # dirpath = './241230_vector_npy/'
    # filenames = os.listdir(dirpath)
    # filenames.sort()  # 确保文件按字母顺序排序

    # # 储存图像分解结果文件
    # savedir = './241230_decomposed/'

    # # 储存分解结果的numpy数组
    # savedir_npy = './241230_decomposed_npy/'

    for file in tqdm(filenames):
        if file.endswith('.npy'):
            # 读取信号
            X = np.load(os.path.join(dirpath, file))
            time = np.linspace(0, 100, len(X))  # 时间轴
            base_filename = file[:-4]  # 去掉.npy后缀
            P = np.load(os.path.join(savedir_npy, f'{base_filename}_P.npy'))
            A = np.load(os.path.join(savedir_npy, f'{base_filename}_A.npy'))

            # 全局字体设置（非常重要）
            plt.rcParams.update({
            "font.size": 26,          # 基础字体
            "axes.titlesize": 26,     # 子图标题
            "axes.labelsize": 22,
            "xtick.labelsize": 22,
            "ytick.labelsize": 22,
            "legend.fontsize": 22
            })
            plt.figure(figsize=(14, 9), dpi=300)

            plt.subplot(4, 1, 1)
            plt.plot(time, X, label='Original Signal', linewidth=1.5)
            plt.title('Original Signal')

            plt.subplot(4, 1, 2)
            plt.plot(time, P, label='Periodic', color='green', linewidth=1.5)
            plt.title('Periodic')

            plt.subplot(4, 1, 3)
            plt.plot(time, A, label='Anomaly', color='red', linewidth=1.5)
            plt.title('Anomaly')

            plt.subplot(4, 1, 4)
            plt.plot(time, X - P - A, label='Residual', color='orange', linewidth=1.5)
            plt.title('Residual')

            plt.tight_layout()
            plt.subplots_adjust(hspace=0.6)
            if savedir == './synthetic_result/':
                plt.savefig(os.path.join(savedir, file[:-4] + '_result.png'), bbox_inches='tight')
            else:
                plt.savefig(os.path.join(savedir, file[:-4] + '.png'), bbox_inches='tight')
            plt.close()
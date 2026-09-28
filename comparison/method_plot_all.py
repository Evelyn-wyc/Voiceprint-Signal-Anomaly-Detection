import numpy as np
from scipy.linalg import toeplitz, cholesky, solve_toeplitz
from scipy.fft import fft, ifft
from scipy.optimize import minimize
import os
from matplotlib import pyplot as plt
from tqdm import tqdm

def filepath(num, data):
    if data == 'simulation':
        dirpath = '../synthetic_signal/'
        savepath_rst = f'./simulation/method{num}/result/'
        savepath_npy = f'./simulation/method{num}/npy/'
    elif data == 'case':
        dirpath = '../241230_vector_npy/'
        savepath_rst = f'./case/method{num}/result/'
        savepath_npy = f'./case/method{num}/npy/'
    return dirpath, savepath_rst, savepath_npy

if __name__ == '__main__':
    
    # 遍历所有method和data类型的组合
    # method_nums = [1, 2, 3]
    # data_types = ['simulation','case']
    method_nums = [3]
    data_types = ['simulation']
    
    for method_num in method_nums:
        for data_type in data_types:
            print(f"\n{'='*50}")
            print(f"Processing method{method_num} with {data_type}...")
            print(f"{'='*50}")
            
            # Read the data from savepath_npy
            dirpath, savepath_rst, savepath_npy = filepath(method_num, data_type)
            
            # 检查目录是否存在
            if not os.path.exists(dirpath):
                print(f"  Warning: Directory {dirpath} does not exist, skipping...")
                continue
            
            files = os.listdir(dirpath)
            files.sort()
            
            if len(files) == 0:
                print(f"  Warning: No files found in {dirpath}, skipping...")
                continue

            for file in tqdm(files, desc=f"method{method_num}-{data_type}"):
                if file.endswith('.npy'):
                    try:
                        Y_data = np.load(os.path.join(dirpath, file))
                        n_points = len(Y_data)
                        time = np.linspace(0, 100, n_points)  # 时间轴

                        periodic_component = np.load(os.path.join(savepath_npy, file[:-4] + '_P.npy'))
                        A = np.load(os.path.join(savepath_npy, file[:-4] + '_A.npy'))

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
                        plt.plot(time, Y_data, label='Original Signal', linewidth=1.5)
                        plt.title('Original Signal')

                        plt.subplot(4, 1, 2)
                        plt.plot(time, periodic_component, label='Periodic', color='green', linewidth=1.5)
                        plt.title('Periodic')

                        plt.subplot(4, 1, 3)
                        plt.plot(time, A, label='Anomaly', color='red', linewidth=1.5)
                        plt.title('Anomaly')

                        plt.subplot(4, 1, 4)
                        plt.plot(time, Y_data - periodic_component - A, label='Residual', color='orange', linewidth=1.5)
                        plt.title('Residual')

                        plt.tight_layout()
                        plt.subplots_adjust(hspace=0.6)
                        
                        # 创建保存目录（如果不存在）
                        os.makedirs(savepath_rst, exist_ok=True)
                        if method_num == 1:
                            plt.savefig(os.path.join(savepath_rst, file[:-4] + '_stl_result.png'), bbox_inches='tight')
                        elif method_num == 2:
                            plt.savefig(os.path.join(savepath_rst, file[:-4] + '_vmd_result.png'), bbox_inches='tight')
                        elif method_num == 3:
                            plt.savefig(os.path.join(savepath_rst, file[:-4] + '_QRGP_result.png'), bbox_inches='tight')
                        plt.close()
                    except Exception as e:
                        print(f"  Error processing {file}: {e}")
                        continue
            
            print(f"Completed method{method_num} with {data_type}")
    
    print("\n" + "="*50)
    print("All processing completed!")
    print("="*50)

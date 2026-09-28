'''
Decompose the vecmode=1 vector signal into a normal (sometimes periodic, others stationary) and residual part
Using a joint optimization approach with periodic and anomaly components
支持4卡GPU并行处理
'''
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
from joblib import Parallel, delayed
import pdb

# 定义模型    
class Periodic(nn.Module):
    def __init__(self, K_init, signal_length):
        super().__init__()
        self.K = K_init  # 初始周期数
        self.M = nn.Parameter(torch.ones(signal_length))  # 幅度调制函数
        self.M_independent = nn.Parameter(torch.ones(K_init))  # 每个周期的独立幅度
        self.T_active = nn.Parameter(torch.ones(K_init)*100)  # 每个周期的持续时间
        self.t_k = nn.Parameter(torch.linspace(0, signal_length, K_init+1)[:-1])  # 周期起始时间
        self.A = nn.Parameter(torch.zeros(signal_length))  # 异常信号

    def forward(self):
        device = self.M.device
        t = torch.arange(len(self.M), device=device).float().unsqueeze(0)  # [1, T]
        t_k = self.t_k.unsqueeze(1)  # [K, 1]
        T_active = self.T_active.unsqueeze(1)  # [K, 1]
        relative_t = (t - t_k) / (T_active + 1e-6)  # [K, T]

        # smooth mask: cosine window for periodic situation
        mask_start = torch.cos(relative_t * np.pi / 2)  # [K, T]
        mask_end = torch.cos((1 - relative_t) * np.pi / 2)  # [K, T]
        mask = mask_start * mask_end  # shape: [K, T]
        mask = torch.clamp(mask, min=0, max=1)  # 确保mask在[0, 1]范围内

        # 合成周期信号
        relative_t = (t - t_k) / (T_active + 1e-6)  # [K, T]
        phase = torch.pi * relative_t  # [K, T]
        P_k = self.M_independent.unsqueeze(1) * mask  # 每个周期段乘以独立幅度

        P = P_k.sum(dim=0)  # 聚合所有周期段
        A = torch.nn.functional.leaky_relu(self.A, negative_slope=0.01)  # 确保异常信号为非负

        return P, A

def extract_period_segments(P, t_k, T_active):
    segments = []
    for k in range(len(t_k)):
        start = int(t_k[k].item())
        end = int(start + T_active[k].item())
        if end > len(P):  # 边界裁剪
            end = len(P)
        seg = P[start:end]
        if len(seg) > 5:  # 过滤过短片段
            segments.append(seg)
    return segments


from scipy.signal import find_peaks

class JointOptimizer:
    def __init__(self, X, params, device_id=0):
        self.device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")
        # 初始化模型
        self.periodic_model = Periodic(K_init=10, signal_length=len(X)).to(self.device)
        self.X = torch.tensor(X, device=self.device, dtype=torch.float32)

        # 超参数配置
        self.lambda1 = params.get('lambda1', 0.5) # 异常L1系数
        self.lambda2 = params.get('lambda2', 0.2) # 异常TV系数
        self.eta1 = params.get('eta1', 0.1)    # M的TV系数
        self.eta2 = params.get('eta2', 0.1)    # M的二阶平滑系数
        self.eta3 = params.get('eta3', 0.1)    # T_active的TV系数
        self.psi = params.get('psi', 10)    # 周期项相似性系数
        
    def compute_loss(self):
        # 各分量预测值
        P, A = self.periodic_model()
        t_k = self.periodic_model.t_k
        T_active = self.periodic_model.T_active
        segments = extract_period_segments(P, t_k, T_active)
        
        # 数据保真项 (l1)
        recon = P + A
        loss_data = torch.norm(self.X - recon, p=2)**2
                
        # 异常项 (l2)
        loss_A_l1 = self.lambda1 * torch.norm(A, p=1)
        A_diff = torch.diff(A)
        loss_A_tv = self.lambda2 * torch.norm(A_diff, p=1)
        
        # 周期参数 (l3)
        M_diff = torch.diff(self.periodic_model.M_independent)
        loss_M_tv = self.eta1 * torch.norm(M_diff, p=1)
        M_2_diff = torch.diff(self.periodic_model.M_independent, 2)
        loss_M_2_smooth = self.eta2 * torch.norm(M_2_diff, p=1)
        T_diff = torch.diff(self.periodic_model.T_active)
        loss_T_tv = self.eta3 * torch.norm(T_diff, p=1)
        loss_prd = loss_M_tv + loss_M_2_smooth + loss_T_tv

        # 周期波动项相似性 (l4)：对于相邻的两个active，要求形状相似
        loss_similarity = 0
        for i in range(len(segments)-1):
            seg1 = segments[i]
            seg2 = segments[i+1]
            min_len = min(len(seg1), len(seg2))
            if min_len > 5:  # 过滤过短片段
                loss_similarity += torch.norm(seg1[:min_len] - seg2[:min_len], p=2) ** 2
        loss_similarity = self.psi * loss_similarity
        
        # 总损失
        total_loss = (loss_data + loss_A_l1 + loss_A_tv + loss_prd + loss_similarity) / len(self.X)
        
        return total_loss

    def alternating_optimization(self, epochs=100):
        # 文件名
        current_file = getattr(self, 'current_file', 'signal')
        epoch_bar = tqdm(range(epochs), desc=f"Processing {current_file}")

        # 为每个组件定义优化器
        # period_optim = optim.Adam(self.periodic_model.parameters(), lr=0.01, amsgrad=True)
        period_optim = optim.Adam(self.periodic_model.parameters(), lr=0.01)

        for epoch in epoch_bar:

            # 优化周期项
            period_optim.zero_grad()
            loss = self.compute_loss() * 10
            loss.backward()
            period_optim.step()
            epoch_bar.set_postfix(loss=f"{loss.item(): .4f}")

    def get_components(self):
        with torch.no_grad():
            P, A = self.periodic_model()
            print("T_active: ", self.periodic_model.T_active, 
                  "t_k: ", self.periodic_model.t_k, 
                  "M: ", self.periodic_model.M_independent,
                  "A: ", self.periodic_model.A)
        return P.cpu().numpy(), A.cpu().numpy()
    
    def get_anomaly_score(self):
        """获取异常分数"""
        with torch.no_grad():
            P, A = self.periodic_model()
        return A.cpu().numpy()


def process_single_signal(idx, file, filedir, savedir_npy, savedir_res, params, num_gpus):
    """处理单个信号的函数（用于joblib并行）
    
    Args:
        idx: 信号索引（用于GPU轮值）
        file: 文件名
        filedir: 输入目录
        savedir_npy: numpy保存目录
        savedir_res: 结果保存目录
        params: 优化参数
        num_gpus: GPU总数
    
    Returns:
        str: 处理结果信息
    """
    gpu_id = idx % num_gpus
    
    try:
        X = np.load(os.path.join(filedir, file))
        time = np.linspace(0, 100, len(X))
        
        # 创建优化器实例（使用指定的GPU）
        optimizer = JointOptimizer(X.copy(), params, device_id=gpu_id)
        optimizer.current_file = file
        optimizer.alternating_optimization(epochs=2000)
        P, A = optimizer.get_components()
        
        # 保存结果
        base_filename = file[:-4]
        p_save_path = os.path.join(savedir_npy, f'{base_filename}_P.npy')
        a_save_path = os.path.join(savedir_npy, f'{base_filename}_A.npy')
        np.save(p_save_path, P)
        np.save(a_save_path, A)

        # 全局字体设置（非常重要）
        plt.rcParams.update({
        "font.size": 26,          # 基础字体
        "axes.titlesize": 26,     # 子图标题
        "axes.labelsize": 22,
        "xtick.labelsize": 22,
        "ytick.labelsize": 22,
        "legend.fontsize": 22
        })

        # 可视化结果
        plt.figure(figsize=(14, 9), dpi = 300)
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
        plt.savefig(os.path.join(savedir_res, f'{base_filename}_result.png'), bbox_inches='tight')
        plt.close()
        
        return f"✓ {file} [GPU {gpu_id}]"
    
    except Exception as e:
        return f"✗ {file} - Error: {str(e)}"


if __name__ == "__main__":

    # 读取仿真信号
    filedir = './synthetic_signal/'
    filenames = os.listdir(filedir)
    filenames.sort()  # 确保文件名排序
    savedir_npy = './synthetic_signal_npy/'
    savedir_res = './synthetic_result/'
    
    if not os.path.exists(filedir):
        os.makedirs(filedir)
    if not os.path.exists(savedir_res):
        os.makedirs(savedir_res)
    if not os.path.exists(savedir_npy):
        os.makedirs(savedir_npy)
    
    # 参数配置
    params = {
        'lambda1': 1.5, # 异常L1系数
        'lambda2': 1.0, # 异常TV系数
        'eta1': 0.15, # M的TV系数
        'eta2': 0.15, # M的二阶平滑系数
        'eta3': 0.05, # T_active的TV系数
        'psi': 2 # 周期项相似性系数
    }
    
    # 获取所有需要处理的.npy文件
    signal_files = sorted([f for f in filenames if f.endswith('.npy')])
    
    # GPU配置
    num_gpus = 4
    num_available_gpus = torch.cuda.device_count()
    num_gpus = min(num_gpus, num_available_gpus) if num_available_gpus > 0 else 1
    
    print(f"Found {len(signal_files)} signal files")
    print(f"Available GPUs: {num_available_gpus}, Using: {num_gpus}")
    print("="*60)
    
    # 使用joblib进行4卡GPU并行处理
    print("Starting parallel processing...\n")
    results = Parallel(n_jobs=num_gpus, verbose=10)(
        delayed(process_single_signal)(idx, file, filedir, savedir_npy, savedir_res, params, num_gpus) 
        for idx, file in enumerate(signal_files)
    )
    
    print("\n" + "="*60)
    print("Processing Complete!")
    print("="*60)
    for result in results:
        print(result)

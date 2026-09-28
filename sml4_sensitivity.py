'''
参数敏感性分析脚本
分析异常(lambda1, lambda2)和周期(eta1, eta2, eta3)参数对AUROC和F1-score的影响
固定psi=2，通过缩放因子进行网格搜索（n=3），并绘制热力图（3*3网格）

四卡GPU并行运行说明：
- 使用joblib.Parallel进行多进程并行处理
- 支持自动GPU分配：4个并行工作进程轮询使用4张GPU
- 可修改 n_gpus 和 n_jobs 参数来调整并行度
- 每个信号的分析完全独立，适合并行化

直接运行sml4_sensitivity.py即可，包含画图和保存结果
单个文件单卡25-30秒，4卡并行，搜索9次，50个文件大约1个小时。
'''
import numpy as np
import torch
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import seaborn as sns
from joblib import Parallel, delayed
from sml2_decompose import JointOptimizer
from sml3_anomaly import calculate_anomaly_metrics

def process_single_signal(signal_file, signal_dir, gt_dir, save_dir, gpu_id=0):
    """
    处理单个信号的敏感性分析
    
    Args:
        signal_file: 信号文件名
        signal_dir: 信号目录
        gt_dir: 标签目录
        save_dir: 保存目录
        gpu_id: GPU设备ID（用于多GPU并行）
    
    Returns:
        dict: 包含分析结果和信号类型的字典
    """
    # 设置GPU
    torch.cuda.set_device(gpu_id)
    
    # 跳过基准信号
    if signal_file == 'cpl_h0_s0.npy' or signal_file == 'h0_s0.npy':
        return None
    
    signal_path = os.path.join(signal_dir, signal_file)
    base_name = signal_file[:-4]  # 去掉.npy后缀
    
    # 对应的真实异常标签文件
    gt_file = f'{base_name}_A.npy'
    gt_path = os.path.join(gt_dir, gt_file)
    
    # 检查标签文件是否存在
    if not os.path.exists(gt_path):
        print(f"[GPU {gpu_id}] 警告：找不到标签文件 {gt_path}，跳过 {signal_file}")
        return None
    
    print(f"[GPU {gpu_id}] 处理信号: {signal_file}")
    
    try:
        # 加载信号和标签
        X = np.load(signal_path)
        A_true = np.load(gt_path)
        
        # 生成二进制标签 (异常阈值设为0)
        y_true = (A_true > 0).astype(int)
        
        # 如果信号太长，进行下采样以加快处理
        if len(X) > 2000:
            downsample_factor = len(X) // 1000
            X = X[::downsample_factor]
            y_true = y_true[::downsample_factor]
        
        # 运行敏感性分析
        analyzer = SensitivityAnalyzer(X, y_true=y_true, n_scales=3, 
                                       base_name=base_name,
                                       base_save_dir='./sensitivity/',
                                       gpu_id=gpu_id)
        
        # 执行网格搜索
        analyzer.grid_search()
        
        # 绘制热力图
        analyzer.plot_heatmaps(save_dir=save_dir, name_prefix=base_name)
        
        # 打印摘要
        analyzer.print_summary()
        
        # 判断信号类型
        signal_type = None
        if base_name.startswith('cpl'):
            signal_type = 'cpl'
        elif base_name.startswith('h'):
            signal_type = 'h'
        
        # 返回结果
        return {
            'base_name': base_name,
            'auroc_matrix': analyzer.auroc_matrix,
            'f1_matrix': analyzer.f1_matrix,
            'signal_type': signal_type
        }
    
    except Exception as e:
        print(f"[GPU {gpu_id}] 处理信号 {signal_file} 失败: {e}")
        import traceback
        traceback.print_exc()
        return None

# 基准参数
BASELINE_PARAMS = {
    'lambda1': 1.5,   # 异常L1系数
    'lambda2': 1.0,   # 异常TV系数
    'eta1': 0.15,     # M的TV系数
    'eta2': 0.15,     # M的二阶平滑系数
    'eta3': 0.05,     # T_active的TV系数
    'psi': 2          # 周期项相似性系数（固定）
}

class SensitivityAnalyzer:
    def __init__(self, X, y_true=None, n_scales=3, base_name=None, base_save_dir='./sensitivity/', gpu_id=0):
        """
        参数敏感性分析器
        
        Args:
            X: 输入信号
            y_true: 真实异常标签（可选）
            n_scales: 缩放因子的数量（会生成n_scales x n_scales的热力图）
            base_name: 信号基础名称（用于保存P和A文件）
            base_save_dir: 基础保存目录（默认./sensitivity/）
            gpu_id: GPU设备ID（默认0）
        """
        self.X = X
        self.y_true = y_true
        self.n_scales = n_scales
        self.base_name = base_name
        self.base_save_dir = base_save_dir
        self.gpu_id = gpu_id
        
        # 定义缩放范围 [0.5, 2.0]
        self.scale_factors = np.linspace(0.5, 2.0, n_scales)
        
        # 结果存储
        self.P = None
        self.A = None
        self.auroc_matrix = np.zeros((n_scales, n_scales))
        self.f1_matrix = np.zeros((n_scales, n_scales))
        
    def run_experiment(self, scale_anomaly, scale_periodic):
        """
        运行单次实验
        
        Args:
            scale_anomaly: 异常参数缩放因子
            scale_periodic: 周期参数缩放因子
            params_idx: 参数组合索引
        
        Returns:
            auroc, f1_score 或 None 如果没有真实标签
        """
        # 调整参数
        params = BASELINE_PARAMS.copy()
        params['lambda1'] *= scale_anomaly
        params['lambda2'] *= scale_anomaly
        params['eta1'] *= scale_periodic
        params['eta2'] *= scale_periodic
        params['eta3'] *= scale_periodic
        # psi保持固定
        params['psi'] = 10
        
        # 运行优化（设置GPU设备）
        if self.gpu_id is not None:
            torch.cuda.set_device(self.gpu_id)
        optimizer = JointOptimizer(self.X, params)
        optimizer.alternating_optimization(epochs=2000)
        
        # 获取异常
        P, A = optimizer.get_components()
        
        # 计算指标（如果有真实标签）
        auroc = None
        f1 = None
        if self.y_true is not None:
            try:
                # 使用sml3中的计算函数获取完整指标
                metrics = calculate_anomaly_metrics(self.y_true, A)
                auroc = metrics['AUROC']
                f1 = metrics['F1 Score']
            except Exception as e:
                print(f"计算指标失败: {e}")
                auroc = 0
                f1 = 0
        else:
            # 如果没有真实标签，使用异常分数的平均值作为代理指标
            auroc = np.mean(A)
            f1 = np.std(A)
        
        return P, A, auroc, f1
    
    def grid_search(self):
        """执行网格搜索"""
        pbar = tqdm(total=self.n_scales*self.n_scales, desc="Grid Search")
        
        for i, scale_anom in enumerate(self.scale_factors):
            for j, scale_prd in enumerate(self.scale_factors):
                try:
                    P, A, auroc, f1 = self.run_experiment(scale_anom, scale_prd)
                    self.P = P
                    self.A = A
                    self.auroc_matrix[i, j] = auroc if auroc is not None else 0
                    self.f1_matrix[i, j] = f1 if f1 is not None else 0
                    
                    # 保存P和A到指定文件夹
                    if self.base_name is not None:
                        # 创建文件夹名称: A{scale_anomaly}_P{scale_periodic}
                        folder_name = f'A{scale_anom:.4f}_P{scale_prd:.4f}'
                        save_path = os.path.join(self.base_save_dir, folder_name)
                        
                        # 检查并创建文件夹
                        if not os.path.exists(save_path):
                            os.makedirs(save_path, exist_ok=True)
                        
                        # 保存P和A
                        p_file = os.path.join(save_path, f'{self.base_name}_P.npy')
                        a_file = os.path.join(save_path, f'{self.base_name}_A.npy')
                        np.save(p_file, P)
                        np.save(a_file, A)
                        
                except Exception as e:
                    print(f"实验失败 (scale_anom={scale_anom:.2f}, scale_prd={scale_prd:.2f}): {e}")
                    self.auroc_matrix[i, j] = 0
                    self.f1_matrix[i, j] = 0
                
                pbar.update(1)
        
        pbar.close()
    
    def plot_heatmaps(self, save_dir='./sensitivity_results/', name_prefix=None):
        """绘制热力图
        
        Args:
            save_dir: 保存目录
            name_prefix: 文件名前缀（如果为None，则使用默认命名）
        """
        # 字体设置
        plt.rcParams.update({
            "font.size": 26,          # 基础字体
            "axes.titlesize": 26,     # 子图标题
            "axes.labelsize": 22,
            "xtick.labelsize": 22,
            "ytick.labelsize": 22,
            "legend.fontsize": 22
        })
        
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        # 准备坐标轴标签
        scale_labels = [f'{s:.2f}' for s in self.scale_factors]
        
        # 确定文件名
        if name_prefix is None:
            auroc_filename = 'sensitivity_auroc_heatmap.png'
            f1_filename = 'sensitivity_f1_heatmap.png'
        else:
            auroc_filename = f'{name_prefix}_AUROC.png'
            f1_filename = f'{name_prefix}_F1.png'
        
        # 绘制AUROC热力图
        plt.figure(figsize=(10, 8))
        sns.heatmap(self.auroc_matrix, 
                    xticklabels=scale_labels,
                    yticklabels=scale_labels,
                    cmap='YlOrRd',
                    annot=True,
                    fmt='.3f',
                    cbar_kws={'label': 'AUROC'})
        plt.xlabel('Periodic Scale Factor (eta1, eta2, eta3)', fontsize=18)
        plt.ylabel('Anomaly Scale Factor (lambda1, lambda2)', fontsize=18)
        plt.title('Parameter Sensitivity Analysis - AUROC', fontsize=22, fontweight='bold')
        plt.tight_layout()
        auroc_path = os.path.join(save_dir, auroc_filename)
        plt.savefig(auroc_path, dpi=300)
        print(f"AUROC热力图已保存: {auroc_path}")
        plt.close()
        
        # 绘制F1-score热力图
        plt.figure(figsize=(10, 8))
        sns.heatmap(self.f1_matrix,
                    xticklabels=scale_labels,
                    yticklabels=scale_labels,
                    cmap='YlOrRd',
                    annot=True,
                    fmt='.3f',
                    cbar_kws={'label': 'F1-Score'})
        plt.xlabel('Periodic Scale Factor (eta1, eta2, eta3)', fontsize=18)
        plt.ylabel('Anomaly Scale Factor (lambda1, lambda2)', fontsize=18)
        plt.title('Parameter Sensitivity Analysis - F1-Score', fontsize=22, fontweight='bold')
        plt.tight_layout()
        f1_path = os.path.join(save_dir, f1_filename)
        plt.savefig(f1_path, dpi=300)
        print(f"F1-score热力图已保存: {f1_path}")
        plt.close()
        
        # 保存数值结果
        if name_prefix:
            np.save(os.path.join(save_dir, f'{name_prefix}_auroc_matrix.npy'), self.auroc_matrix)
            np.save(os.path.join(save_dir, f'{name_prefix}_f1_matrix.npy'), self.f1_matrix)
    
    def print_summary(self):
        """打印分析摘要"""
        print("\n" + "="*60)
        print("参数敏感性分析摘要")
        print("="*60)
        print(f"缩放因子数量: {self.n_scales}")
        print(f"缩放因子范围: {self.scale_factors[0]:.2f} - {self.scale_factors[-1]:.2f}")
        print(f"\n基准参数:")
        for key, value in BASELINE_PARAMS.items():
            print(f"  {key}: {value}")
        print(f"\nAUROC统计:")
        print(f"  最大值: {np.max(self.auroc_matrix):.4f}")
        print(f"  最小值: {np.min(self.auroc_matrix):.4f}")
        print(f"  平均值: {np.mean(self.auroc_matrix):.4f}")
        best_idx = np.unravel_index(np.argmax(self.auroc_matrix), self.auroc_matrix.shape)
        print(f"  最优参数: Anomaly scale={self.scale_factors[best_idx[0]]:.2f}, Periodic scale={self.scale_factors[best_idx[1]]:.2f}")
        
        print(f"\nF1-Score统计:")
        print(f"  最大值: {np.max(self.f1_matrix):.4f}")
        print(f"  最小值: {np.min(self.f1_matrix):.4f}")
        print(f"  平均值: {np.mean(self.f1_matrix):.4f}")
        best_idx_f1 = np.unravel_index(np.argmax(self.f1_matrix), self.f1_matrix.shape)
        print(f"  最优参数: Anomaly scale={self.scale_factors[best_idx_f1[0]]:.2f}, Periodic scale={self.scale_factors[best_idx_f1[1]]:.2f}")
        print("="*60 + "\n")


if __name__ == "__main__":
    
    # 从sml1_simulation.py生成的文件夹读取数据
    signal_dir = './synthetic_signal/'
    gt_dir = './synthetic_gtanomaly/'
    save_dir = './sensitivity_results/'
    
    # GPU配置
    n_gpus = 4  # 使用4卡GPU
    n_jobs = 4  # 4个并行工作
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # 获取所有信号文件
    signal_files = sorted([f for f in os.listdir(signal_dir) if f.endswith('.npy')])
    
    print(f"找到 {len(signal_files)} 个信号文件")
    print(f"使用 {n_gpus} 卡GPU进行并行处理")
    print("="*60)
    
    # 使用joblib.Parallel进行并行处理（4卡并行）
    print("开始并行处理信号...")
    results = Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(process_single_signal)(
            signal_file,
            signal_dir,
            gt_dir,
            save_dir,
            gpu_id=i % n_gpus  # 轮询分配GPU
        )
        for i, signal_file in enumerate(signal_files)
    )
    
    print("\n" + "="*60)
    print("所有信号的并行处理完成！")
    print("="*60)
    
    # 收集结果
    all_results = {}
    cpl_results = []  # 存储cpl开头的分析结果
    hs_results = []   # 存储h开头的分析结果
    
    for result in results:
        if result is None:
            continue
        
        all_results[result['base_name']] = {
            'auroc_matrix': result['auroc_matrix'],
            'f1_matrix': result['f1_matrix']
        }
        
        # 分类存储结果
        if result['signal_type'] == 'cpl':
            cpl_results.append((result['base_name'], result['auroc_matrix'], result['f1_matrix']))
        elif result['signal_type'] == 'h':
            hs_results.append((result['base_name'], result['auroc_matrix'], result['f1_matrix']))
    
    print("\n" + "="*60)
    print("所有信号的敏感性分析完成！")
    print(f"总共处理了 {len(all_results)} 个信号")
    print("="*60)
    
    # 计算并保存cpl_mean结果
    if cpl_results:
        print("\n" + "="*60)
        print("计算cpl信号的平均结果...")
        print("="*60)
        cpl_auroc_mean = np.mean([r[1] for r in cpl_results], axis=0)
        cpl_f1_mean = np.mean([r[2] for r in cpl_results], axis=0)
        
        # 创建临时analyzer对象用于绘制平均热力图（不需要保存P和A）
        temp_analyzer = SensitivityAnalyzer(np.zeros(100), y_true=None, n_scales=3)
        temp_analyzer.auroc_matrix = cpl_auroc_mean
        temp_analyzer.f1_matrix = cpl_f1_mean
        temp_analyzer.plot_heatmaps(save_dir=save_dir, name_prefix='cpl_mean')
        
        print(f"cpl_mean - AUROC: 最大值={np.max(cpl_auroc_mean):.4f}, 最小值={np.min(cpl_auroc_mean):.4f}, 平均值={np.mean(cpl_auroc_mean):.4f}")
        print(f"cpl_mean - F1-Score: 最大值={np.max(cpl_f1_mean):.4f}, 最小值={np.min(cpl_f1_mean):.4f}, 平均值={np.mean(cpl_f1_mean):.4f}")
    
    # 计算并保存hs_mean结果
    if hs_results:
        print("\n" + "="*60)
        print("计算h开头信号的平均结果...")
        print("="*60)
        hs_auroc_mean = np.mean([r[1] for r in hs_results], axis=0)
        hs_f1_mean = np.mean([r[2] for r in hs_results], axis=0)
        
        # 创建临时analyzer对象用于绘制平均热力图（不需要保存P和A）
        temp_analyzer = SensitivityAnalyzer(np.zeros(100), y_true=None, n_scales=3)
        temp_analyzer.auroc_matrix = hs_auroc_mean
        temp_analyzer.f1_matrix = hs_f1_mean
        temp_analyzer.plot_heatmaps(save_dir=save_dir, name_prefix='hs_mean')
        
        print(f"hs_mean - AUROC: 最大值={np.max(hs_auroc_mean):.4f}, 最小值={np.min(hs_auroc_mean):.4f}, 平均值={np.mean(hs_auroc_mean):.4f}")
        print(f"hs_mean - F1-Score: 最大值={np.max(hs_f1_mean):.4f}, 最小值={np.min(hs_f1_mean):.4f}, 平均值={np.mean(hs_f1_mean):.4f}")
    
    print("\n" + "="*60)
    print("敏感性分析及聚合完成！")
    print(f"所有结果已保存到: {save_dir}")
    print("="*60)

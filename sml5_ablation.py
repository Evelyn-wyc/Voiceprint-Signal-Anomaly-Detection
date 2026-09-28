'''
Ablation Study: 分解信号的周期和残差分量
测试不同损失函数组件的影响，支持4卡GPU并行训练

运行sml5_ablation.py --config <config_name> 可单独运行某个配置
或直接运行sml5_ablation.py 可并行运行所有配置
结果保存在 ./ablation_results/ 目录下
运行plot_abl.py绘画出每个数据的origin-period-anomaly-residual对比图
单个文件单卡10-15秒，52个文件（包含h0s0和cpl0）约3-4分钟。
'''
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DataParallel, DistributedDataParallel
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
from joblib import Parallel, delayed
import json
import argparse
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


# 定义模型    
class Periodic(nn.Module):
    def __init__(self, K_init, signal_length):
        super().__init__()
        self.K = K_init
        self.M = nn.Parameter(torch.ones(signal_length))
        self.M_independent = nn.Parameter(torch.ones(K_init))
        self.T_active = nn.Parameter(torch.ones(K_init)*100)
        self.t_k = nn.Parameter(torch.linspace(0, signal_length, K_init+1)[:-1])
        self.A = nn.Parameter(torch.zeros(signal_length))

    def forward(self):
        device = self.M.device
        t = torch.arange(len(self.M), device=device).float().unsqueeze(0)  # [1, T]
        t_k = self.t_k.unsqueeze(1)  # [K, 1]
        T_active = self.T_active.unsqueeze(1)  # [K, 1]
        relative_t = (t - t_k) / (T_active + 1e-6)  # [K, T]

        # smooth mask: cosine window for periodic situation
        mask_start = torch.cos(relative_t * np.pi / 2)
        mask_end = torch.cos((1 - relative_t) * np.pi / 2)
        mask = mask_start * mask_end
        mask = torch.clamp(mask, min=0, max=1)

        relative_t = (t - t_k) / (T_active + 1e-6)
        phase = torch.pi * relative_t
        sinusoid = torch.sin(phase)
        P_k = self.M_independent.unsqueeze(1) * mask

        P = P_k.sum(dim=0)
        A = torch.nn.functional.leaky_relu(self.A, negative_slope=0.01)

        return P, A
    
    
def extract_period_segments(P, t_k, T_active):
    segments = []
    for k in range(len(t_k)):
        start = int(t_k[k].item())
        end = int(start + T_active[k].item())
        if end > len(P):
            end = len(P)
        seg = P[start:end]
        if len(seg) > 5:
            segments.append(seg)
    return segments


class JointOptimizer:
    def __init__(self, X, params, device, rank=0):
        self.device = device
        self.rank = rank
        self.periodic_model = Periodic(K_init=10, signal_length=len(X)).to(self.device)
        self.X = torch.tensor(X, device=self.device, dtype=torch.float32)
        
        # 超参数配置 - 支持ablation实验的启用/禁用
        self.lambda1 = params.get('lambda1', 1.5)
        self.lambda2 = params.get('lambda2', 1.0)
        self.eta1 = params.get('eta1', 0.3)
        self.eta2 = params.get('eta2', 0.3)
        self.eta3 = params.get('eta3', 0.1)
        self.psi = params.get('psi', 10)
        
        # Ablation标志
        self.enable_A_l1 = params.get('enable_A_l1', True)
        self.enable_A_tv = params.get('enable_A_tv', True)
        self.enable_M_tv = params.get('enable_M_tv', True)
        self.enable_M_2smooth = params.get('enable_M_2smooth', True)
        self.enable_T_tv = params.get('enable_T_tv', True)
        self.enable_similarity = params.get('enable_similarity', True)
        
        self.loss_history = []
        
    def compute_loss(self):
        P, A = self.periodic_model()
        t_k = self.periodic_model.t_k
        T_active = self.periodic_model.T_active
        segments = extract_period_segments(P, t_k, T_active)
        
        # 数据保真项 (必需)
        recon = P + A
        loss_data = torch.norm(self.X - recon, p=2)**2
        
        loss_dict = {'data': loss_data.item()}
        total_loss = loss_data
                
        # 异常L1稀疏项
        if self.enable_A_l1:
            loss_A_l1 = self.lambda1 * torch.norm(A, p=1)
            total_loss += loss_A_l1
            loss_dict['A_l1'] = loss_A_l1.item()
        
        # 异常TV平滑项
        if self.enable_A_tv:
            A_diff = torch.diff(A)
            loss_A_tv = self.lambda2 * torch.norm(A_diff, p=1)
            total_loss += loss_A_tv
            loss_dict['A_tv'] = loss_A_tv.item()
        
        # 周期参数M的TV项
        if self.enable_M_tv:
            M_diff = torch.diff(self.periodic_model.M_independent)
            loss_M_tv = self.eta1 * torch.norm(M_diff, p=1)
            total_loss += loss_M_tv
            loss_dict['M_tv'] = loss_M_tv.item()
        
        # 周期参数M的二阶平滑项
        if self.enable_M_2smooth:
            M_2_diff = torch.diff(self.periodic_model.M_independent, 2)
            loss_M_2_smooth = self.eta2 * torch.norm(M_2_diff, p=1)
            total_loss += loss_M_2_smooth
            loss_dict['M_2smooth'] = loss_M_2_smooth.item()
        
        # 周期时间T_active的TV项
        if self.enable_T_tv:
            T_diff = torch.diff(self.periodic_model.T_active)
            loss_T_tv = self.eta3 * torch.norm(T_diff, p=1)
            total_loss += loss_T_tv
            loss_dict['T_tv'] = loss_T_tv.item()

        # 周期波动项相似性
        if self.enable_similarity and len(segments) > 1:
            loss_similarity = 0
            for i in range(len(segments)-1):
                seg1 = segments[i]
                seg2 = segments[i+1]
                min_len = min(len(seg1), len(seg2))
                if min_len > 5:
                    loss_similarity += torch.norm(seg1[:min_len] - seg2[:min_len], p=2) ** 2
            loss_similarity = self.psi * loss_similarity
            total_loss += loss_similarity
            loss_dict['similarity'] = loss_similarity.item()
        
        # 归一化总损失
        final_loss = total_loss / len(self.X)
        loss_dict['total'] = final_loss.item()
        
        return final_loss, loss_dict

    def train(self, epochs=2000):
        optim_adam = optim.Adam(self.periodic_model.parameters(), lr=0.01)
        epoch_bar = tqdm(range(epochs), disable=(self.rank != 0))

        for epoch in epoch_bar:
            optim_adam.zero_grad()
            loss, loss_dict = self.compute_loss()
            loss = loss * 10
            loss.backward()
            optim_adam.step()

            self.loss_history.append(loss_dict)
            
            if self.rank == 0:
                epoch_bar.set_postfix({k: f"{v:.4f}" for k, v in loss_dict.items()})

    def get_components(self):
        with torch.no_grad():
            P, A = self.periodic_model()
        return P.cpu().numpy(), A.cpu().numpy()


class AblationExperiment:
    """ablation实验管理类"""
    def __init__(self, signal_dir, output_dir, num_gpus=4, epochs=2000):
        self.signal_dir = signal_dir
        self.output_dir = output_dir
        self.num_gpus = num_gpus
        self.epochs = epochs
        self.results = {}
        
        # 创建输出目录
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
    def get_ablation_configs(self):
        """定义ablation实验配置"""
        base_params = {
            'lambda1': 1.5,
            'lambda2': 1.0,
            'eta1': 0.15,
            'eta2': 0.15,
            'eta3': 0.05,
            'psi': 2
        }
        
        configs = {
            # 1. Full model (基线)
            'full_model': {
                **base_params,
                'enable_A_l1': True,
                'enable_A_tv': True,
                'enable_M_tv': True,
                'enable_M_2smooth': True,
                'enable_T_tv': True,
                'enable_similarity': True,
            },
            
            # 2. 移除异常L1稀疏项
            'w/o_A_l1': {
                **base_params,
                'enable_A_l1': False,
                'enable_A_tv': True,
                'enable_M_tv': True,
                'enable_M_2smooth': True,
                'enable_T_tv': True,
                'enable_similarity': True,
            },
            
            # 3. 移除异常TV平滑项
            'w/o_A_tv': {
                **base_params,
                'enable_A_l1': True,
                'enable_A_tv': False,
                'enable_M_tv': True,
                'enable_M_2smooth': True,
                'enable_T_tv': True,
                'enable_similarity': True,
            },
            
            # 4. 移除周期M的TV项
            'w/o_M_tv': {
                **base_params,
                'enable_A_l1': True,
                'enable_A_tv': True,
                'enable_M_tv': False,
                'enable_M_2smooth': True,
                'enable_T_tv': True,
                'enable_similarity': True,
            },
            
            # 5. 移除周期M的二阶平滑项
            'w/o_M_2smooth': {
                **base_params,
                'enable_A_l1': True,
                'enable_A_tv': True,
                'enable_M_tv': True,
                'enable_M_2smooth': False,
                'enable_T_tv': True,
                'enable_similarity': True,
            },
            
            # 6. 移除时间T_active的TV项
            'w/o_T_tv': {
                **base_params,
                'enable_A_l1': True,
                'enable_A_tv': True,
                'enable_M_tv': True,
                'enable_M_2smooth': True,
                'enable_T_tv': False,
                'enable_similarity': True,
            },
            
            # 7. 移除周期相似性约束
            'w/o_similarity': {
                **base_params,
                'enable_A_l1': True,
                'enable_A_tv': True,
                'enable_M_tv': True,
                'enable_M_2smooth': True,
                'enable_T_tv': True,
                'enable_similarity': False,
            },
            
            # 8. 仅异常分解（无周期约束）
            'only_anomaly': {
                **base_params,
                'enable_A_l1': True,
                'enable_A_tv': True,
                'enable_M_tv': False,
                'enable_M_2smooth': False,
                'enable_T_tv': False,
                'enable_similarity': False,
            },
            
            # 9. 仅周期分解（无异常约束）
            'only_periodic': {
                **base_params,
                'enable_A_l1': False,
                'enable_A_tv': False,
                'enable_M_tv': True,
                'enable_M_2smooth': True,
                'enable_T_tv': True,
                'enable_similarity': True,
            },
        }
        
        return configs
    
    def _process_single_signal(self, signal_idx, signal_file, config_name, params):
        """处理单个信号的方法（用于joblib并行）
        
        Args:
            signal_idx: 信号索引（用于GPU轮值）
            signal_file: 信号文件名
            config_name: 配置名称
            params: 参数字典
        
        Returns:
            dict: 包含信号结果的字典
        """
        # GPU轮值分配
        gpu_id = signal_idx % self.num_gpus
        device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
        
        # 加载信号
        X = np.load(os.path.join(self.signal_dir, signal_file))
        
        # 初始化优化器
        optimizer = JointOptimizer(X, params, device=device)
        optimizer.train(epochs=self.epochs)
        P, A = optimizer.get_components()
        
        # 计算指标
        recon_error = np.mean((X - P - A) ** 2)
        anomaly_sparsity = np.sum(np.abs(A) > 1e-3) / len(A)
        
        signal_base = signal_file[:-4]
        
        # 保存结果
        save_dir = os.path.join(self.output_dir, config_name)
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        
        np.save(os.path.join(save_dir, f'{signal_base}_P.npy'), P)
        np.save(os.path.join(save_dir, f'{signal_base}_A.npy'), A)
        
        # 返回结果（不包括loss_history以简化序列化）
        return {
            'signal_base': signal_base,
            'recon_error': float(recon_error),
            'anomaly_sparsity': float(anomaly_sparsity),
            'loss_history': optimizer.loss_history,
        }
    
    def run_experiment(self, config_name, params, gpu_id=0):
        """运行单个配置的实验（支持多信号并行处理，分别处理h和cpl两组数据）"""
        # 加载信号列表（分别获取h和cpl开头的文件）
        all_signal_files = sorted([f for f in os.listdir(self.signal_dir) if f.endswith('.npy')])
        h_signal_files = sorted([f for f in all_signal_files if f.startswith('h')])
        cpl_signal_files = sorted([f for f in all_signal_files if f.startswith('cpl')])
        
        # 整理结果
        config_results = {
            'config': config_name,
            'params': params,
            'h_signals': {},
            'cpl_signals': {}
        }
        
        # 处理h开头的信号
        if h_signal_files:
            print(f"\n[配置: {config_name}] 开始处理 {len(h_signal_files)} 个h信号...")
            print(f"使用 {self.num_gpus} 卡GPU进行轮值并行处理")
            print("="*60)
            
            h_results = Parallel(n_jobs=self.num_gpus, verbose=10)(
                delayed(self._process_single_signal)(
                    idx,
                    signal_file,
                    config_name,
                    params
                )
                for idx, signal_file in enumerate(h_signal_files)
            )
            
            print("="*60)
            print(f"[配置: {config_name}] h信号处理完成！")
            
            for result in h_results:
                signal_base = result['signal_base']
                config_results['h_signals'][signal_base] = {
                    'recon_error': result['recon_error'],
                    'anomaly_sparsity': result['anomaly_sparsity'],
                    'loss_history': result['loss_history'],
                }
        
        # 处理cpl开头的信号
        if cpl_signal_files:
            print(f"\n[配置: {config_name}] 开始处理 {len(cpl_signal_files)} 个cpl信号...")
            print(f"使用 {self.num_gpus} 卡GPU进行轮值并行处理")
            print("="*60)
            
            cpl_results = Parallel(n_jobs=self.num_gpus, verbose=10)(
                delayed(self._process_single_signal)(
                    idx,
                    signal_file,
                    config_name,
                    params
                )
                for idx, signal_file in enumerate(cpl_signal_files)
            )
            
            print("="*60)
            print(f"[配置: {config_name}] cpl信号处理完成！")
            
            for result in cpl_results:
                signal_base = result['signal_base']
                config_results['cpl_signals'][signal_base] = {
                    'recon_error': result['recon_error'],
                    'anomaly_sparsity': result['anomaly_sparsity'],
                    'loss_history': result['loss_history'],
                }
        
        self.results[config_name] = config_results
        return config_results
    
    def _run_single_config(self, config_idx, config_name, params):
        """运行单个配置的包装方法（用于joblib并行）"""
        return self.run_experiment(config_name, params)
    
    def run_all_experiments_parallel(self):
        """并行运行所有ablation配置（配置级别 + 信号级别双层并行）"""
        configs = self.get_ablation_configs()
        config_items = list(configs.items())
        
        print(f"Starting ablation experiments with {self.num_gpus} GPUs")
        print(f"Total configurations: {len(configs)}")
        print("="*60)
        
        # 使用joblib.Parallel进行并行处理（配置级别）
        print("开始并行处理配置...")
        results = Parallel(n_jobs=min(2, len(config_items)), verbose=10)(
            delayed(self._run_single_config)(
                idx,
                config_name,
                params
            )
            for idx, (config_name, params) in enumerate(config_items)
        )
        
        # 收集结果
        for idx, (config_name, params) in enumerate(config_items):
            self.results[config_name] = results[idx]
        
        print("="*60)
        print("所有配置的并行处理完成！")
    
    def generate_report(self):
        """生成实验报告（分别统计h和cpl两组数据）"""
        report = {
            'total_configs': len(self.results),
            'experiments': {},
            'summary': {
                'h_signals': {},
                'cpl_signals': {}
            }
        }
        
        # 遍历每个配置
        for config_name, result in self.results.items():
            report['experiments'][config_name] = result
            
            # 统计h信号信息
            if result['h_signals']:
                h_recon_errors = [v['recon_error'] for v in result['h_signals'].values()]
                h_sparsity = [v['anomaly_sparsity'] for v in result['h_signals'].values()]
                
                report['summary']['h_signals'][config_name] = {
                    'mean_recon_error': float(np.mean(h_recon_errors)),
                    'std_recon_error': float(np.std(h_recon_errors)),
                    'min_recon_error': float(np.min(h_recon_errors)),
                    'max_recon_error': float(np.max(h_recon_errors)),
                    'mean_anomaly_sparsity': float(np.mean(h_sparsity)),
                    'std_anomaly_sparsity': float(np.std(h_sparsity)),
                    'min_anomaly_sparsity': float(np.min(h_sparsity)),
                    'max_anomaly_sparsity': float(np.max(h_sparsity)),
                    'num_signals': len(result['h_signals'])
                }
            
            # 统计cpl信号信息
            if result['cpl_signals']:
                cpl_recon_errors = [v['recon_error'] for v in result['cpl_signals'].values()]
                cpl_sparsity = [v['anomaly_sparsity'] for v in result['cpl_signals'].values()]
                
                report['summary']['cpl_signals'][config_name] = {
                    'mean_recon_error': float(np.mean(cpl_recon_errors)),
                    'std_recon_error': float(np.std(cpl_recon_errors)),
                    'min_recon_error': float(np.min(cpl_recon_errors)),
                    'max_recon_error': float(np.max(cpl_recon_errors)),
                    'mean_anomaly_sparsity': float(np.mean(cpl_sparsity)),
                    'std_anomaly_sparsity': float(np.std(cpl_sparsity)),
                    'min_anomaly_sparsity': float(np.min(cpl_sparsity)),
                    'max_anomaly_sparsity': float(np.max(cpl_sparsity)),
                    'num_signals': len(result['cpl_signals'])
                }
        
        # 保存报告
        report_path = os.path.join(self.output_dir, 'ablation_report.json')
        with open(report_path, 'w') as f:
            # 转换numpy类型为可序列化的格式
            def convert_to_serializable(obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, (np.integer, np.floating)):
                    return obj.item()
                elif isinstance(obj, dict):
                    return {k: convert_to_serializable(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_to_serializable(v) for v in obj]
                return obj
            
            json.dump(convert_to_serializable(report), f, indent=2)
        
        print(f"\nReport saved to {report_path}")
        
        # 打印统计摘要
        print("\n" + "="*80)
        print("ABLATION STUDY SUMMARY")
        print("="*80)
        
        for config_name in self.results.keys():
            print(f"\n[配置: {config_name}]")
            
            if config_name in report['summary']['h_signals']:
                h_stats = report['summary']['h_signals'][config_name]
                print(f"  H信号 (n={h_stats['num_signals']}):")
                print(f"    重建误差: {h_stats['mean_recon_error']:.6f} ± {h_stats['std_recon_error']:.6f}")
                print(f"             (min: {h_stats['min_recon_error']:.6f}, max: {h_stats['max_recon_error']:.6f})")
                print(f"    异常稀疏度: {h_stats['mean_anomaly_sparsity']:.4f} ± {h_stats['std_anomaly_sparsity']:.4f}")
                print(f"               (min: {h_stats['min_anomaly_sparsity']:.4f}, max: {h_stats['max_anomaly_sparsity']:.4f})")
            
            if config_name in report['summary']['cpl_signals']:
                cpl_stats = report['summary']['cpl_signals'][config_name]
                print(f"  CPL信号 (n={cpl_stats['num_signals']}):")
                print(f"    重建误差: {cpl_stats['mean_recon_error']:.6f} ± {cpl_stats['std_recon_error']:.6f}")
                print(f"             (min: {cpl_stats['min_recon_error']:.6f}, max: {cpl_stats['max_recon_error']:.6f})")
                print(f"    异常稀疏度: {cpl_stats['mean_anomaly_sparsity']:.4f} ± {cpl_stats['std_anomaly_sparsity']:.4f}")
                print(f"               (min: {cpl_stats['min_anomaly_sparsity']:.4f}, max: {cpl_stats['max_anomaly_sparsity']:.4f})")
        
        print("\n" + "="*80)
        
        return report
    
    def plot_comparison(self):
        """保留结构但不绘制对比图（暂不需要）"""
        if not self.results:
            print("No results to plot")
            return
        
        print("\n图表生成已跳过（暂不需要）")
        print("详细结果已保存在 ablation_report.json 中")


def main():
    parser = argparse.ArgumentParser(description='Ablation Study for Signal Decomposition')
    parser.add_argument('--signal_dir', type=str, default='./synthetic_signal/',
                        help='Directory containing input signals')
    parser.add_argument('--output_dir', type=str, default='./ablation_results/',
                        help='Directory for saving results')
    parser.add_argument('--num_gpus', type=int, default=4,
                        help='Number of GPUs for parallel training')
    parser.add_argument('--epochs', type=int, default=2000,
                        help='Number of training epochs')
    parser.add_argument('--config', type=str, default=None,
                        help='Run single configuration (if None, run all)')
    
    args = parser.parse_args()
    
    # 检查GPU可用性
    num_available_gpus = torch.cuda.device_count()
    print(f"Available GPUs: {num_available_gpus}")
    print(f"Using: {min(args.num_gpus, num_available_gpus)} GPUs")
    
    # 初始化实验
    experiment = AblationExperiment(
        signal_dir=args.signal_dir,
        output_dir=args.output_dir,
        num_gpus=min(args.num_gpus, num_available_gpus),
        epochs=args.epochs
    )
    
    # 运行实验
    if args.config:
        # 单个配置
        configs = experiment.get_ablation_configs()
        if args.config in configs:
            print(f"Running single configuration: {args.config}")
            experiment.run_experiment(args.config, configs[args.config])
        else:
            print(f"Configuration '{args.config}' not found")
            print(f"Available: {list(configs.keys())}")
    else:
        # 并行运行所有配置
        experiment.run_all_experiments_parallel()
    
    # 生成报告和图表
    report = experiment.generate_report()
    experiment.plot_comparison()
    
    print("\nAblation study completed!")


if __name__ == "__main__":
    main()

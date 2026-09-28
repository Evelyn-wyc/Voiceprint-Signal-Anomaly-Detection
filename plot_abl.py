'''
Ablation Study Results Analysis
读取 ablation_results 中不同 config 下的 npy 文件
计算 AUROC, AUPR, Accuracy, Precision, Recall, F1 Score 等指标
分别统计 h 和 cpl 两组数据的结果
'''
import numpy as np
import pandas as pd
import os
import json
import matplotlib.pyplot as plt
from pathlib import Path
from sml3_anomaly import calculate_anomaly_metrics

class AblationResultsAnalyzer:
    """Ablation 实验结果分析类"""
    
    def __init__(self, ablation_results_dir, gt_dir, output_dir='./ablation_analysis/'):
        self.ablation_results_dir = ablation_results_dir
        self.gt_dir = gt_dir
        self.output_dir = output_dir
        self.results = {}
        
        # 创建输出目录
        Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    def get_configs(self):
        """获取所有配置目录"""
        configs = []
        if os.path.exists(self.ablation_results_dir):
            configs = sorted([d for d in os.listdir(self.ablation_results_dir) 
                            if os.path.isdir(os.path.join(self.ablation_results_dir, d))])
        return configs
    
    def analyze_single_config(self, config_name):
        """分析单个配置的所有信号"""
        config_dir = os.path.join(self.ablation_results_dir, config_name)
        
        # 获取该配置下的所有 _A.npy 文件
        signal_files = sorted([f for f in os.listdir(config_dir) if f.endswith('_A.npy')])
        
        h_results = []
        cpl_results = []
        
        for signal_file in signal_files:
            signal_base = signal_file[:-6]  # 去掉 _A.npy
            
            # 读取预测的异常分数
            A_path = os.path.join(config_dir, f'{signal_base}_A.npy')
            if not os.path.exists(A_path):
                continue
            
            A_pred = np.load(A_path)
            
            # 读取真实标签
            gt_path = os.path.join(self.gt_dir, f'{signal_base}_A.npy')
            if not os.path.exists(gt_path):
                print(f"警告: 找不到 {gt_path}")
                continue
            
            A_true = np.load(gt_path)
            
            try:
                # 计算异常检测指标
                metrics = calculate_anomaly_metrics(A_true, A_pred)
                
                # 额外计算SNR（基于原始信号和重构信号）
                snr = 0
                try:
                    signal_path = os.path.join(self.ablation_results_dir, '..', 'synthetic_signal', f'{signal_base}.npy')
                    p_path = os.path.join(self.ablation_results_dir, config_name, f'{signal_base}_P.npy')
                    
                    if os.path.exists(signal_path) and os.path.exists(p_path):
                        x_original = np.load(signal_path)
                        p_periodic = np.load(p_path)
                        
                        # 确保长度一致
                        min_len = min(len(x_original), len(p_periodic), len(A_pred))
                        x_original = x_original[:min_len]
                        p_periodic = p_periodic[:min_len]
                        a_pred = A_pred[:min_len]
                        
                        # 计算SNR：原始信号与重构信号的比值
                        x_recon = p_periodic + a_pred
                        error = x_original - x_recon
                        power_signal = np.sum(x_original ** 2)
                        power_error = np.sum(error ** 2)
                        
                        if power_signal > 0:
                            snr = 10 * np.log10(power_signal / (power_error + 1e-10))
                except Exception as snr_error:
                    snr = 0
                
                result = {
                    'signal_base': signal_base,
                    'Accuracy': metrics['Accuracy'],
                    'Precision': metrics['Precision'],
                    'Recall': metrics['Recall'],
                    'F1 Score': metrics['F1 Score'],
                    'AUROC': metrics['AUROC'],
                    'AUPR': metrics['AUPR'],
                    'SNR': snr,
                    'RMSE': metrics['RMSE'],
                    'MAE': metrics['MAE'],
                }
                
                # 根据信号名称前缀分类
                if signal_base.startswith('cpl'):
                    cpl_results.append(result)
                elif signal_base.startswith('h'):
                    h_results.append(result)
                    
            except Exception as e:
                print(f"错误: 处理 {signal_base} 失败: {e}")
        
        return {
            'config': config_name,
            'h_results': h_results,
            'cpl_results': cpl_results
        }
    
    def run_analysis(self):
        """运行完整的分析"""
        configs = self.get_configs()
        
        print(f"找到 {len(configs)} 个配置")
        print("="*100)
        
        for config_name in configs:
            print(f"\n正在分析配置: {config_name}")
            result = self.analyze_single_config(config_name)
            self.results[config_name] = result
        
        return self.results
    
    def generate_summary_report(self):
        """生成汇总报告"""
        summary_report = {}
        
        # 关键指标列表（包括方向标识：↑ 越高越好，↓ 越低越好）
        key_metrics = ['AUROC', 'AUPR', 'SNR', 'Accuracy', 'Precision', 'Recall', 'F1 Score', 'RMSE', 'MAE']
        
        for config_name, result in self.results.items():
            summary_report[config_name] = {
                'h_signals': {},
                'cpl_signals': {}
            }
            
            # 统计 h 信号
            if result['h_results']:
                h_df = pd.DataFrame(result['h_results'])
                h_stats = summary_report[config_name]['h_signals']
                h_stats['num_signals'] = len(result['h_results'])
                
                for metric in key_metrics:
                    h_stats[f'{metric}_mean'] = float(h_df[metric].mean())
                    h_stats[f'{metric}_std'] = float(h_df[metric].std())
                    h_stats[f'{metric}_min'] = float(h_df[metric].min())
                    h_stats[f'{metric}_max'] = float(h_df[metric].max())
            
            # 统计 cpl 信号
            if result['cpl_results']:
                cpl_df = pd.DataFrame(result['cpl_results'])
                cpl_stats = summary_report[config_name]['cpl_signals']
                cpl_stats['num_signals'] = len(result['cpl_results'])
                
                for metric in key_metrics:
                    cpl_stats[f'{metric}_mean'] = float(cpl_df[metric].mean())
                    cpl_stats[f'{metric}_std'] = float(cpl_df[metric].std())
                    cpl_stats[f'{metric}_min'] = float(cpl_df[metric].min())
                    cpl_stats[f'{metric}_max'] = float(cpl_df[metric].max())
        
        return summary_report
    
    def plot_decomposition(self, signal_base, config_name, save_dir):
        """绘制分解结果图（origin-period-anomaly-residual）
        
        Args:
            signal_base: 信号基础名称
            config_name: 配置名称
            save_dir: 保存目录
        """
        try:
            # 读取原始信号
            signal_path = os.path.join(self.ablation_results_dir, '..', 'synthetic_signal', f'{signal_base}.npy')
            if not os.path.exists(signal_path):
                return
            
            X = np.load(signal_path)
            time = np.linspace(0, 100, len(X))
            
            # 读取分解结果
            config_dir = os.path.join(self.ablation_results_dir, config_name)
            P_path = os.path.join(config_dir, f'{signal_base}_P.npy')
            A_path = os.path.join(config_dir, f'{signal_base}_A.npy')
            
            if not os.path.exists(P_path) or not os.path.exists(A_path):
                return
            
            P = np.load(P_path)
            A = np.load(A_path)
            
            # 创建图表
            plt.rcParams.update({
                "font.size": 26,
                "axes.titlesize": 26,
                "axes.labelsize": 22,
                "xtick.labelsize": 22,
                "ytick.labelsize": 22,
                "legend.fontsize": 22
            })
            
            fig = plt.figure(figsize=(14, 9), dpi=300)
            
            # 原始信号
            plt.subplot(4, 1, 1)
            plt.plot(time, X, label='Original Signal', linewidth=1.5)
            plt.title('Original Signal')
            plt.ylabel('Amplitude')
            plt.grid(True, alpha=0.3)
            
            # 周期分量
            plt.subplot(4, 1, 2)
            plt.plot(time, P, label='Periodic', color='green', linewidth=1.5)
            plt.title('Periodic')
            plt.ylabel('Amplitude')
            plt.grid(True, alpha=0.3)
            
            # 异常分量
            plt.subplot(4, 1, 3)
            plt.plot(time, A, label='Anomaly', color='red', linewidth=1.5)
            plt.title('Anomaly')
            plt.ylabel('Amplitude')
            plt.grid(True, alpha=0.3)
            
            # 残差
            plt.subplot(4, 1, 4)
            residual = X - P - A
            plt.plot(time, residual, label='Residual', color='orange', linewidth=1.5)
            plt.title('Residual')
            plt.ylabel('Amplitude')
            plt.xlabel('Time')
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.subplots_adjust(hspace=0.6)
            
            # 保存图表
            save_path = os.path.join(save_dir, f'{signal_base}_decomposition.png')
            plt.savefig(save_path, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            print(f"绘制 {signal_base} 的分解图失败: {e}")
    
    def save_results(self):
        """保存所有结果到文件和图表"""
        # 保存详细结果
        detailed_results = {}
        for config_name, result in self.results.items():
            detailed_results[config_name] = {
                'h_signals': result['h_results'],
                'cpl_signals': result['cpl_results']
            }
        
        # 保存为 JSON
        detailed_json_path = os.path.join(self.output_dir, 'detailed_results.json')
        with open(detailed_json_path, 'w') as f:
            json.dump(detailed_results, f, indent=2)
        print(f"\n详细结果已保存: {detailed_json_path}")
        
        # 生成汇总报告
        summary = self.generate_summary_report()
        summary_json_path = os.path.join(self.output_dir, 'summary_report.json')
        with open(summary_json_path, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"汇总报告已保存: {summary_json_path}")
        
        # 保存为 CSV (h 信号)
        self.save_csv_by_signal_type(summary, 'h_signals', 'h_signals_summary')
        
        # 保存为 CSV (cpl 信号)
        self.save_csv_by_signal_type(summary, 'cpl_signals', 'cpl_signals_summary')
        
        # 绘制每个配置中的信号分解图
        print("\n开始绘制分解图...")
        for config_name, result in self.results.items():
            # 创建配置对应的文件夹
            config_plot_dir = os.path.join(self.output_dir, config_name)
            Path(config_plot_dir).mkdir(parents=True, exist_ok=True)
            
            # 绘制 h 信号的分解图
            for h_result in result['h_results']:
                signal_base = h_result['signal_base']
                self.plot_decomposition(signal_base, config_name, config_plot_dir)
            
            # 绘制 cpl 信号的分解图
            for cpl_result in result['cpl_results']:
                signal_base = cpl_result['signal_base']
                self.plot_decomposition(signal_base, config_name, config_plot_dir)
            
            print(f"[{config_name}] 分解图绘制完成")
    
    def save_csv_by_signal_type(self, summary, signal_type, filename):
        """保存特定信号类型的 CSV 文件"""
        key_metrics = ['AUROC', 'AUPR', 'SNR', 'Accuracy', 'Precision', 'Recall', 'F1 Score', 'RMSE', 'MAE']
        
        rows = []
        for config_name, stats_dict in summary.items():
            if signal_type in stats_dict and stats_dict[signal_type]:
                stats = stats_dict[signal_type]
                row = {'Config': config_name, 'Num_Signals': stats.get('num_signals', 0)}
                
                for metric in key_metrics:
                    mean_key = f'{metric}_mean'
                    std_key = f'{metric}_std'
                    if mean_key in stats:
                        row[f'{metric}_Mean'] = stats[mean_key]
                        row[f'{metric}_Std'] = stats[std_key]
                
                rows.append(row)
        
        if rows:
            df = pd.DataFrame(rows)
            csv_path = os.path.join(self.output_dir, f'{filename}.csv')
            df.to_csv(csv_path, index=False)
            print(f"{signal_type} 结果已保存: {csv_path}")
    
    def print_summary(self):
        """打印汇总信息"""
        summary = self.generate_summary_report()
        
        # 指标方向说明：↑ 越高越好，↓ 越低越好
        metric_directions = {
            'AUROC': '↑',      # 越高越好
            'AUPR': '↑',       # 越高越好
            'SNR': '↑',        # 越高越好（Signal-to-Noise Ratio）
            'Accuracy': '↑',   # 越高越好
            'Precision': '↑',  # 越高越好
            'Recall': '↑',     # 越高越好
            'F1 Score': '↑',   # 越高越好
            'RMSE': '↓',       # 越低越好（Root Mean Square Error）
            'MAE': '↓'         # 越低越好（Mean Absolute Error）
        }
        
        key_metrics = ['AUROC', 'AUPR', 'SNR', 'Accuracy', 'Precision', 'Recall', 'F1 Score', 'RMSE', 'MAE']
        
        print("\n" + "="*140)
        print("ABLATION STUDY RESULTS SUMMARY".center(140))
        print("="*140)
        
        for config_name in sorted(summary.keys()):
            print(f"\n[配置: {config_name}]")
            
            # H 信号
            if summary[config_name]['h_signals']:
                h_stats = summary[config_name]['h_signals']
                print(f"  H 信号 (n={h_stats['num_signals']}):")
                for metric in key_metrics:
                    mean_key = f'{metric}_mean'
                    std_key = f'{metric}_std'
                    direction = metric_directions.get(metric, '')
                    if mean_key in h_stats:
                        print(f"    {metric:15s} {direction:1s}: {h_stats[mean_key]:8.4f} ± {h_stats[std_key]:.4f}")
            
            # CPL 信号
            if summary[config_name]['cpl_signals']:
                cpl_stats = summary[config_name]['cpl_signals']
                print(f"  CPL 信号 (n={cpl_stats['num_signals']}):")
                for metric in key_metrics:
                    mean_key = f'{metric}_mean'
                    std_key = f'{metric}_std'
                    direction = metric_directions.get(metric, '')
                    if mean_key in cpl_stats:
                        print(f"    {metric:15s} {direction:1s}: {cpl_stats[mean_key]:8.4f} ± {cpl_stats[std_key]:.4f}")
        
        print("\n" + "="*140)
        print("指标说明: ↑ 越高越好  |  ↓ 越低越好")
        print("="*140)


def main():
    # 配置路径
    ablation_results_dir = './ablation_results/'
    gt_dir = './synthetic_gtanomaly/'
    output_dir = './ablation_analysis/'
    
    # 检查目录是否存在
    if not os.path.exists(ablation_results_dir):
        print(f"错误: 找不到 {ablation_results_dir}")
        return
    
    if not os.path.exists(gt_dir):
        print(f"错误: 找不到 {gt_dir}")
        return
    
    # 创建分析器
    analyzer = AblationResultsAnalyzer(
        ablation_results_dir=ablation_results_dir,
        gt_dir=gt_dir,
        output_dir=output_dir
    )
    
    # 运行分析
    print("开始分析 Ablation 实验结果...")
    analyzer.run_analysis()
    
    # 保存结果
    analyzer.save_results()
    
    # 打印汇总
    analyzer.print_summary()
    
    print(f"\n所有结果已保存到: {output_dir}")


if __name__ == "__main__":
    main()

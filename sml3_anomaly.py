'''
Anomaly detection module: compared gt and anomaly.
Only use time.
anomaly_npy > 0 means anomaly. 
'''

import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score

def calculate_reconstruction_metrics(x_original, p_periodic, a_anomaly):
    """
    计算重构质量指标
    
    Args:
        x_original: 原始信号
        p_periodic: 周期分量
        a_anomaly: 异常分量
    
    Returns:
        dict: 包含SNR, RMSE, MAE的字典
    """
    # 确保长度一致
    min_len = min(len(x_original), len(p_periodic), len(a_anomaly))
    x_original = x_original[:min_len]
    p_periodic = p_periodic[:min_len]
    a_anomaly = a_anomaly[:min_len]
    
    # 重构信号
    x_recon = p_periodic + a_anomaly
    
    # 重构误差
    error = x_original - x_recon
    
    # -- SNR: 原始信号和重构信号之间的信噪比 --
    power_signal = np.sum(x_original ** 2)
    power_error = np.sum(error ** 2)
    if power_signal == 0:
        snr = 0  # 如果原始信号全0，设为0而不是inf
    else:
        snr = 10 * np.log10(power_signal / (power_error + 1e-10))
    
    # -- RMSE --
    rmse = np.sqrt(np.mean(error ** 2))
    
    # -- MAE --
    mae = np.mean(np.abs(error))
    
    return {'SNR': snr, 'RMSE': rmse, 'MAE': mae}

def calculate_anomaly_metrics(gt, anomaly_score):
    """
    计算异常检测的所有指标
    
    Args:
        gt: 真实异常标签 (连续值)
        anomaly_score: 预测的异常分数 (连续值)
    
    Returns:
        dict: 包含所有指标的字典
    """
    # 确保长度一致
    min_length = min(len(gt), len(anomaly_score))
    gt = gt[:min_length]
    anomaly_score = anomaly_score[:min_length]
    
    # 转换为二进制标签
    labels = (gt > 0).astype(int)
    
    # 计算TP, TN, FP, FN (使用阈值0)
    threshold = 0
    TP = np.sum((gt > 0) & (anomaly_score > threshold))
    TN = np.sum((gt == 0) & (anomaly_score <= threshold))
    FP = np.sum((gt == 0) & (anomaly_score > threshold))
    FN = np.sum((gt > 0) & (anomaly_score <= threshold))
    
    # 计算准确率、精确率、召回率、F1分数
    accuracy = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1_score_val = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    # 计算ROC和PR指标
    try:
        fpr, tpr, _ = roc_curve(labels, anomaly_score)
        roc_auc = auc(fpr, tpr)
    except:
        roc_auc = 0
    
    try:
        prec, rec, _ = precision_recall_curve(labels, anomaly_score)
        avg_prec = average_precision_score(labels, anomaly_score)
    except:
        avg_prec = 0
    
    # 计算预测误差指标（gt vs anomaly_score）
    error = gt - anomaly_score
    
    # SNR: 信号功率 vs 误差功率
    power_signal = np.sum(gt ** 2)
    power_error = np.sum(error ** 2)
    if power_signal == 0:
        snr = 0
    else:
        snr = 10 * np.log10(power_signal / (power_error + 1e-10))
    
    # RMSE
    rmse = np.sqrt(np.mean(error ** 2))
    
    # MAE
    mae = np.mean(np.abs(error))
    
    return {
        'TP': TP,
        'TN': TN,
        'FP': FP,
        'FN': FN,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1_score_val,
        'AUROC': roc_auc,
        'AUPR': avg_prec,
        'SNR': snr,
        'RMSE': rmse,
        'MAE': mae
    }


def process_signal_group(prefix, gtdir, anomalydir, savedir, gtfiles, anomalyfiles):
    """处理特定前缀的信号组（h或cpl）
    
    Args:
        prefix: 文件前缀 ('h' 或 'cpl')
        gtdir: ground truth目录
        anomalydir: 异常检测结果目录
        savedir: 保存目录
        gtfiles: ground truth文件列表
        anomalyfiles: 异常检测文件列表
    
    Returns:
        pd.DataFrame: 结果数据框
    """
    print(f"\n{'='*60}")
    print(f"Processing {prefix.upper()} signals...")
    print(f"{'='*60}")
    
    result = []
    processed_count = 0
    
    for gtfile in gtfiles:
        if gtfile.endswith('.npy') and gtfile.startswith(prefix):
            anomalyfile = gtfile
            if anomalyfile in anomalyfiles:
                gt = np.load(os.path.join(gtdir, gtfile))
                anomaly = np.load(os.path.join(anomalydir, anomalyfile))
                
                # 使用计算函数
                metrics = calculate_anomaly_metrics(gt, anomaly)
                
                # 绘制ROC曲线和Precision-Recall曲线
                labels = (gt > 0).astype(int)
                fig, ax = plt.subplots(1, 2, figsize=(12,6))

                # ROC
                fpr, tpr, thresholds = roc_curve(labels, anomaly)
                roc_auc = metrics['AUROC']
                ax[0].plot(fpr, tpr, label=f'AUC = {roc_auc:.3f}')
                ax[0].plot([0, 1], [0, 1], 'k--')
                ax[0].set_xlabel('False Positive Rate')
                ax[0].set_ylabel('True Positive Rate')
                ax[0].set_title(f'ROC Curve - {gtfile[:-4]}')
                ax[0].legend(loc = 'lower right')

                # PR
                prec, rec, thresholds_pr = precision_recall_curve(labels, anomaly)
                avg_prec = metrics['AUPR']
                ax[1].plot(rec, prec, label=f'AP = {avg_prec:.3f}')
                ax[1].set_xlabel('Recall')
                ax[1].set_ylabel('Precision')
                ax[1].set_title(f'Precision-Recall Curve - {gtfile[:-4]}')
                ax[1].legend(loc = 'lower left')

                plt.tight_layout()
                plt.savefig(os.path.join(savedir, f'{gtfile[:-4]}_roc_pr.png'))
                plt.close()

                # 保存结果
                result_path = os.path.join(savedir, f'{gtfile[:-6]}_result.txt')
                with open(result_path, 'w') as f:
                    f.write(f'TP: {metrics["TP"]}\n')
                    f.write(f'TN: {metrics["TN"]}\n')
                    f.write(f'FP: {metrics["FP"]}\n')
                    f.write(f'FN: {metrics["FN"]}\n')
                    f.write(f'Accuracy: {metrics["Accuracy"]:.4f}\n')
                    f.write(f'Precision: {metrics["Precision"]:.4f}\n')
                    f.write(f'Recall: {metrics["Recall"]:.4f}\n')
                    f.write(f'F1 Score: {metrics["F1 Score"]:.4f}\n')
                    f.write(f'AUROC: {metrics["AUROC"]:.3f}\n')
                    f.write(f'AUPR: {metrics["AUPR"]:.3f}\n')
                    f.write(f'SNR: {metrics["SNR"]:.3f} dB\n')
                    f.write(f'RMSE: {metrics["RMSE"]:.6f}\n')
                    f.write(f'MAE: {metrics["MAE"]:.6f}\n')
                
                result.append([gtfile[:-6], metrics["TP"], metrics["TN"], metrics["FP"], metrics["FN"], 
                             metrics["Accuracy"], metrics["Precision"], metrics["Recall"], metrics["F1 Score"], 
                             metrics["AUROC"], metrics["AUPR"], metrics["SNR"], metrics["RMSE"], metrics["MAE"]])
                processed_count += 1
    
    print(f"Processed {processed_count} {prefix.upper()} files")
    
    # 创建结果DataFrame
    result_df = pd.DataFrame(result, columns=['Filename', 'TP', 'TN', 'FP', 'FN', 'Accuracy', 
                                              'Precision', 'Recall', 'F1 Score', 'AUROC', 'AUPR', 
                                              'SNR', 'RMSE', 'MAE'])
    
    # 保存该组的结果
    result_df.to_csv(os.path.join(savedir, f'synthetic_{prefix}_anomaly_results.csv'), index=False)
    
    # 计算统计摘要
    metrics_list = ['AUROC', 'AUPR', 'SNR', 'RMSE', 'MAE', 'Accuracy', 'Precision', 'Recall', 'F1 Score']
    clean_df = result_df.dropna(subset=metrics_list)
    
    for m in metrics_list:
        clean_df = clean_df[clean_df[m] != 0]
    
    if len(clean_df) > 0:
        summary_data = {}
        for m in metrics_list:
            mean_val = clean_df[m].mean()
            std_val = clean_df[m].std()
            summary_data[m] = [mean_val, std_val]
        
        summary_df = pd.DataFrame(summary_data, index=['Mean', 'Std'])
        summary_df.to_csv(os.path.join(savedir, f'metrics_summary_{prefix}.csv'))
        
        print(f"\n{prefix.upper()} Summary Statistics:")
        print(summary_df.round(4))
    
    return result_df


if __name__ == "__main__":
    
    gtdir = './synthetic_gtanomaly/'
    anomalydir = './synthetic_signal_npy/'
    savedir = './synthetic_result/'
    if not os.path.exists(savedir):
        os.makedirs(savedir)

    gtfiles = os.listdir(gtdir)
    gtfiles.sort()
    anomalyfiles = os.listdir(anomalydir)
    anomalyfiles.sort()
    
    print(f"Total GT files: {len(gtfiles)}")
    print(f"Total anomaly files: {len(anomalyfiles)}")

    # 自动处理h和cpl两组数据
    signal_groups = ['h', 'cpl']
    all_results = {}
    
    for prefix in signal_groups:
        result_df = process_signal_group(prefix, gtdir, anomalydir, savedir, gtfiles, anomalyfiles)
        all_results[prefix] = result_df
    
    print(f"\n{'='*60}")
    print("All processing complete!")
    print(f"{'='*60}")
    print(f"Results saved to: {savedir}")
    print(f"  - synthetic_h_anomaly_results.csv")
    print(f"  - synthetic_cpl_anomaly_results.csv")
    print(f"  - metrics_summary_h.csv")
    print(f"  - metrics_summary_cpl.csv")

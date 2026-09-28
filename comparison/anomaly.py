'''
1, Choose the dataset (case or simulation). Simulation has two branches: non-cpl and cpl. Filename changes.
2, Choose the comparison method (1, 2, 3)
3, Define the file path: 
    npy folder saves the learned anomaly and period npy files;
    result folder saves the metrics (txt, figure, csv in total, summary statistics)
'''

import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score


# arr = np.load(os.path.join('./simulation/method1/npy/', 'cpl_h1_s1_A.npy'))
# pd.DataFrame(arr).to_csv(os.path.join('./simulation/method1/result/', 'sample_anomaly.csv'), index=False, header=False)

def calculate_reconstruction_metrics(a_true, a_est):

    # 确保长度一致
    min_len = min(len(a_true), len(a_est))
    a_true = a_true[:min_len]
    a_est = a_est[:min_len]
    
    # 计算误差
    error = a_true - a_est
    
    # -- SNR --
    power_signal = np.sum(a_true ** 2)
    power_error = np.sum(error ** 2)
    if power_signal == 0:
        snr = float('inf')
    else:
        snr = 10 * np.log10(power_signal / (power_error + 1e-4))
    
    # -- RMSE --
    rmse = np.sqrt(np.mean(error ** 2))
    
    # -- MAE --
    mae = np.mean(np.abs(error))
    
    return {'SNR': snr, 'RMSE': rmse, 'MAE': mae}

def anomaly_evaluation(dataset, method):
    """
    dataset: 'case' or 'simulation'
    method: comparison method
    """

    if dataset == 'case':
        gtdir = '../241230_gt_xwidth/'
    else:
        gtdir = '../synthetic_gtanomaly/'
    gtfiles = [f for f in os.listdir(gtdir) if f.endswith('.npy')]
    gtfiles.sort()

    if method == 'method1':
        method_name = 'STL'
    elif method == 'method2':
        method_name = 'VMD'
    elif method == 'method3':
        method_name = 'QPGP'

    savedir_npy = f'./{dataset}/{method}/npy/'
    savedir_rst = f'./{dataset}/{method}/result/'

    print(f"Processing dataset '{dataset}', method {method_name}")
        
    anomalyfiles = [f for f in os.listdir(savedir_npy) if f.endswith('_A.npy')]
    anomalyfiles.sort()

    flag = 0  # simulation: 0 for non-cpl, 1 for cpl; case: always 0
    result = []
    for gtfile in gtfiles:

        if dataset == 'case':
            base_filename = gtfile[:-11]  # 去掉_xwidth.npy
        else:
            base_filename = gtfile[:-6]  # 去掉_A.npy
        anomalyfile = f'{base_filename}_A.npy'  # 约定命名
        
        if anomalyfile not in anomalyfiles:
            print(f"Anomaly file {anomalyfile} not found for GT {gtfile} in method {method}, skip")
            continue  


        # Remember to change this!
        # if gtfile.startswith('cpl'):
        #     flag = 1  # cpl anomaly detection
        if gtfile.startswith('h') or gtfile.startswith('2'):
            gt_path = os.path.join(gtdir, gtfile)
            anomaly_path = os.path.join(savedir_npy, anomalyfile)

            gt = np.load(gt_path)
            anomaly = np.load(anomaly_path)

            min_length = min(len(gt), len(anomaly))
            gt = gt[:min_length]
            anomaly = anomaly[:min_length]

            threshold = 0

            TP = np.sum((gt > 0) & (anomaly != threshold))
            TN = np.sum((gt == 0) & (anomaly == threshold))
            FP = np.sum((gt == 0) & (anomaly != threshold))
            FN = np.sum((gt > 0) & (anomaly == threshold))

            accuracy = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0
            precision = TP / (TP + FP) if (TP + FP) > 0 else 0 # 误报率
            recall = TP / (TP + FN) if (TP + FN) > 0 else 0 # 漏报率
            f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

            labels = (gt > 0).astype(int)

            fpr, tpr, _ = roc_curve(labels, anomaly)
            roc_auc = auc(fpr, tpr)
            avg_prec = average_precision_score(labels, anomaly)

            current_result_row = [base_filename, method, TP, TN, FP, FN, accuracy, precision, recall, f1_score, roc_auc, avg_prec]

            if dataset == 'simulation':
                recon_metrics = calculate_reconstruction_metrics(gt, anomaly)
                current_result_row.extend([recon_metrics["SNR"], recon_metrics["RMSE"], recon_metrics["MAE"]])
            
            result.append(current_result_row)

    # 保存csv结果
    if dataset == 'simulation':
        columns = ['Filename', 'Method', 'TP', 'TN', 'FP', 'FN', 'Accuracy', 'Precision', 'Recall', 'F1 Score', 'AUROC', 'AUPR', 'SNR', 'RMSE', 'MAE']
    else:
        columns = ['Filename', 'Method', 'TP', 'TN', 'FP', 'FN', 'Accuracy', 'Precision', 'Recall', 'F1 Score', 'AUROC', 'AUPR']

    result_df = pd.DataFrame(result, columns=columns)
    if flag == 1:
        csv_path = os.path.join(savedir_rst, f'cpl_anomaly_detection_results_{method_name}.csv')
    else:
        csv_path = os.path.join(savedir_rst, f'anomaly_detection_results_{method_name}.csv')
    result_df.to_csv(csv_path, index=False)

    # 计算统计信息
    if dataset == 'simulation':
        metrics = ['AUROC', 'AUPR', 'SNR', 'RMSE', 'MAE', 'Accuracy', 'Precision', 'Recall', 'F1 Score']
    else:
        metrics = ['AUROC', 'AUPR', 'Accuracy', 'Precision', 'Recall', 'F1 Score']

    if 'SNR' in result_df.columns:
        result_df = result_df.replace([np.inf, -np.inf], np.nan)

    clean_df = result_df.dropna(subset=metrics)
    for m in metrics:
        clean_df = clean_df[clean_df[m] != 0]

    summary_data = {m: [clean_df[m].mean(), clean_df[m].std()] for m in metrics}
    summary_df = pd.DataFrame(summary_data, index=['Mean', 'Std'])
    if flag == 1:
        summary_path = os.path.join(savedir_rst, f'cpl_metrics_summary_{method_name}.csv')
    else:
        summary_path = os.path.join(savedir_rst, f'metrics_summary_{method_name}.csv')
    summary_df.to_csv(summary_path)

    return summary_df


if __name__ == '__main__':
    # 用户输入示例，方便切换
    datasets = ['case', 'simulation']
    methods = ['method1', 'method2', 'method3'] # STL, VMD, QPGP

    # results = anomaly_evaluation(datasets[1], methods[0])
    # results = anomaly_evaluation(datasets[1], methods[1])
    results = anomaly_evaluation(datasets[0], methods[2])




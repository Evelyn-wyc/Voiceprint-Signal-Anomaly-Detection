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

gtdir = './241230_gt_xwidth/'
anomalydir = './241230_decomposed_npy/'
savedir = './241230_result/'
if not os.path.exists(savedir):
    os.makedirs(savedir)

gtfiles = os.listdir(gtdir)
gtfiles.sort()
anomalyfiles = os.listdir(anomalydir)
anomalyfiles.sort()

# 将所有结果保存到一个csv文件中
result = []

# arr = np.load(os.path.join(anomalydir, '2024-11-19_12_08_46_673C0F4E[0258]_1_A.npy'))
# pd.DataFrame(arr).to_csv(os.path.join(savedir, 'sample_anomaly.csv'), index=False, header=False)

for gtfile in gtfiles:
    if gtfile.endswith('.npy'):
        base_filename = gtfile[:-11]  # 去掉_xwidth.npy后缀
        anomalyfile = f'{base_filename}_A.npy'
        if anomalyfile in anomalyfiles:
            gt = np.load(os.path.join(gtdir, gtfile))
            anomaly = np.load(os.path.join(anomalydir, anomalyfile))
            # print(f"Processing {base_filename}: GT shape {gt.shape}, Anomaly shape {anomaly.shape}")                
            
            # 确保gt和anomaly长度一致
            min_length = min(len(gt), len(anomaly))
            gt = gt[:min_length]
            anomaly = anomaly[:min_length]

            threshold = 0

            # 计算TP, TN, FP, FN
            TP = np.sum((gt == 1) & (anomaly > threshold))
            TN = np.sum((gt == 0) & (anomaly <= threshold))
            FP = np.sum((gt == 0) & (anomaly > threshold))
            FN = np.sum((gt == 1) & (anomaly <= threshold))

            # 计算准确率、精确率、召回率、F1分数
            accuracy = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0
            precision = TP / (TP + FP) if (TP + FP) > 0 else 0
            recall = TP / (TP + FN) if (TP + FN) > 0 else 0
            f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

            # 绘制ROC曲线和Precision-Recall曲线
            labels = (gt > 0).astype(int)
            fig, ax = plt.subplots(1, 2, figsize=(12,6))

            # ROC
            fpr, tpr, thresholds = roc_curve(labels, anomaly)
            roc_auc = auc(fpr, tpr)
            ax[0].plot(fpr, tpr, label=f'AUC = {roc_auc:.3f}')
            ax[0].plot([0, 1], [0, 1], 'k--')
            ax[0].set_xlabel('False Positive Rate')
            ax[0].set_ylabel('True Positive Rate')
            ax[0].set_title(f'ROC Curve - {gtfile[:-11]}')
            ax[0].legend(loc = 'lower right')

            # PR
            prec, rec, thresholds_pr = precision_recall_curve(labels, anomaly)
            avg_prec = average_precision_score(labels, anomaly)
            ax[1].plot(rec, prec, label=f'AP = {avg_prec:.3f}')
            ax[1].set_xlabel('Recall')
            ax[1].set_ylabel('Precision')
            ax[1].set_title(f'Precision-Recall Curve - {gtfile[:-11]}')
            ax[1].legend(loc = 'lower left')

            plt.tight_layout()
            plt.savefig(os.path.join(savedir, f'{gtfile[:-11]}_roc_pr.png'))
            plt.close()

            # 保存结果
            result_path = os.path.join(savedir, f'{base_filename}_result.txt')
            with open(result_path, 'w') as f:
                f.write(f'TP: {TP}\n')
                f.write(f'TN: {TN}\n')
                f.write(f'FP: {FP}\n')
                f.write(f'FN: {FN}\n')
                f.write(f'Accuracy: {accuracy:.4f}\n')
                f.write(f'Precision: {precision:.4f}\n')
                f.write(f'Recall: {recall:.4f}\n')
                f.write(f'F1 Score: {f1_score:.4f}\n')
                f.write(f'AUROC: {roc_auc:.3f}\n')
                f.write(f'AUPR: {avg_prec:.3f}\n')
            
            result.append([base_filename, TP, TN, FP, FN, accuracy, precision, recall, f1_score, roc_auc, avg_prec])
result_df = pd.DataFrame(result, columns=['Filename', 'TP', 'TN', 'FP', 'FN', 'Accuracy', 'Precision', 'Recall', 'F1 Score', 'AUROC', 'AUPR'])
result_df.to_csv(os.path.join(savedir, 'anomaly_detection_results.csv'), index=False)

# 去掉nan和0之后计算每个指标的均值和标准差并保存
metrics = ['AUROC', 'AUPR', 'Accuracy', 'Precision', 'Recall', 'F1 Score']
clean_df = result_df.dropna(subset=metrics)

for m in metrics:
    clean_df = clean_df[clean_df[m] != 0]

summary_data = {}
for m in metrics:
    mean_val = clean_df[m].mean()
    std_val = clean_df[m].std()
    summary_data[m] = [mean_val, std_val]

summary_df = pd.DataFrame(summary_data, index=['Mean', 'Std'])
summary_df.to_csv(os.path.join(savedir, 'metrics_summary.csv'))
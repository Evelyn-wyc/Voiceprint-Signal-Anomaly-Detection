'''
Processing ground truth xwidth data from 241230_gt
'''
import numpy as np
import pandas as pd
import os

filedir = './241230_gt/'
filenames = os.listdir(filedir)
filenames.sort()  # 确保文件名排序
savedir = './241230_gt_xwidth/'
if not os.path.exists(savedir):
    os.makedirs(savedir)

for file in filenames:
    if file.endswith('.csv'):
        data = pd.read_csv(os.path.join(filedir, file))
        data.columns = data.columns.str.strip()  # 去除列名空格

        if data.empty:
            df_out = pd.DataFrame(columns=["start", "end"])
            df_out.to_csv(os.path.join(savedir, file.replace('.csv', '_xwidth.csv')), index=False)
        else:
            df_out = pd.DataFrame()
            df_out["start"] = data["X"]
            df_out["end"] = data["X"] + data["Width"]
            df_out.to_csv(os.path.join(savedir, file.replace('.csv', '_xwidth.csv')), index=False)

        # 存成3600长度的npy便于比较异常
        arr = np.zeros(3600, dtype=int)
        for _, row in df_out.iterrows():
            start, end = int(row['start']), int(row['end'])
            if start < 3600 and end <= 3600:
                arr[start:end] = 1
        
        new_filename = file.replace('Overlay Elements of ', '')
        np.save(os.path.join(savedir, new_filename.replace('.csv', '_xwidth.npy')), arr)
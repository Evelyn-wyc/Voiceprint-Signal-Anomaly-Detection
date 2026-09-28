'''
csv file to npy
'''
import os
import numpy as np

def csv_to_npy(csv_file, npy_file):
    data = np.loadtxt(csv_file, delimiter=',', skiprows=1, usecols=40)

    np.save(npy_file, data)

csv_to_npy('./510009.csv', '.510009.npy')
npy_file = './510009.npy'
print(npy_file.shape())
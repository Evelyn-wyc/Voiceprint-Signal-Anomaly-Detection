'''
1. audio data cut into t seconds slices,
2. wave to image (waveform)
'''
import librosa
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import os
filedir = './241230/'
savedir = './241230_wave/'

def figure(y, sr, filename, start_time):

    # 绘制Time Domain波形图
    time = np.linspace(0, len(y)/sr, len(y))
    plt.figure(dpi=300, figsize=(12,6))

    plt.plot(time + start_time, y)
    plt.xlabel('Time/s',fontsize=20)
    plt.ylabel('Amplitude',fontsize=20)
    plt.title("Time Domain Waveform")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)

def slice_audio(filename, t = 15):

    # 读取音频文件
    y, sr = librosa.load(f'{filedir}{filename}', sr=None)

    # 将音频数据切片为t秒
    for i in range(0, len(y), t * sr):
        if i + t * sr <= len(y):
            y_slice = y[i: i + t * sr]
        else:
            break
        start_time = i / sr

        # 绘制Time Domain波形图
        print(f'{filename[:-4]}_{i // (t * sr) + 1}.png')
        figure(y_slice, sr, f'{savedir}{filename[:-4]}_{i // (t * sr) + 1}.png', start_time = start_time)

path = os.listdir(filedir)
path.sort()   
for file in path:

    # 如果后缀是aac格式的文件，转化为wav
    if file.endswith('.aac'):
        os.system(f'ffmpeg -i ./241230/{file} ./241230/{file[:-4]}.wav')
        os.system(f'rm ./241230/{file}')
        file = file[:-4] + '.wav'
    
    # 读取音频文件
    slice_audio(file, t = 15)



    
    # # 绘制Time-Frequency Domain语谱图 —— Mel Spectrogram
    #     S = librosa.feature.melspectrogram(y=y_slice, sr=sr, n_mels=128)
    #     log_S = librosa.power_to_db(S, ref=np.max)
    #     plt.figure(dpi=300, figsize=(12, 6))
    #     librosa.display.specshow(log_S, sr=sr, x_axis='time', y_axis='mel', cmap='YlGnBu')
    #     plt.gca().set_axis_off()
    #     plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    #     # plt.colorbar(format='%+2.0f dB')
    #     # plt.title('Mel Spectrogram')
    #     plt.savefig(f'./241230_image/{file[:-4]}_{i//(15*sr)+1}.png', dpi=300)
    #     plt.close('all')
    
    # # 绘制Frequency Domain 幅度谱
    # fft_result = np.fft.fft(y)
    # magnitude = np.abs(fft_result)
    # frequency = np.fft.fftfreq(len(magnitude), 1/sr)
    # # 截取正频率数据
    # positive_frequency = frequency[:len(frequency)//2]
    # positive_magnitude = magnitude[:len(magnitude)//2]
    # # 截取低频数据
    # mask = positive_frequency <= 4096
    # positive_frequency = positive_frequency[mask]
    # positive_magnitude = positive_magnitude[mask]

    # plt.figure(dpi=200,figsize=(10,4))
    # plt.plot(positive_frequency, positive_magnitude, linewidth=0.5)
    # plt.xlabel('Frequency/Hz',fontsize=12)
    # plt.ylabel('Amplitude',fontsize=12)
    # plt.title('Amplitude Spectrum')
    # plt.grid(True)
    # plt.savefig(f'./241230_image/{file}_amplitude.png', dpi=300)  # 保存为png文件

    # # 绘制PSD功率谱密度图
    # freq, power = welch(y, fs=sr, nperseg=1024)
    # power_db = librosa.power_to_db(power, ref=np.max)
    # mask = freq <= 4096
    # freq = freq[mask]
    # power_db = power_db[mask]
    # plt.figure(dpi=200,figsize=(10,4))
    # plt.plot(freq, power_db)
    # plt.xlabel('Frequency/Hz',fontsize=12)
    # plt.ylabel('Power',fontsize=12)
    # plt.title("Power Spectral Density (PSD)")
    # plt.xlim([0, 4096])
    # plt.grid(True)
    # plt.savefig(f'./241230_image/{file}_psd.png', dpi=300)  # 保存为png文件
    
    # # 绘制Time-Frequency Domain语谱图 —— Mel Spectrogram
    # S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    # log_S = librosa.power_to_db(S, ref=np.max)
    # plt.figure(dpi=200,figsize=(10,4))
    # librosa.display.specshow(log_S, sr=sr, x_axis='time', y_axis='mel', cmap='YlGnBu')
    # plt.colorbar(format='%+2.0f dB')
    # plt.title('Mel Spectrogram')
    # plt.savefig(f'./241230_image/{file}_melspectrogram.png', dpi=300)  # 保存为png文件
    # plt.close('all')

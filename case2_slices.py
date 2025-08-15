'''
1. audio data cut into t seconds slices,
2. save the spectrogram of each slice as an image
'''
import librosa
import soundfile as sf
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import os
filedir = './241230/'
savedir = './241230_slice/'
savedir2 = './241230_audslice/'

if not os.path.exists(savedir):
    os.makedirs(savedir)

def figure(y, sr, file, note, start_time):
    fmax = sr/2
    # 绘制Time-Frequency Domain语谱图 —— Mel Spectrogram
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=fmax)

    log_S = librosa.power_to_db(S, ref=np.max)
    plt.figure(dpi=300, figsize=(12, 6))
    # 计算每个时间帧对应的绝对时间
    times = librosa.times_like(S, sr=sr)
    x_coords = start_time + times
    librosa.display.specshow(log_S, sr=sr, x_coords = x_coords, x_axis='time', y_axis='mel', cmap='YlGnBu', fmax=fmax)
    # note 表示是否显示坐标轴等信息
    if note == True:
        plt.colorbar(format='%+2.0f dB')
        plt.title('Mel Spectrogram')
        ax = plt.gca()
        ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        ax.xaxis.set_major_locator(plt.MaxNLocator(10))  # 控制刻度密度
    else:
        plt.gca().set_axis_off()
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.savefig(file, dpi=300)
    plt.close('all')


def slice_audio(filename, note, t = 15):
    print(filename)
    # 读取音频文件
    y, sr = librosa.load(f'{filedir}{filename}', sr=None)
    duration = librosa.get_duration(y=y, sr=sr)
    time = np.linspace(0, duration, len(y))

    # 将音频数据切片为t秒
    for i in range(0, len(y), t * sr):
        if i + t * sr <= len(y):
            y_slice = y[i: i + t * sr]
        else:
            break
        start_time = i / sr

        # 绘制Time-Frequency Domain语谱图 —— Mel Spectrogram
        figure(y_slice, sr, f'{savedir}{filename[:-4]}_{i // (t * sr) + 1}.png', note, start_time = start_time)

        # 保存切片音频
        if not os.path.exists(savedir2):
            os.makedirs(savedir2)
        sf.write(f'{savedir2}{filename[:-4]}_{i // (t * sr) + 1}.wav', y_slice, sr)

# slice_audio('2024-12-07-14-14-32_192.168.182.64.wav', note = True, t = 5) # test
for file in os.listdir(filedir):
    slice_audio(file, note = False, t = 15)
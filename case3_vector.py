'''
1. project slices into vector(s) (1D space): mean? sum? square sum? remember to normalize!
'''
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt

filedir = './241230_slice/'
# filedir = './241230_cpr/'
# filedir = './241230_gabor/'
savedir = './241230_vector/'
savedir_npy = './241230_vector_npy/'
savedir_multi = './241230_mulvec4/'

if not os.path.exists(savedir):
    os.makedirs(savedir)
if not os.path.exists(savedir_multi):
    os.makedirs(savedir_multi)
if not os.path.exists(savedir_npy):
    os.makedirs(savedir_npy)

def project_vector(filename, method = 'square_sum', vecmode = 1):
    img = cv2.imread(f'{filedir}{filename}', cv2.IMREAD_GRAYSCALE)
    if filedir != './241230_gabor/':
        img = 1 - img / 255.
    else:
        img = img / 255.
    
    if vecmode == 1:
        if method == 'mean':
            img = np.mean(img, axis = 0)
        elif method == 'sum':
            img = np.sum(img, axis = 0)
        elif method == 'square_sum':
            img = np.sum(img ** 2, axis = 0)
            img = np.sqrt(img)  # 开平方根
            img_copy = img.copy()
            # 减掉第100小的值，归一化到[0, max]
            smallest = np.partition(img_copy.flatten(), 100)[99]
            print("100 smallest value: ", smallest)
            img = np.maximum(img - smallest, 0)
        else:
            raise ValueError('method should be mean, sum or square_sum')
        print("max: ", img.max())
        return img
    else:
        # 将img按照列均分成vecmode份
        img_total = np.array_split(img, vecmode, axis = 0)
        img_total_copy = img_total.copy()
        for num in range(vecmode):
            # # 确定img_total[num]是哪一段频率对应的图像，使用colormap
            # plt.figure(figsize=(12, 6 / vecmode))
            # plt.imshow(img_total_copy[num], cmap = 'YlGnBu')
            # plt.savefig(f'{savedir_multi}{filename[:-4]}_{num}_ori.png', dpi=300)
            # plt.close()
            if method == 'mean':
                img_total[num] = np.mean(img_total[num], axis = 0)
            elif method == 'sum':
                img_total[num] = np.sum(img_total[num], axis = 0)
            elif method == 'square_sum':
                img_total[num] = np.sum(img_total[num] ** 2, axis = 0)
                # # 减掉第100小的值，归一化到[0, max]
                # smallest = np.partition(img_copy.flatten(), 100)[99]
                # print("100 smallest value: ", smallest)
                # img_total = np.maximum(img - smallest, 0)
            else:
                raise ValueError('method should be mean, sum or square_sum')
            print(img_total[num].max())
        return img_total

def plot_vector(img, filename, vecmode = 1):
    if vecmode == 1:
        plt.figure(figsize=(12, 6))
        plt.xlim(0, 3600) # without compressed
        # plt.ylim(0, 1) # 1 for mean, 241230_cpr, 241230_gabor
        # plt.ylim(0, 120) # 120 for sum, 241230_cpr, 241230_gabor
        plt.ylim(0, 25) # 80 for square_sum, 241230_cpr, 241230_gabor；650 for 241230_slice
        plt.plot(img)
        plt.savefig(f'{savedir}{filename[:-4]}.png', dpi=300)
        plt.close()
    # # plot multiple vectors in multiple figures
    # else:
    #     for num in range(vecmode):
    #         plt.figure(figsize=(12, 6))
    #         plt.xlim(0, 360)
    #         # plt.ylim(0, 1) # 1 for mean, 241230_cpr, 241230_gabor
    #         # plt.ylim(0, 120) # 120 for sum, 241230_cpr, 241230_gabor
    #         plt.ylim(0, 30) # 30 for square_sum, 241230_cpr, vecmode = 5
    #         plt.plot(img[num])
    #         plt.savefig(f'{savedir_multi}{filename[:-4]}_{num}.png', dpi=300)
    #         plt.close()
    
    # plot multiple vectors in one figure
    else:
        plt.figure(figsize=(12, 6))
        # x轴为时间，每一张img是15秒。标记从start_time开始，到start_time + 15秒结束
        # start_time是filename[:-4][-1]，即（filename[:-4]的最后一位数字-1）*15
        if filename[:-4][-2] == '_':
            start_time = (int(filename[:-4][-1]) - 1) * 15
        else:
            start_time = (int(filename[:-4][-2:]) - 1) * 15
        x = np.linspace(start_time, start_time + 15, 3600) # without compressed
        plt.xticks(np.arange(start_time, start_time + 16, 1))
        plt.xlabel('Time (s)')
        plt.xlim(start_time, start_time + 15)
        # plt.ylim(0, 1) # 1 for mean, 241230_cpr, 241230_gabor
        # plt.ylim(0, 120) # 120 for sum, 241230_cpr, 241230_gabor
        plt.ylim(0, 30) # 30 for square_sum, 241230_cpr, vecmode = 4 / 5
        for num in range(vecmode):
            current_level = vecmode - num
            if current_level == 1:
                plt.plot(x, img[num], label = f'level {current_level} low frequency')
            elif current_level == vecmode:
                plt.plot(x, img[num], label = f'level {current_level} high frequency')
            else:
                plt.plot(x, img[num], label = f'level {current_level}')
        handles, labels = plt.gca().get_legend_handles_labels()
        plt.legend(handles[::-1], labels[::-1], title = f'{vecmode} levels from low to high', loc = 'upper right')
        plt.title(f'{filename[:-4]}')
        plt.ylabel('Square Sum')
        plt.savefig(f'{savedir_multi}{filename[:-4]}.png', dpi=300)
        plt.close()



path = os.listdir(filedir)
path.sort()

# project into one vector
for file in path:
    img_npy = project_vector(file, 'square_sum')
    np.save(f'{savedir_npy}{file[:-4]}.npy', img_npy) # save img_npy as .npy
    print(f'{file} is projected into vector')
    plot_vector(img_npy, file)

# # project into multiple vectors
# vecmode = 4
# for file in path:
#     img_npy = project_vector(file, 'square_sum', vecmode)
#     print(f'{file} is projected into multiple vectors')
#     plot_vector(img_npy, file, vecmode)
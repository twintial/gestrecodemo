import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.fftpack import rfft, irfft, fftfreq, fft, ifft, fftshift

import scipy.signal as signal

from audiotools.util import load_audio_data
from dsptools.filter import butter_bandpass_filter
from realtimesys.srpphat import get_steering_vector, cons_uca, plot_angspect, vec2theta

def power2db(power):
    if np.any(power < 0):
        raise ValueError("power less than 0")
    db = (10 * np.log10(power) + 300) - 300
    return db

def normalized_signal_fft(data, fs=48e3, figure=True, xlim=(19e3, 21e3)):
    N = len(data)
    y = np.abs(fft(data)) / N
    # 这里要不要乘2？要
    y_signle: np.ndarray = y[:int(np.round(N / 2))] * 2
    x = fftfreq(N) * fs
    x = x[x >= 0]
    if figure:
        plt.figure()
        plt.plot(x, y_signle)
        plt.xlim(xlim)
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('power')
        plt.title('single-side spectrum')
        plt.show()
    return x, y_signle

def normalized_signal_fft_with_fft(fft, title, fs=48e3, figure=True, xlim=(19e3, 21e3)):
    N = len(fft)
    y_signle: np.ndarray = fft[:int(np.round(N / 2))] * 2
    x = fftfreq(N) * fs
    x = x[x >= 0]
    if figure:
        plt.figure()
        plt.plot(x, y_signle)
        plt.xlim(xlim)
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('power')
        plt.title(title)
    return x, y_signle

def ifft1(a):
    N = len(a)
    f = []
    for k in range(N):
        F = 0
        for m in range(N):
            F += a[m] * np.exp(2j * np.pi * (m*k) / N)/N
        f.append(F)
    return f

def gcc_phat(a, b):
    w = np.hanning(len(a))
    # w = np.ones(len(a))
    f_s1 = fft(a * w)
    normalized_signal_fft_with_fft(np.abs(f_s1), 'f_s1',xlim=(0,20e3))
    plt.figure()
    f_s2 = fft(b * w)
    plt.plot(np.abs(f_s2))
    f_s2c = np.conj(f_s2)
    f_s = f_s1 * f_s2c
    denom = np.abs(f_s) + np.finfo(np.float32).eps
    f_s = f_s / denom  # This line is the only difference between GCC-PHAT and normal cross correlation
    return np.abs(ifft(f_s))

def denoise_fft(target_fft, noise_fft):
    # f1 = power2db(np.abs(target_fft))
    # f2 = power2db(np.abs(noise_fft))
    phase = np.angle(target_fft)
    f1 = np.abs(target_fft)
    f2 = np.abs(noise_fft)

    # f1 = target_fft
    # f2 = noise_fft
    doppler_fft_abs = f1 - f2



    doppler_fft_abs[doppler_fft_abs < 0] = 0

    # 完全去除noise，但是感觉有问题
    # sorted_index = np.argsort(f2)[::-1]
    # bins = f2[sorted_index]
    # print(np.sum(f2 > 2000))
    # noise_bins_num = 4
    # doppler_fft_abs[sorted_index[:noise_bins_num]] = 0

    # plt.plot(doppler_fft_abs)
    # plt.show()

    # for i in range(doppler_fft_abs.shape[0]):
    #     normalized_signal_fft_with_fft(f1[0], 'target_fft')
    #     normalized_signal_fft_with_fft(f2[0], 'noise_fft')
    #     normalized_signal_fft_with_fft(doppler_fft_abs[i], 'doppler_fft')
    #     plt.show()

    normalized_signal_fft_with_fft(f1 / len(doppler_fft_abs), 'target_fft')
    normalized_signal_fft_with_fft(f2 / len(doppler_fft_abs), 'noise_fft')
    normalized_signal_fft_with_fft(doppler_fft_abs / len(doppler_fft_abs), 'doppler_fft')
    plt.show()

    doppler_fft = doppler_fft_abs * np.exp(1j*phase)
    return doppler_fft

def gcc_phat_search(x_i, x_j, fs, tau):
    """
    :param x_i: real signal of mic i
    :param x_j: real signal of mic j
    :param fs: sample rate
    :return: np array, shape = (n_frames, num_of_search_grid)
    """

    w = np.hanning(len(x_j))

    fft_xi = fft(x_i * w)
    fft_xj = fft(x_j * w)

    P = fft_xi * fft_xj.conj()
    A = P / (np.abs(P)+np.finfo(np.float32).eps)
    # 为之后使用窗口做准备
    A = A.reshape(1, -1)

    num_bins = A.shape[1]
    # k = np.linspace(0, fs / 2, num_bins)
    # exp_part = np.outer(k, 2j * np.pi * tau)
    # R = np.dot(A, np.exp(exp_part))
    k = np.arange(num_bins)
    exp_part = np.outer(k, 2j * np.pi * tau * fs/num_bins)
    R = np.dot(A, np.exp(exp_part)) / num_bins
    return np.abs(R)


def gcc_phat_search_fft_denoise(x_i, x_j, fs, tau, noise_fft_i, noise_fft_j, window):
    """
    :param x_i: real signal of mic i
    :param x_j: real signal of mic j
    :param fs: sample rate
    :return: np array, shape = (n_frames, num_of_search_grid)
    """
    # 这里对fft去噪声
    fft_xi = fft(x_i * window)
    fft_xj = fft(x_j * window)

    denoised_fft_xi = denoise_fft(fft_xi, noise_fft_i)
    denoised_fft_xj = denoise_fft(fft_xj, noise_fft_j)

    P = denoised_fft_xi * denoised_fft_xj.conj()
    A = P / (np.abs(P) + np.finfo(np.float32).eps)
    # 为之后使用窗口做准备
    A = A.reshape(1, -1)

    num_bins = A.shape[1]
    # k = np.linspace(0, fs / 2, num_bins)
    # exp_part = np.outer(k, 2j * np.pi * tau)
    # R = np.dot(A, np.exp(exp_part))
    k = np.arange(num_bins)
    exp_part = np.outer(k, 2j * np.pi * tau * fs / num_bins)
    R = np.dot(A, np.exp(exp_part)) / num_bins
    return np.abs(R)

def srp_phat_denoise(raw_signal, mic_array_pos, search_grid, c, fs, noise_fft, window):
    assert raw_signal.shape[0] == mic_array_pos.shape[0]
    mic_num = mic_array_pos.shape[0]
    # grid, _ = create_spherical_grids(level)
    # print(grid.shape)
    E_d = np.zeros((1, search_grid.shape[0]))  # (num_frames, num_points), 之后要改
    for i in range(mic_num):
        for j in range(i + 1, mic_num):
            # tau is given in second, 这个也可以提前计算
            tau = get_steering_vector(mic_array_pos[i], mic_array_pos[j], c, search_grid)
            # t1 = time.time()
            R_ij = gcc_phat_search_fft_denoise(raw_signal[i], raw_signal[j], fs, tau, noise_fft[i], noise_fft[j], window)
            # t2 = time.time()
            E_d += R_ij
            # print('each pair time consuption: ', t2 - t1)
    return E_d

# 用ifft和用search_grid做对比
def compare():
    LENG = 2048
    data, fs = load_audio_data(r'D:\projects\pyprojects\gesturerecord\location\sound\0.wav', 'wav')
    # data = butter_bandpass_filter(data.T, 15e3, 23e3)
    data = data.T
    t = 3
    data = data[:-1, int(fs * t):int(fs * t) + LENG]
    #
    # for i, d in enumerate(data):
    #     plt.subplot(4, 2, i + 1)
    #     plt.plot(d)
    # plt.show()
    i = 0
    j = 1
    a = data[i]
    b = data[j]
    y = gcc_phat(a, b)
    print('ifft max ccor val: ', np.max(y))
    print('ifft delay of sample num: ', np.argmax(y))
    plt.figure()
    plt.plot(y)
    plt.title('ifft gcc')

    mic_array_pos = cons_uca(0.043)
    c = 343
    level = 4
    grid = np.load(rf'grid/{level}.npz')['grid']
    tau = get_steering_vector(mic_array_pos[i], mic_array_pos[j], c, grid)
    R = gcc_phat_search(a, b, fs, tau)

    # 画射线图
    # plot_angspect(R[0], grid, percentile=99)

    # r = R[0]
    # sorted_arg = np.argsort(r)[::-1]
    # print(r[sorted_arg])

    print('gccphat search max val: ', np.max(R))
    print('gccphat delay of sample num: ', tau[np.argmax(R)] * fs)
    max_p = grid[np.argmax(R)]
    print('point of  max val: ', max_p)
    print('angle of  max val: ', np.rad2deg(vec2theta([max_p])))
    plt.figure()
    plt.plot(R.reshape(-1))
    plt.title('gcc search')
    # plt.figure()
    # plt.plot(np.correlate(a,b,'full'))
    plt.show()

    # fig = plt.figure()
    # ax = Axes3D(fig)
    # ax.plot((0, max_p[0]), (0, max_p[1]), (0, max_p[2]))
    # ax.set_xlim3d(-1, 1)
    # ax.set_ylim3d(-1, 1)
    # ax.set_zlim3d(-1, 1)
    # ax.set_zlabel('Z', fontdict={'size': 15, 'color': 'red'})
    # ax.set_ylabel('Y', fontdict={'size': 15, 'color': 'red'})
    # ax.set_xlabel('X', fontdict={'size': 15, 'color': 'red'})
    # plt.show()


# 谱减法去噪
def fft_denoise_test():

    frame_len = 2048
    # 加窗窗口
    window = np.hanning(frame_len)
    # window = np.ones(frame_len)
    # window = np.hamming(frame_len)

    data, fs = load_audio_data(r'D:\projects\pyprojects\gesturerecord\location\20khz\0.wav', 'wav')
    data = data[fs * 1:, :-1].T
    data_filter = butter_bandpass_filter(data, 19e3, 21e3)

    # for f in data_filter:
    #     normalized_signal_fft(f)

    t_fs_n = 40000
    noise_data = data_filter[:, t_fs_n:t_fs_n+frame_len]
    t_fs_t = 48000 * 5 + 1000
    test_data = data_filter[:, t_fs_t:t_fs_t+frame_len]
    # denoise_fft(fft(test_data)/frame_len, fft(noise_data)/frame_len)

    i = 0
    j = 1
    mic_array_pos = cons_uca(0.043)
    c = 343
    level = 4
    grid = np.load(rf'grid/{level}.npz')['grid']
    tau = get_steering_vector(mic_array_pos[i], mic_array_pos[j], c, grid)
    noise_fft = fft(noise_data * window)
    R = gcc_phat_search_fft_denoise(test_data[i], test_data[j], fs, tau, noise_fft[i], noise_fft[j], window)

    print('gccphat search max val: ', np.max(R))
    print('gccphat delay of sample num: ', tau[np.argmax(R)] * fs)
    max_p = grid[np.argmax(R)]
    print('point of  max val: ', max_p)
    print('angle of  max val: ', np.rad2deg(vec2theta([max_p])))
    plt.figure()
    plt.plot(R.reshape(-1))
    plt.title('gcc search')
    # plt.figure()
    # plt.plot(np.correlate(a,b,'full'))
    plt.show()


# 谱减法去噪
def srp_fft_denoise_test():

    frame_len = 2048
    # 加窗窗口
    window = np.hanning(frame_len)
    # window = np.ones(frame_len)
    # window = np.hamming(frame_len)

    data, fs = load_audio_data(r'D:\projects\pyprojects\gesturerecord\location\20khz\0.wav', 'wav')
    data = data[fs * 1:, :-1].T
    data_filter = butter_bandpass_filter(data, 19e3, 21e3)

    # for f in data_filter:
    #     normalized_signal_fft(f)

    t_fs_n = 40000
    noise_data = data_filter[:, t_fs_n:t_fs_n+frame_len]
    t_fs_t = 48000 * 5 + 1000
    test_data = data_filter[:, t_fs_t:t_fs_t+frame_len]
    # denoise_fft(fft(test_data)/frame_len, fft(noise_data)/frame_len)

    mic_array_pos = cons_uca(0.043)
    c = 343
    level = 4
    grid = np.load(rf'grid/{level}_north.npz')['grid']
    noise_fft = fft(noise_data * window)
    E = srp_phat_denoise(test_data, mic_array_pos, grid, c, fs, noise_fft, window)
    sdevc = grid[np.argmax(E, axis=1)]  # source direction vector
    # print(sdevc)
    print('angle of  max val: ', np.rad2deg(vec2theta(sdevc)))
    print('=' * 50)

if __name__ == '__main__':
    compare()
    # fft_denoise_test()
    # srp_fft_denoise_test()
    # hanning_window = np.hanning(2048)
    # plt.plot(hanning_window)
    # plt.show()
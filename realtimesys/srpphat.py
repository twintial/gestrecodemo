import collections
import socket
import time

import numpy as np
import scipy.signal as signal
from scipy.fftpack import fft, fftfreq

from concurrent.futures import ThreadPoolExecutor

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # 空间三维画图

from audiotools.util import load_audio_data
from dsptools.filter import butter_bandpass_filter
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


def cons_uca(r):
    theta = np.pi / 3
    pos = [[0, 0, 0]]
    for i in range(6):
        pos.append([r * np.cos(theta * i), r * np.sin(theta * i), 0])
    return np.array(pos)


def plot_grid(points, ta):
    # 绘制散点图
    fig = plt.figure()
    ax = Axes3D(fig)
    # ax.scatter(a[:,0], a[:,1], a[:,2])
    for i in ta:
        ax.plot((i.a[0], i.b[0]), (i.a[1], i.b[1]), (i.a[2], i.b[2]), color='red')
        ax.plot((i.a[0], i.c[0]), (i.a[1], i.c[1]), (i.a[2], i.c[2]), color='red')
        ax.plot((i.b[0], i.c[0]), (i.b[1], i.c[1]), (i.b[2], i.c[2]), color='red')

    for i in range(points.shape[0]):
        ax.text(points[i, 0], points[i, 1], points[i, 2], str(i))
    # ax.scatter(0,0,0)
    ax.set_xlim3d(-1, 1)
    ax.set_ylim3d(-1, 1)
    ax.set_zlim3d(-1, 1)

    # 添加坐标轴(顺序是Z, Y, X)
    ax.set_zlabel('Z', fontdict={'size': 15, 'color': 'red'})
    ax.set_ylabel('Y', fontdict={'size': 15, 'color': 'red'})
    ax.set_xlabel('X', fontdict={'size': 15, 'color': 'red'})
    plt.show()
    # f, t, zxx = signal.spectrogram(x, fs)


def plot_angspect(R, grid, percentile=95):
    threshold = np.percentile(R, percentile)
    fig = plt.figure()
    ax = Axes3D(fig)
    # ax.scatter(grid[:, 0], grid[:, 1], grid[:, 2])
    for i, energy in enumerate(R):
        if energy < threshold:
            continue
        p = grid[i] * energy
        ax.plot((0, p[0]), (0, p[1]), (0, p[2]))
    ax.set_zlabel('Z', fontdict={'size': 15, 'color': 'red'})
    ax.set_ylabel('Y', fontdict={'size': 15, 'color': 'red'})
    ax.set_xlabel('X', fontdict={'size': 15, 'color': 'red'})
    # plt.show()


def get_steering_vector(pos_i, pos_j, c, dvec):
    """
    :param pos_i:
    :param pos_j:
    :param c:
    :param dvec: np array, shape=(10 × 4^L + 2, 3)
    :return:
    """
    pos_i = np.array(pos_i, dtype=np.float)
    pos_j = np.array(pos_j, dtype=np.float)
    dist = np.dot((pos_i - pos_j).reshape(1, 3), dvec.T)
    return (-dist.T / c).reshape(-1)  # 这个负号要不要?


def theta2vec(theta, r=1):
    # vectors = np.zeros((theta.shape[0], 3))
    # vectors[:, 0] = r*np.cos(theta[:, 1])*np.cos(theta[:, 0])
    # vectors[:, 1] = r*np.cos(theta[:, 1])*np.sin(theta[:, 0])
    # vectors[:, 2] = r*np.sin(theta[:, 1])
    vectors = []
    for t in theta:
        vectors.append(
            [r * np.cos(t[1]) * np.cos(t[0]),
             r * np.cos(t[1]) * np.sin(t[0]),
             r * np.sin(t[1])])
    return vectors
def vec2theta(vec):
    vec = np.array(vec)
    r = np.linalg.norm(vec[0])
    theta = np.zeros((vec.shape[0], 2))
    theta[:, 0] = np.arctan2(vec[:, 1], vec[:, 0])
    theta[:, 1] = np.arcsin(vec[:, 2]/r)
    return theta


def create_spherical_grids(r=0):
    class Triangle:
        def __init__(self, a: np.ndarray, b: np.ndarray, c: np.ndarray):
            self.a = a
            self.b = b
            self.c = c

    """
    :param r:resolution_level
    :return:np array, shape=(10 × 4^L + 2, 3)
    """
    def initial_icosahedral_grid_theta():
        theta = []
        theta.append([0, np.pi / 2])
        for i in range(2, 7):
            theta.append([(i - 3 / 2) * 2 * np.pi / 5, 2 * np.arcsin(1 / (2 * np.cos(3 * np.pi / 10))) - np.pi / 2])
        for i in range(7, 12):
            theta.append([(i - 7) * 2 * np.pi / 5, -2 * np.arcsin(1 / (2 * np.cos(3 * np.pi / 10))) + np.pi / 2])
        theta.append([0, -np.pi / 2])
        return np.array(theta)

    def initial_triangles(points):
        trianles = []
        trianles.append(Triangle(points[0], points[2], points[1]))
        trianles.append(Triangle(points[0], points[3], points[2]))
        trianles.append(Triangle(points[0], points[4], points[3]))
        trianles.append(Triangle(points[0], points[5], points[4]))
        trianles.append(Triangle(points[0], points[1], points[5]))
        trianles.append(Triangle(points[9], points[3], points[4]))
        trianles.append(Triangle(points[10], points[4], points[5]))
        trianles.append(Triangle(points[6], points[5], points[1]))
        trianles.append(Triangle(points[7], points[1], points[2]))
        trianles.append(Triangle(points[8], points[2], points[3]))
        trianles.append(Triangle(points[4], points[9], points[10]))
        trianles.append(Triangle(points[5], points[10], points[6]))
        trianles.append(Triangle(points[1], points[6], points[7]))
        trianles.append(Triangle(points[2], points[7], points[8]))
        trianles.append(Triangle(points[3], points[8], points[9]))
        trianles.append(Triangle(points[11], points[6], points[7]))
        trianles.append(Triangle(points[11], points[7], points[8]))
        trianles.append(Triangle(points[11], points[8], points[9]))
        trianles.append(Triangle(points[11], points[9], points[10]))
        trianles.append(Triangle(points[11], points[10], points[6]))
        return trianles

    def get_next_points_and_triangles(points, triangles):
        def find_point(point, points):
            for p in points:
                if np.all(point == p):
                    return False
            return True

        new_triangles = []
        for triangle in triangles:
            point1 = triangle.a + triangle.b
            point2 = triangle.a + triangle.c
            point3 = triangle.b + triangle.c
            point1 = point1 / np.linalg.norm(point1)
            point2 = point2 / np.linalg.norm(point2)
            point3 = point3 / np.linalg.norm(point3)
            if find_point(point1, points):
                points.append(point1)
            if find_point(point2, points):
                points.append(point2)
            if find_point(point3, points):
                points.append(point3)
            # 1 triangle subdivide into 4
            new_triangles.append(Triangle(triangle.a, point1, point2))
            new_triangles.append(Triangle(triangle.b, point1, point3))
            new_triangles.append(Triangle(triangle.c, point3, point2))
            new_triangles.append(Triangle(point1, point2, point3))
        return points, new_triangles

    initial_theta = initial_icosahedral_grid_theta()
    # print(np.rad2deg(initial_theta))
    points = theta2vec(initial_theta)
    # x = vec2theta(points)
    # print(np.rad2deg(x).astype(np.int))
    # points = theta2vec(x)

    # print(np.all())

    # initial triangles
    triangles = initial_triangles(np.array(points))
    ilevel = 0
    while ilevel < r:
        points, triangles = get_next_points_and_triangles(points, triangles)
        ilevel += 1
    return np.array(points), triangles


# 暂时不用窗口直接对输入的数据做fft
def gcc_phat(x_i, x_j, fs, tau):
    """
    :param x_i: real signal of mic i
    :param x_j: real signal of mic j
    :param fs: sample rate
    :param search_grid: grid for search, each point in grid is a 3-D vector
    :return: np array, shape = (n_frames, num_of_search_grid)
    """
    w = np.hanning(len(x_j))
    # w = np.ones(len(x_j))
    P = fft(x_i * w) * fft(x_j * w).conj()
    A = P / (np.abs(P)+np.finfo(np.float32).eps)

    # 为之后使用窗口做准备
    A = A.reshape(1, -1)

    num_bins = A.shape[1]
    k = np.arange(num_bins)
    # t1 = time.time()
    exp_part = np.outer(k, 2j * np.pi * tau * fs/num_bins)
    # t2 = time.time()
    # print('ifft1 time consuption: ', t2 - t1)
    # t1 = time.time()
    R = np.dot(A, np.exp(exp_part)) / num_bins
    # t2 = time.time()
    # print('ifft2 time consuption: ', t2 - t1)
    return np.abs(R)
def srp_phat(raw_signal, mic_array_pos, search_grid, c, fs):
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
            R_ij = gcc_phat(raw_signal[i], raw_signal[j], fs, tau)
            # t2 = time.time()
            E_d += R_ij
            # print('each pair time consuption: ', t2 - t1)
    return E_d
# 貌似没有改进
@DeprecationWarning
def calculate_pairs_tau(mic_array_pos, search_grid, c):
    mic_num = mic_array_pos.shape[0]
    pair_dic = {}
    for i in range(mic_num):
        for j in range(i + 1, mic_num):
            tau = get_steering_vector(mic_array_pos[i], mic_array_pos[j], c, search_grid)
            pair_dic[str(i) + str(j)] = tau
    return pair_dic
@DeprecationWarning
def srp_phat_previous_tau(raw_signal, mic_array_pos, search_grid, pairs_tau, fs):
    assert raw_signal.shape[0] == mic_array_pos.shape[0]
    mic_num = mic_array_pos.shape[0]
    # grid, _ = create_spherical_grids(level)
    # print(grid.shape)
    E_d = np.zeros((1, search_grid.shape[0]))  # (num_frames, num_points), 之后要改
    for i in range(mic_num):
        for j in range(i + 1, mic_num):
            # tau is given in second, 这个也可以提前计算
            tau = pairs_tau[str(i) + str(j)]
            R_ij = gcc_phat(raw_signal[i], raw_signal[j], fs, tau)
            E_d += R_ij
    return E_d

# 多线程,失败
@DeprecationWarning
def srp_phat_muti_thread(raw_signal, mic_array_pos, search_grid, c, fs):
    def calculate_Rij(i, j):
        tau = get_steering_vector(mic_array_pos[i], mic_array_pos[j], c, search_grid)
        R_ij = gcc_phat(raw_signal[i], raw_signal[j], fs, tau)
        return R_ij
    assert raw_signal.shape[0] == mic_array_pos.shape[0]
    mic_num = mic_array_pos.shape[0]
    executor = ThreadPoolExecutor(max_workers=1)
    E_d = np.zeros((1, search_grid.shape[0]))  # (num_frames, num_points), 之后要改
    t1 = time.time()
    i_s = []
    j_s = []
    for i in range(mic_num):
        for j in range(i+1):
            i_s.append(i)
            j_s.append(j)
    for R_ij in executor.map(calculate_Rij, i_s, j_s):
        E_d += R_ij
    t2 = time.time()
    print('each pair time consuption: ', t2 - t1)
    return E_d


# 尝试用矩阵加速。没用，矩阵太大了
@DeprecationWarning
def calculate_stack_fft_and_pairs_tau(raw_signals, mic_array_pos, search_grid, c):
    mic_num = mic_array_pos.shape[0]
    pair_tau = []
    pair_fft = []
    for i in range(mic_num):
        for j in range(i + 1, mic_num):
            tau = get_steering_vector(mic_array_pos[i], mic_array_pos[j], c, search_grid)
            pair_tau.append(tau)
            P = fft(raw_signals[i]) * fft(raw_signals[j]).conj()
            A = P / (np.abs(P) + np.finfo(np.float32).eps)
            # 为之后使用窗口做准备
            A = A.reshape(1, -1)
            pair_fft.append(A)
    h_stack_fft = np.hstack([pf for pf in pair_fft])
    return np.array(pair_tau), h_stack_fft
@DeprecationWarning
def gcc_phat_m(stack_fft, fs, pair_tau):
    """
    :param x_i: real signal of mic i
    :param x_j: real signal of mic j
    :param fs: sample rate
    :param search_grid: grid for search, each point in grid is a 3-D vector
    :return: np array, shape = (n_frames, num_of_search_grid)
    """

    num_bins = stack_fft.shape[1] / pair_tau.shape[0]
    k = np.arange(num_bins)
    t1 = time.time()
    exp_part = np.outer(k, 2j * np.pi * pair_tau[0] * fs/num_bins)
    for i in range(1, pair_tau.shape[0]):
        exp_part_temp = np.outer(k, 2j * np.pi * pair_tau[i] * fs / num_bins)
        exp_part = np.vstack((exp_part, exp_part_temp))
    t2 = time.time()
    print('ifft1 time consuption: ', t2 - t1)
    t1 = time.time()
    R = np.dot(stack_fft, np.exp(exp_part)) / (num_bins * pair_tau.shape[0])
    t2 = time.time()
    print('ifft2 time consuption: ', t2 - t1)
    return np.abs(R)
@DeprecationWarning
def srp_phat_m(raw_signal, mic_array_pos, stack_raw_signal_fft, pairs_tau, fs):
    assert raw_signal.shape[0] == mic_array_pos.shape[0]
    mic_num = mic_array_pos.shape[0]
    # grid, _ = create_spherical_grids(level)
    # print(grid.shape)
    t1 = time.time()
    E_d = gcc_phat_m(stack_raw_signal_fft, fs, pairs_tau)
    t2 = time.time()
    return E_d

def split_frame():
    c = 343
    frame_count = 1024
    data, fs = load_audio_data(r'D:\projects\pyprojects\gesturerecord\location\1khz\0.wav', 'wav')
    skip_time = int(fs * 1)
    data = data[skip_time:, :-1].T
    # search unit circle
    level = 3
    grid: np.ndarray = np.load(rf'grid/{level}_north.npz')['grid']
    # mic mem pos
    pos = cons_uca(0.043)
    # calculate tau previously
    # pairs_tau = calculate_pairs_tau(pos, grid, c)
    for i in range(0, data.shape[1], frame_count):
        data_seg = data[:, i:i+frame_count]
        # 噪声不做,随便写的
        if np.max(abs(fft(data_seg[0] / len(data_seg[0])))) < 10:
            continue
        print('time: ', (skip_time + i)/fs)
        t1 = time.time()
        E = srp_phat(data_seg, pos, grid, c, fs)
        # E = srp_phat_muti_thread(data, pos, grid, c, fs)
        # E = srp_phat_previous_tau(data_seg, pos, grid, pairs_tau, fs)
        t2 = time.time()
        print('srp_phat time consumption: ', t2-t1)
        sdevc = grid[np.argmax(E, axis=1)]  # source direction vector
        # print(sdevc)
        print('angle of  max val: ', np.rad2deg(vec2theta(sdevc)))
        print('='*50)
        # plot_angspect(E_d[0], grid)

# 失败
@DeprecationWarning
def split_frame_m():
    c = 343
    frame_count = 2048
    data, fs = load_audio_data(r'D:\projects\pyprojects\soundphase\calib\0\0.wav', 'wav')
    skip_time = int(fs * 1)
    data = data[skip_time:, :-1].T
    # search unit circle
    level = 4
    grid: np.ndarray = np.load(rf'grid/{level}.npz')['grid']
    # mic mem pos
    pos = cons_uca(0.043)
    for i in range(0, data.shape[1], frame_count):
        data_seg = data[:, i:i+frame_count]
        # 噪声不做,随便写的
        if np.max(abs(fft(data_seg[0] / len(data_seg[0])))) < 1:
            continue
        print('time: ', (skip_time + i)/fs)
        t1 = time.time()
        pair_tau, stack_fft = calculate_stack_fft_and_pairs_tau(data_seg, pos, grid, c)
        t2 = time.time()
        print('previous calculation time consumption: ', t2-t1)
        t1 = time.time()
        E = srp_phat_m(data_seg, pos, stack_fft, pair_tau, fs)
        # E = srp_phat_previous_tau(data_seg, pos, grid, pairs_tau, fs)
        t2 = time.time()
        print('srp_phat time consumption: ', t2-t1)
        sdevc = grid[np.argmax(E, axis=1)]  # source direction vector
        # print(sdevc)
        print('angle of  max val: ', np.rad2deg(vec2theta(sdevc)))
        print('='*50)
        # plot_angspect(E_d[0], grid)

# 不清楚式超声波不行还是正弦波不行，多半是后者
def real_time_run_audible_voice():
    frame_count = 2048
    channels = 8
    c = 343
    fs = 48000
    # search unit circle
    level = 4
    grid: np.ndarray = np.load(rf'grid/{level}_north.npz')['grid']
    # mic mem pos
    pos = cons_uca(0.043)
    # socket
    address = ('127.0.0.1', 31500)
    tcp_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    tcp_socket.connect(address)
    while True:
        rdata = tcp_socket.recv(frame_count * channels * 2)
        if len(rdata) == 0:
            break
        data = np.frombuffer(rdata, dtype=np.int16)
        data = data.reshape(-1, channels).T
        data = data[:7, 512:-512]
        # 噪声不做,随便写的
        if np.max(abs(fft(data[0] / len(data[0])))) < 10:
            continue
        print("hear voice")
        E = srp_phat(data, pos, grid, c, fs)
        sdevc = grid[np.argmax(E, axis=1)]  # source direction vector
        print('angle of  max val: ', np.rad2deg(vec2theta(sdevc)))
        print("="*50)

def denoise_fft(target_fft, noise_fft):
    # f1 = power2db(np.abs(target_fft))
    # f2 = power2db(np.abs(noise_fft))
    phase = np.angle(target_fft)
    f1 = np.abs(target_fft)
    f2 = np.abs(noise_fft)

    # f1 = target_fft
    # f2 = noise_fft
    assert f1.shape == f2.shape
    doppler_fft_abs = f1 - f2

    doppler_fft_abs[doppler_fft_abs < 0] = 0

    sorted_index = np.argsort(f2)[:, ::-1]
    # bins = f2[sorted_index]
    noise_bins_num = 8
    for i in range(target_fft.shape[0]):
        doppler_fft_abs[i, sorted_index[i, :noise_bins_num]] = 0

    # plt.plot(doppler_fft_abs)
    # plt.show()

    # for i in range(doppler_fft_abs.shape[0]):
    #     normalized_signal_fft_with_fft(f1[0], 'target_fft')
    #     normalized_signal_fft_with_fft(f2[0], 'noise_fft')
    #     normalized_signal_fft_with_fft(doppler_fft_abs[i], 'doppler_fft')
    #     plt.show()
    #
    # normalized_signal_fft_with_fft(f1 / len(doppler_fft_abs), 'target_fft')
    # normalized_signal_fft_with_fft(f2 / len(doppler_fft_abs), 'noise_fft')

    # if np.mean(np.max(doppler_fft_abs, axis=1)) > 1:
    #     normalized_signal_fft_with_fft(doppler_fft_abs[0] / len(doppler_fft_abs), 'doppler_fft')
    #     plt.show()

    doppler_fft = doppler_fft_abs * np.exp(1j*phase)
    return doppler_fft
def gcc_phat_search_fft_denoise(denoised_fft_xi, denoised_fft_xj, fs, tau):
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
def srp_phat_denoise(doppler_fft, mic_array_pos, search_grid, c, fs):
    assert doppler_fft.shape[0] == mic_array_pos.shape[0]
    mic_num = mic_array_pos.shape[0]
    # grid, _ = create_spherical_grids(level)
    # print(grid.shape)
    E_d = np.zeros((1, search_grid.shape[0]))  # (num_frames, num_points), 之后要改
    for i in range(mic_num):
        for j in range(i + 1, mic_num):
            # tau is given in second, 这个也可以提前计算
            tau = get_steering_vector(mic_array_pos[i], mic_array_pos[j], c, search_grid)
            # t1 = time.time()
            R_ij = gcc_phat_search_fft_denoise(doppler_fft[i], doppler_fft[j], fs, tau)
            # t2 = time.time()
            E_d += R_ij
            # print('each pair time consuption: ', t2 - t1)
    return E_d
# 失败
def real_time_run_reflection_ultrasonic_sound():
    frame_count = 2048
    channels = 8
    c = 343
    fs = 48000
    # search unit circle
    level = 4
    grid: np.ndarray = np.load(rf'grid/{level}_north.npz')['grid']
    # mic mem pos
    pos = cons_uca(0.043)
    # noise
    noise_fft = None
    # fft window
    window = np.hanning(2048)
    # socket
    address = ('127.0.0.1', 31500)
    tcp_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    tcp_socket.connect(address)
    while True:
        rdata = tcp_socket.recv(frame_count * channels * 2)
        if len(rdata) == 0:
            break
        data = np.frombuffer(rdata, dtype=np.int16)
        data = data.reshape(-1, channels).T
        data = data[:7, :]
        data = butter_bandpass_filter(data, 19e3, 21e3)
        # if noise_fft is None:
        #     noise_fft = fft(data * window)
        #     continue
        data_fft = fft(data * window)
        # 噪声不做,随便写的
        data_fft_abs = np.abs(data_fft)
        # plt.plot(data_fft_abs[0])
        # plt.show()
        non_noise_bins = data_fft_abs > 2000
        # print(np.mean(np.sum(non_noise_bins, axis=1)))
        if np.mean(np.sum(non_noise_bins, axis=1)) > 6:
            if noise_fft is None:
                noise_fft = fft(data * window)
                continue
        if np.mean(np.sum(non_noise_bins, axis=1)) < 10:
            # noise_fft = data_fft
            continue
        doppler_fft = denoise_fft(data_fft, noise_fft)
        print("hear voice, fft amplitude: ", np.mean(np.max(doppler_fft, axis=1)))
        E = srp_phat_denoise(doppler_fft, pos, grid, c, fs)
        sorted_i_E = np.argsort(E)[:, ::-1]

        n_max = grid[sorted_i_E[0]]

        all_vector = np.rad2deg(vec2theta(n_max))

        sdevc = grid[np.argmax(E, axis=1)]  # source direction vector
        print('angle of  max val: ', np.rad2deg(vec2theta(sdevc)))
        print("="*50)

if __name__ == '__main__':
    # genrate grid
    # r = 3
    # p, ta = create_spherical_grids(r=r)
    # p = p[p[:, 2] >= 0]
    # print(p.shape)
    # plot_grid(p, ta)
    # np.savez_compressed(rf'grid/{r}_north.npz', grid=p)
    pass
    # data, fs = load_audio_data(r'D:\projects\pyprojects\soundphase\calib\0\mic2.wav', 'wav')
    # # data = data[48000 * 1 + 44000:48000+44000+1024, :-1].T
    # data = data[48000 * 1+90000:48000 + 90000+1024, :-1].T
    # for i, d in enumerate(data):
    #     plt.subplot(4,2,i+1)
    #     plt.plot(d)
    # plt.show()
    # pos = cons_uca(0.043)
    # # plt.plot(pos[:,0],pos[:,1])
    # plt.show()
    # c = 343
    # E = srp_phat(data, pos, c, fs, level=4)

    # split_frame()
    # split_frame_m()
    real_time_run_audible_voice()
    # real_time_run_reflection_ultrasonic_sound()

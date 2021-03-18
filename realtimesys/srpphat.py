import collections
import socket

import numpy as np
import scipy.signal as signal
from scipy.fftpack import fft, fftfreq

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # 空间三维画图

from audiotools.util import load_audio_data


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
    # 要看是否对应上了
    P = fft(x_i) * fft(x_j).conj()
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

def srp_phat(raw_signal, mic_array_pos, c, fs, level=1):
    assert raw_signal.shape[0] == mic_array_pos.shape[0]
    mic_num = mic_array_pos.shape[0]
    # grid, _ = create_spherical_grids(level)
    grid: np.ndarray = np.load(rf'grid/{level}.npz')['grid']
    # print(grid.shape)
    E_d = np.zeros((1, grid.shape[0]))  # (num_frames, num_points)
    for i in range(mic_num):
        for j in range(i + 1, mic_num):
            # tau is given in second, 这个也可以提前计算
            tau = get_steering_vector(mic_array_pos[i], mic_array_pos[j], c, grid)
            R_ij = gcc_phat(raw_signal[i], raw_signal[j], fs, tau)
            E_d += R_ij
    sdevc = grid[np.argmax(E_d, axis=1)]  # source direction vector
    # print(sdevc)
    print('angle of  max val: ', np.rad2deg(vec2theta(sdevc)))
    # plot_angspect(E_d[0], grid)
    return E_d


def split_frame():
    c = 343
    frame_count = 256
    data, fs = load_audio_data(r'D:\projects\pyprojects\soundphase\calib\0\0.wav', 'wav')
    skip_time = int(fs * 1.8)
    data = data[skip_time:, :-1].T
    for i in range(0, data.shape[1], frame_count):
        data_seg = data[:, i:i+frame_count]
        # 噪声不做,随便写的
        if np.max(abs(fft(data_seg[0] / len(data_seg[0])))) < 10:
            continue
        print('time: ', (skip_time + i)/fs)
        pos = cons_uca(0.043)
        E = srp_phat(data_seg, pos, c, fs, level=4)


def real_time_run():
    pos = cons_uca(0.043)
    frame_count = 2048
    channels = 8
    c = 343
    fs = 48000
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
        # 噪声不做,随便写的
        if np.max(abs(fft(data[0] / len(data[0])))) < 10:
            continue
        E = srp_phat(data, pos, c, fs, level=4)


if __name__ == '__main__':
    # r = 0
    # p, ta = create_spherical_grids(r=r)
    # plot_grid(p, ta)
    # np.savez_compressed(rf'grid/{r}.npz', grid=p)
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
    real_time_run()

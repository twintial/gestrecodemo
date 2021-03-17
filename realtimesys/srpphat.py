import collections

import numpy as np
import scipy.signal as signal
from scipy.fftpack import fft, fftfreq

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # 空间三维画图


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
    print(np.rad2deg(initial_theta))
    points = theta2vec(initial_theta)
    x = vec2theta(points)
    print(np.rad2deg(x).astype(np.int))
    # points = theta2vec(x)
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
    A = P / np.abs(P)
    # 为之后使用窗口做准备
    A = A.reshape(1, -1)

    num_bins = A.reshape[1]
    k = np.linspace(0, fs / 2, num_bins)
    exp_part = np.outer(k, 2j * np.pi * tau)
    R = np.dot(A, exp_part)
    return R.real


def srp_phat(raw_signal, mic_array_pos, c, fs):
    assert raw_signal.shape[0] == mic_array_pos.shape[0]
    mic_num = mic_array_pos.shape[0]
    grid, _ = create_spherical_grids(0)
    E_d = np.zeros(grid.shape[0])
    for i in range(mic_num):
        for j in range(i, mic_num):
            # tau is given in second
            tau = get_steering_vector(mic_array_pos[i], mic_array_pos[j], c, grid)
            R_ij = gcc_phat(raw_signal[i], raw_signal[j], fs, tau)
            E_d += R_ij
    sdevc = grid[np.argmax(E_d, axis=1)]  # source direction vector


if __name__ == '__main__':
    pass
    p, ta = create_spherical_grids(r=0)
    plot_grid(p, ta)
    print(p.shape)
    # print(a)
    # print(a.shape[0])

import numpy as np
import scipy.signal as signal
from scipy.fftpack import fft, fftfreq

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # 空间三维画图\

def cons_uca(r):
    theta = np.pi / 3
    pos = [[0, 0, 0]]
    for i in range(6):
        pos.append([r * np.cos(theta * i), r * np.sin(theta * i), 0])
    return np.array(pos)


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

def theta2vector(theta, r=1):
    vectors = np.zeros((theta.shape[0], 3))
    vectors[:, 0] = r*np.cos(theta[:, 1])*np.cos(theta[:, 0])
    vectors[:, 1] = r*np.cos(theta[:, 1])*np.sin(theta[:, 0])
    vectors[:, 2] = r*np.sin(theta[:, 1])
    return vectors

def create_spherical_grids(r=0):
    class Triangle:
        def __init__(self, a, b, c):
            self.a = a
            self.b = b
            self.c = c
    """
    :param r:resolution_level
    :return:np array, shape=(10 × 4^L + 2, 3)
    """
    def initial_icosahedral_grid_theta():
        theta = []
        theta.append([0, np.pi/2])
        for i in range(2, 7):
            theta.append([(i-3/2)*2*np.pi/5, 2*np.arcsin(1/(2*np.cos(3*np.pi/10)))-np.pi/2])
        for i in range(7, 12):
            theta.append([(i-7)*2*np.pi/5, -2*np.arcsin(1/(2*np.cos(3*np.pi/10)))+np.pi/2])
        theta.append([0, -np.pi/2])
        return np.array(theta)
    initial_theta = initial_icosahedral_grid_theta()
    print(initial_theta)
    points = theta2vector(initial_theta)
    # initial triangles
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
    triangles = initial_triangles(points)
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
    mic_num = mic_array_pos.shape[0]
    grid = create_spherical_grids(0)
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
    a, ta = create_spherical_grids(r=0)
    print(a)

    # 绘制散点图
    fig = plt.figure()
    ax = Axes3D(fig)
    # ax.scatter(a[:,0], a[:,1], a[:,2])
    for i in ta:
        ax.plot((i.a[0], i.b[0]),(i.a[1], i.b[1]),(i.a[2], i.b[2]), color='red')
        ax.plot((i.a[0], i.c[0]),(i.a[1], i.c[1]),(i.a[2], i.c[2]), color='red')
        ax.plot((i.b[0], i.c[0]),(i.b[1], i.c[1]),(i.b[2], i.c[2]), color='red')

    for i in range(a.shape[0]):
        ax.text(a[i, 0], a[i, 1], a[i, 2], str(i))
    # ax.scatter(0,0,0)
    ax.set_xlim3d(-1,1)
    ax.set_ylim3d(-1,1)
    ax.set_zlim3d(-1,1)

    # 添加坐标轴(顺序是Z, Y, X)
    ax.set_zlabel('Z', fontdict={'size': 15, 'color': 'red'})
    ax.set_ylabel('Y', fontdict={'size': 15, 'color': 'red'})
    ax.set_xlabel('X', fontdict={'size': 15, 'color': 'red'})
    plt.show()
    # f, t, zxx = signal.spectrogram(x, fs)
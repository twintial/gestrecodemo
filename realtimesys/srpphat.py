import numpy as np
import scipy.signal as signal
from scipy.fftpack import fft, fftfreq

def cons_uca(r):
    theta = np.pi / 3
    pos = [[0, 0, 0]]
    for i in range(6):
        pos.append([r * np.cos(theta * i), r * np.sin(theta * i), 0])
    return np.array(pos)

# 这里使用时，pos只放2个元素
def get_steering_vector(pos_i,pos_j, c, dvec):
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


def create_spherical_grid(r=0):
    """
    :param r:resolution_level
    :return:np array, shape=(10 × 4^L + 2, 3)
    """
    spherical_grid = []
    return np.array(spherical_grid)

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
    grid = create_spherical_grid(0)
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
    a=[[1,2,3,4],[5,6,7,1]]
    a = np.array(a)
    print(np.argmax(a, axis=1))
    # f, t, zxx = signal.spectrogram(x, fs)
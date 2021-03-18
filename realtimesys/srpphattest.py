import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.fftpack import rfft, irfft, fftfreq, fft, ifft, fftshift

from audiotools.util import load_audio_data
from realtimesys.srpphat import get_steering_vector, cons_uca, plot_angspect, vec2theta


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
    f_s1 = fft(a)
    f_s2 = fft(b)
    f_s2c = np.conj(f_s2)
    f_s = f_s1 * f_s2c
    denom = np.abs(f_s) + np.finfo(np.float32).eps
    f_s = f_s / denom  # This line is the only difference between GCC-PHAT and normal cross correlation
    return np.abs(ifft(f_s))

def gcc_phat_search(x_i, x_j, fs, tau):
    """
    :param x_i: real signal of mic i
    :param x_j: real signal of mic j
    :param fs: sample rate
    :param search_grid: grid for search, each point in grid is a 3-D vector
    :return: np array, shape = (n_frames, num_of_search_grid)
    """
    # 要看是否对应上了
    P = fft(x_i) * fft(x_j).conj()
    plt.figure()
    plt.plot(abs(fft(x_i)) / len(x_i))
    # plt.plot(abs(fft(x_i))/len(x_i))
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

LENG = 500
# a = np.array(np.random.rand(LENG))
# b = np.array(np.random.rand(LENG))

# data, fs = load_audio_data(r'D:\projects\pyprojects\soundphase\calib\0\0.wav', 'wav')
# data = data[48000 * 1 + 44000:48000+44000+1024, :-1].T
# data = data[48000 * 1:48000 + 2048, :-1].T
# data = data[48000 * 1 + 90000:48000 + 90000 + 1024, :-1].T
# for i, d in enumerate(data):
#     plt.subplot(4, 2, i + 1)
#     plt.plot(d)
# plt.show()

data, fs = load_audio_data(r'D:\projects\pyprojects\gesturerecord\location\0\0.wav', 'wav')
skip_time = int(fs * 1)
data = data[skip_time:, :-1].T
data = data[:,48000:48000+1024]
#
# for i, d in enumerate(data):
#     plt.subplot(4, 2, i + 1)
#     plt.plot(d)
# plt.show()
i=0
j=1
a = data[i]
b = data[j]
y = gcc_phat(a,b)
print('ifft max ccor val: ',np.max(y))
print('ifft delay of sample num: ',np.argmax(y))
plt.figure()
plt.plot(y)
plt.title('ifft gcc')

mic_array_pos = cons_uca(0.043)
c = 343
level = 4
grid = np.load(rf'grid/{level}.npz')['grid']
tau = get_steering_vector(mic_array_pos[i], mic_array_pos[j], c, grid)
R = gcc_phat_search(a, b, fs, tau)

plot_angspect(R[0], grid, percentile=99)

# r = R[0]
# sorted_arg = np.argsort(r)[::-1]
# print(r[sorted_arg])

print('gccphat search max val: ',np.max(R))
print('gccphat delay of sample num: ', tau[np.argmax(R)] * fs)
max_p = grid[np.argmax(R)]
print('point of  max val: ',max_p)
print('angle of  max val: ',np.rad2deg(vec2theta([max_p])))
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
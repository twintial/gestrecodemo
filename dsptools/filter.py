from scipy.signal import lfilter, butter
import numpy as np

# filtfilt没有偏移，可以考虑换成这个

# butter band滤波
def butter_bandpass_filter(data, lowcut, highcut, fs=48e3, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band', analog=False)
    return lfilter(b, a, data)


# butter lowpass
def butter_lowpass_filter(data, cutoff, fs=48e3, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return lfilter(b, a, data)


# butter highpass
def butter_highpass_filter(data, cutoff, fs=48e3, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    return lfilter(b, a, data)


def move_average_overlap_filter(data, win_size=200, overlap=100, axis=-1):
    if len(data.shape) == 1:
        data = data.reshape((1, -1))
    ret = np.cumsum(data, axis=axis)
    ret[:, win_size:] = ret[:, win_size:] - ret[:, :-win_size]
    result = ret[:, win_size - 1:] / win_size
    index = np.arange(0, result.shape[1], overlap)
    return result[:, index]


# demo，很重要
from scipy import signal
import matplotlib.pyplot as plt
t = np.linspace(-1, 1, 201)
x = (np.sin(2 * np.pi * 0.75 * t * (1 - t) + 2.1) +
     0.1 * np.sin(2 * np.pi * 1.25 * t + 1) +
     0.18 * np.cos(2 * np.pi * 3.85 * t))
xn = x + np.random.randn(len(t)) * 0.08
b, a = signal.butter(3, 0.05)
# 构造初始状态
zi = signal.lfilter_zi(b, a)
dss = zi
data = []
data3, _ = signal.lfilter(b, a, xn, zi=dss)
print(dss)
for i in xn:
    z, dss = signal.lfilter(b, a, [i], zi=dss)

    data.append(z)
#     data2.append(z2)
# print(data2)
data2 = signal.filtfilt(b, a, xn)
plt.figure()
plt.plot(t, xn, 'b', alpha=0.75)
plt.plot(t, data, 'r--')
plt.plot(t, data2, 'g')
plt.plot(t, data3, 'y^', alpha=0.3)
plt.grid(True)
plt.show()
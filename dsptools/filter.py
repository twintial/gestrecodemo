from scipy.signal import lfilter, butter
import numpy as np


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

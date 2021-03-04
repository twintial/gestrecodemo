import numpy as np

from dsptools.filter import butter_lowpass_filter


def power2db(power):
    if np.any(power < 0):
        raise ValueError("power less than 0")
    db = 10 * np.log10(power)
    return db


# sinusiod，支持多声道，data.shape = (num_of_channels, frames)
def get_cos_IQ(data: np.ndarray, f, fs=48e3) -> (np.ndarray, np.ndarray):
    frames = data.shape[1]
    times = np.arange(0, frames) * 1 / fs
    I_raw = np.cos(2 * np.pi * f * times) * data
    Q_raw = -np.sin(2 * np.pi * f * times) * data
    # 低通滤波
    # 这里的axis要看一下对不对
    I = butter_lowpass_filter(I_raw, 200)
    Q = butter_lowpass_filter(Q_raw, 200)
    return I, Q


def get_cos_IQ_raw(data: np.ndarray, f, fs=48e3) -> (np.ndarray, np.ndarray):
    frames = data.shape[1]
    times = np.arange(0, frames) * 1 / fs
    I_raw = np.cos(2 * np.pi * f * times) * data
    Q_raw = -np.sin(2 * np.pi * f * times) * data
    return I_raw, Q_raw

# 多了offset
def get_cos_IQ_raw_offset(data: np.ndarray, f, offset, fs=48e3) -> (np.ndarray, np.ndarray):
    frames = data.shape[1]
    # offset会一直增长，可能存在问题
    times = np.arange(offset, offset + frames) * 1 / fs
    I_raw = np.cos(2 * np.pi * f * times) * data
    Q_raw = -np.sin(2 * np.pi * f * times) * data
    return I_raw, Q_raw


def get_phase(I: np.ndarray, Q: np.ndarray) -> np.ndarray:
    signal = I + 1j * Q
    angle = np.angle(signal)
    # angle = angle[:, 2048:]  # 这里存在问题，减少了信息量
    # 这里的axis要看一下对不对
    unwrap_angle = np.unwrap(angle)
    return unwrap_angle


def get_magnitude(I: np.ndarray, Q: np.ndarray) -> np.ndarray:
    signal = I + 1j * Q
    magn = np.abs(signal)
    magn = 10*np.log10(magn)
    return magn

'''
将想要发射的声波生成成wav文件
'''
import numpy as np
import wave
from scipy.io import wavfile

from pyaudio import PyAudio

from realtimesys.signalgenerator import cos_wave

def get_sinusoid(f0, step, n, fs, t):
    A = [1] * n
    alpha = 1 / sum(A)
    y = A[0] * cos_wave(1, f0, fs, t)
    for i in range(1, n):
        y = y + A[i] * cos_wave(1, f0 + i * step, fs, t)
    signal = alpha * y
    return signal

if __name__ == '__main__':
    # a = np.array([1, 2, 3], dtype=np.int)
    # print(b''.join(a))

    fs = 48000
    # t = 300
    # A = [1, 1, 1, 1, 1, 1, 1, 1]
    # alpha = 1 / sum(A)
    # y = A[0] * cos_wave(1, 17000, fs, t)
    # for i in range(1, 7):
    #     y = y + A[i] * cos_wave(1, 17000 + i * 350, fs, t)
    # signal = alpha * y
    # print(signal.dtype)

    # signal = get_sinusoid(18000, 400, 8, fs, 60)
    signal = cos_wave(1, 20e3, fs, 60)

    wavfile.write(r'20khz.wav', fs, signal)

    # 这个方法不知道为什么有问题
    # wf = wave.open(r'sinusoid2.wav', 'wb')
    # wf.setnchannels(1)
    # wf.setsampwidth(4)  # float32
    # wf.setframerate(fs)
    # wf.writeframes(b''.join(signal))
    # wf.close()
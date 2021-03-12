'''
将想要发射的声波生成成wav文件
'''
import numpy as np
import wave

from pyaudio import PyAudio

from realtimesys.signalgenerator import cos_wave

if __name__ == '__main__':
    # a = np.array([1, 2, 3], dtype=np.int)
    # print(b''.join(a))

    fs = 48e3
    t = 300
    A = [1, 1, 1, 1, 1, 1, 1, 1]
    alpha = 1 / sum(A)
    y = A[0] * cos_wave(1, 17000, fs, t)
    for i in range(1, 8):
        y = y + A[i] * cos_wave(1, 17000 + i * 350, fs, t)
    signal = alpha * y
    print(signal.dtype)

    wf = wave.open(r'sinusoid2.wav', 'wb')
    wf.setnchannels(1)
    wf.setsampwidth(4)  # float32
    wf.setframerate(fs)
    wf.writeframes(b''.join(signal))
    wf.close()
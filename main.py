import wave
import numpy as np

from audiotools.util import get_dtype_from_width
from dsptools.util import get_cos_IQ, get_phase

CHUNK = 2048  # audio frame length
STEP = 700


def generate_training_data(audio_file, dataset_file):
    wf = wave.open(audio_file, 'rb')
    nchannels = wf.getnchannels()  # 声道数
    fs = wf.getframerate()  # 采样率
    # 开始处理数据
    t = 0
    str_data = wf.readframes(CHUNK)
    while str_data != b'':
        # 从二进制转化为int16
        unwrapped_phase_list = []
        data = np.frombuffer(str_data, dtype=get_dtype_from_width(wf.getsampwidth()))
        data = data.reshape((-1, nchannels))
        data = data.T  # shape = (num_of_channels, CHUNK)
        # 处理数据，这里可以优化，还需要验证其正确性
        f = 17350
        for i in range(8):
            f = f + i * STEP
            I, Q = get_cos_IQ(data, f, fs)
            unwrapped_phase = get_phase(I, Q)
            unwrapped_phase_list.append(unwrapped_phase)
        # 把8个频率的合并成一个矩阵 shape = (num_of_channels, 8 * CHUNK)
        merged_u_p = np.hstack(([u_p for u_p in unwrapped_phase_list]))
        # 压缩便于保存
        flattened_m_u_p = merged_u_p.flatten()
        with open(dataset_file, 'ab') as f:
            np.savetxt(f, flattened_m_u_p.reshape(1, -1))
        # 输出时间，并读取下一段数据
        t = t + CHUNK / fs
        print(f"time:{t}s")
        str_data = wf.readframes(CHUNK)


if __name__ == '__main__':
    generate_training_data(r'1.wav', r't.txt')
    # a = np.array([[[1, 2, 3], [4, 5, 6]], [[1, 2, 3], [4, 5, 6]]])
    # print(a)
    # # x = np.hstack(([u_p for u_p in a]))
    # # print(x)
    # y = a.reshape((2, -1), order='A')
    # z = y.flatten()
    # print(z)
    # with open('t.txt', 'ab') as f:
    #     np.savetxt(f, z.reshape(1, -1))
    # d = np.loadtxt('t.txt')

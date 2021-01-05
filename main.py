import wave
import numpy as np

from audiotools.util import get_dtype_from_width
from dsptools.util import get_cos_IQ, get_phase

CHUNK = 2048  # audio frame length
STEP = 700


def main():
    file_name = r'1.wav'
    wf = wave.open(file_name, 'rb')
    nchannels = wf.getnchannels()  # 声道数
    fs = wf.getframerate()  # 采样率
    # 开始处理数据
    f = 17350
    t = 0
    str_data = wf.readframes(CHUNK)
    while str_data != b'':
        # 从二进制转化为int16
        unwrapped_phase_list = []
        data = np.frombuffer(str_data, dtype=get_dtype_from_width(wf.getsampwidth()))
        data = data.reshape((-1, nchannels))
        data = data.T  # shape = (num_of_channels, CHUNK)
        # 处理数据,这里可以优化
        for i in range(8):
            f = f + i * STEP
            I, Q = get_cos_IQ(data, f, fs)
            unwrapped_phase = get_phase(I, Q)
            unwrapped_phase_list.append(unwrapped_phase)
        # 把8个频率的合并成一个矩阵 shape = (num_of_channels, 8 * CHUNK)
        merged_u_p = np.hstack(([u_p for u_p in unwrapped_phase_list]))

        # 输出时间，并读取下一段数据
        t = t + CHUNK / fs
        print(f"time:{t}s")
        str_data = wf.readframes(CHUNK)


if __name__ == '__main__':
    # main()
    a = np.array([[[1, 2, 3], [4, 5, 6]], [[1, 2, 3], [4, 5, 6]]])
    x = np.hstack(([u_p for u_p in a]))
    print(x)
    y = a.reshape((2, -1), order='A')
    print(y)

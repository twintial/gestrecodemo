import wave
import numpy as np

from audiotools.util import get_dtype_from_width
from dsptools.filter import butter_bandpass_filter
from dsptools.util import get_cos_IQ, get_phase
import matplotlib.pyplot as plt


CHUNK = 2048  # audio frame length
STEP = 700  # 每个频率的跨度
NUM_OF_FREQ = 8  # 频率数量
DELAY_TIME = 2  # 麦克风的延迟时间
STD_THRESHOLD = 0.022  # 相位标准差阈值


def generate_training_data(audio_file, dataset_file):
    wf = wave.open(audio_file, 'rb')
    nchannels = wf.getnchannels()  # 声道数
    fs = wf.getframerate()  # 采样率
    # 开始处理数据
    t = 0
    f0 = 17350
    str_data = wf.readframes(CHUNK)
    while str_data != b'':
        # 读取下一段数据
        str_data = wf.readframes(CHUNK)
        if str_data == b'':
            break
        t = t + CHUNK / fs
        print(f"time:{t}s")
        # 由于麦克风的原因只看2s之后的
        if t < DELAY_TIME:
            continue
        # 从二进制转化为int16
        unwrapped_phase_list = []
        data = np.frombuffer(str_data, dtype=get_dtype_from_width(wf.getsampwidth()))
        data = data.reshape((-1, nchannels))
        data = data.T  # shape = (num_of_channels, CHUNK)
        # 处理数据，这里可以优化，还需要验证其正确性
        for i in range(NUM_OF_FREQ):
            fc = f0 + i * STEP
            data_filter = butter_bandpass_filter(data, fc-250, fc+250)
            I, Q = get_cos_IQ(data_filter, fc, fs)
            unwrapped_phase = get_phase(I, Q)  # 这里的展开目前没什么效果
            # plt.plot(unwrapped_phase[0])
            # plt.show()
            # 通过标准差判断是否在运动
            u_p_stds = np.std(unwrapped_phase, axis=1)
            # print(np.mean(u_p_stds))
            if np.mean(u_p_stds) > STD_THRESHOLD:
                unwrapped_phase_list.append(unwrapped_phase)
        # 把8个频率的合并成一个矩阵 shape = (num_of_channels * NUM_OF_FREQ, CHUNK)
        if len(unwrapped_phase_list) != NUM_OF_FREQ:
            continue
        merged_u_p = np.vstack(([u_p for u_p in unwrapped_phase_list]))
        # 压缩便于保存
        flattened_m_u_p = merged_u_p.flatten()
        with open(dataset_file, 'ab') as f:
            np.savetxt(f, flattened_m_u_p.reshape(1, -1))


if __name__ == '__main__':
    import os
    try:
        os.remove(r't.txt')
    except:
        pass
    generate_training_data(r'0.wav', r't.txt')
    # a = np.array([[[1, 2, 3], [4, 5, 6]], [[1, 2, 3], [4, 5, 6]]])
    # print(a)
    # # x = np.hstack(([u_p for u_p in a]))
    # # print(x)
    # y = a.reshape((2, -1), order='A')
    # z = y.flatten()
    # print(z)
    d = np.loadtxt('t.txt')
    print(d.shape)

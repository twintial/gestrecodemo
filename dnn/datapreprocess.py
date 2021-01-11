import wave
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose

from audiotools.util import get_dtype_from_width, load_audio_data
from dsptools.filter import butter_bandpass_filter, move_average_overlap_filter
from dsptools.util import get_cos_IQ, get_phase, get_cos_IQ_raw
import re
import os
import matplotlib.pyplot as plt
import keras
from sklearn.model_selection import train_test_split

CHUNK = 2048  # audio frame length
STEP = 700  # 每个频率的跨度
NUM_OF_FREQ = 8  # 频率数量
DELAY_TIME = 1  # 麦克风的延迟时间
STD_THRESHOLD = 0.022  # 相位标准差阈值


def print_history(history):
    plt.plot(history['acc'])
    plt.plot(history['val_acc'])
    plt.title('Model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()

    plt.plot(history['loss'])
    plt.plot(history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()


# 使用窗口的方法存在问题
def generate_training_data(audio_file, dataset_save_file):
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

        if data.shape[1] < CHUNK:
            continue

        # 处理数据，这里可以优化，还需要验证其正确性
        for i in range(NUM_OF_FREQ):
            fc = f0 + i * STEP
            data_filter = butter_bandpass_filter(data, fc - 250, fc + 250)
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
        with open(dataset_save_file, 'ab') as f:
            np.savetxt(f, flattened_m_u_p.reshape(1, -1))


def generate_training_data_pcm(audio_file, dataset_save_file):
    origin_data, fs = load_audio_data(audio_file, 'pcm')
    nchannels = 1  # 声道数
    fs = fs  # 采样率
    # 开始处理数据
    t = 0
    f0 = 17350
    for win in range(CHUNK, len(origin_data), CHUNK):
        # 读取下一段数据
        data = origin_data[win - CHUNK:win + CHUNK]
        t = t + CHUNK / fs
        # print(f"time:{t}s")
        # 由于麦克风的原因只看2s之后的
        if t < DELAY_TIME:
            continue
        unwrapped_phase_list = []
        data = data.reshape((-1, nchannels))
        data = data.T  # shape = (num_of_channels, 2 * CHUNK)
        if data.shape[1] < 2 * CHUNK:
            continue
        # 处理数据，这里可以优化，还需要验证其正确性
        for i in range(NUM_OF_FREQ):
            fc = f0 + i * STEP
            data_filter = butter_bandpass_filter(data, fc - 250, fc + 250)
            I, Q = get_cos_IQ(data_filter, fc, fs)
            unwrapped_phase = get_phase(I, Q)  # 这里的展开目前没什么效果
            # plt.plot(unwrapped_phase[0])
            # plt.show()
            # 通过标准差判断是否在运动
            assert unwrapped_phase.shape[1] > 1
            u_p_stds = np.std(unwrapped_phase, axis=1)
            print(fc, np.mean(u_p_stds))
            if np.mean(u_p_stds) > STD_THRESHOLD:
                unwrapped_phase_list.append(unwrapped_phase)
        # 把8个频率的合并成一个矩阵 shape = (num_of_channels * NUM_OF_FREQ, CHUNK)
        if len(unwrapped_phase_list) != NUM_OF_FREQ:
            continue
        print(f"time:{t}s")
        merged_u_p = np.vstack(([u_p for u_p in unwrapped_phase_list]))
        # 压缩便于保存
        flattened_m_u_p = merged_u_p.flatten()
        with open(dataset_save_file, 'ab') as f:
            np.savetxt(f, flattened_m_u_p.reshape(1, -1))


# 使用整个的方法
def extract_phasedata_from_audio(audio_file, phasedata_save_file, audio_type='pcm'):
    origin_data, fs = load_audio_data(audio_file, audio_type)
    nchannels = 1  # 声道数
    fs = fs  # 采样率
    data = origin_data[int(fs * DELAY_TIME):]
    data = data.reshape((-1, nchannels))
    data = data.T  # shape = (num_of_channels, all_frames)
    # 开始处理数据
    t = 0
    f0 = 17350
    unwrapped_phase_list = []
    for i in range(NUM_OF_FREQ):
        fc = f0 + i * STEP
        data_filter = butter_bandpass_filter(data, fc - 250, fc + 250)
        I_raw, Q_raw = get_cos_IQ_raw(data_filter, fc, fs)
        # 滤波+下采样
        I = move_average_overlap_filter(I_raw)
        Q = move_average_overlap_filter(Q_raw)
        # denoise
        decompositionQ = seasonal_decompose(Q.T, period=5, two_sided=False)
        trendQ = decompositionQ.trend
        decompositionI = seasonal_decompose(I.T, period=5, two_sided=False)
        trendI = decompositionI.trend

        trendQ = trendQ.T
        trendI = trendI.T

        assert trendI.shape == trendQ.shape
        if len(trendI.shape) == 1:
            trendI = trendI.reshape((1, -1))
            trendQ = trendQ.reshape((1, -1))

        trendQ = trendQ[:, 5:]
        trendI = trendI[:, 5:]

        unwrapped_phase = get_phase(trendI, trendQ)  # 这里的展开目前没什么效果
        # plt.plot(unwrapped_phase[0])
        # plt.show()
        assert unwrapped_phase.shape[1] > 1
        # 用diff，和两次diff
        unwrapped_phase_list.append(np.diff(unwrapped_phase)[:, :-1])
        # plt.plot(np.diff(unwrapped_phase).reshape(-1))
        # plt.show()
        unwrapped_phase_list.append(np.diff(np.diff(unwrapped_phase)))
    merged_u_p = np.array(unwrapped_phase_list).reshape((NUM_OF_FREQ * nchannels * 2, -1))
    print(merged_u_p.shape)
    # 压缩便于保存
    flattened_m_u_p = merged_u_p.flatten()
    # 由于长短不一，不能放在一起
    # np.savetxt(dataset_save_file, flattened_m_u_p.reshape(1, -1))
    np.savez_compressed(phasedata_save_file, phasedata=flattened_m_u_p)
    return nchannels


def phasedata_padding_labeling(phasedata_save_dir: str, dataset_save_file, nchannels, mean_len_method='auto'):
    phasedata_save_file_names = os.listdir(phasedata_save_dir)
    phasedata_list = []
    label_list = []
    mean_len = 0
    for f_names in phasedata_save_file_names:
        phasedata_save_file = os.path.join(phasedata_save_dir, f_names)
        if phasedata_save_file.endswith('.npz'):
            data = np.load(phasedata_save_file)
            phase_data = data['phasedata']
            phase_data = phase_data.reshape((NUM_OF_FREQ * nchannels * 2, -1))
            phasedata_list.append(phase_data)
            mean_len = mean_len + phase_data.shape[1] / len(phasedata_save_file_names)
            # label
            m = re.match(r'(\d+)-(\d+).npz', f_names)
            if m:
                label_list.append(int(m.group(2)))
            else:
                raise ValueError('label error')
        else:
            raise ValueError('unsupported file type')
    if mean_len_method == 'auto':
        mean_len = int(mean_len)
    else:
        mean_len = mean_len_method
    for i in range(len(phasedata_list)):
        detla_len = phasedata_list[i].shape[1] - mean_len
        if detla_len > 0:
            phasedata_list[i] = phasedata_list[i][:, detla_len:]
        elif detla_len < 0:
            left_zero_padding_len = abs(detla_len) // 2
            right_zero_padding_len = abs(detla_len) - left_zero_padding_len
            left_zero_padding = np.zeros((NUM_OF_FREQ * nchannels * 2, left_zero_padding_len))
            right_zero_padding = np.zeros((NUM_OF_FREQ * nchannels * 2, right_zero_padding_len))
            phasedata_list[i] = np.hstack((left_zero_padding, phasedata_list[i], right_zero_padding))
    phasedata_list = np.array(phasedata_list)
    # phasedata_list_flatten = phasedata_list.reshape((phasedata_list.shape[0], -1))
    label_list = np.array(label_list)

    np.savez_compressed(dataset_save_file, x=phasedata_list, y=label_list)


# 多声道可能有问题
def load_dataset(dataset_dir):
    file_names = os.listdir(dataset_dir)
    x_dataset_list = []
    y_dataset_list = []
    for file_name in file_names:
        m = re.match(r'(\d+).txt', file_name)
        if m:
            label = m.group(1)
            flattened_m_u_p = np.loadtxt(os.path.join(dataset_dir, file_name))
            merged_u_p = flattened_m_u_p.reshape((flattened_m_u_p.shape[0], NUM_OF_FREQ, -1, 1))
            print(merged_u_p.shape)
            x_dataset_list.append(merged_u_p)
            y_dataset_list.append(merged_u_p.shape[0] * [int(label)])
    x_dataset = np.concatenate(([x for x in x_dataset_list]), axis=0)
    y_dataset = np.concatenate(([y for y in y_dataset_list]), axis=0)
    print(x_dataset.shape)
    print(y_dataset.shape)
    return x_dataset, y_dataset


def load_dataset_v2(dataset_dir, num_classes):
    file_names = os.listdir(dataset_dir)
    x_train_list = []
    y_train_list = []
    x_test_list = []
    y_test_list = []
    for file_name in file_names:
        m = re.match(r'(\d+).txt', file_name)
        if m:
            label = m.group(1)
            flattened_m_u_p = np.loadtxt(os.path.join(dataset_dir, file_name))
            merged_u_p = flattened_m_u_p.reshape((flattened_m_u_p.shape[0], NUM_OF_FREQ, -1, 1))
            print(merged_u_p.shape)
            y_all = merged_u_p.shape[0] * [int(label)]
            y_all = keras.utils.to_categorical(y_all, num_classes)
            x_train_i, x_test_i, y_train_i, y_test_i = train_test_split(merged_u_p, y_all, train_size=0.8)
            x_train_list.append(x_train_i)
            x_test_list.append(x_test_i)
            y_train_list.append(y_train_i)
            y_test_list.append(y_test_i)
    x_train = np.concatenate(([x_tr for x_tr in x_train_list]), axis=0)
    x_test = np.concatenate(([x_te for x_te in x_test_list]), axis=0)
    y_train = np.concatenate(([y_tr for y_tr in y_train_list]), axis=0)
    y_test = np.concatenate(([y_te for y_te in y_test_list]), axis=0)

    shuffled_indices = np.random.permutation(x_train.shape[0])
    return x_train[shuffled_indices], x_test, y_train[shuffled_indices], y_test

# load_dataset('../dataset')

# x = []
# x.append(np.random.randint(1, 10, size=(3,4)))
# x.append(np.random.randint(1, 10, size=(3,4)))
# x.append(np.random.randint(1, 10, size=(3,4)))
# x = np.array(x)
# print(x)
# a = np.array(x).reshape((x.shape[0],-1))
# y = a.reshape((a.shape[0],3,-1))
# print(np.all(x==y))
# print(x)
# y = x.flatten()
# y = y.reshape((1,-1))
# print(y)
# print(y.reshape(x.shape))
# a = np.random.randint(1, 10, size=(3,4,5))
# print(a)
# b = a.reshape((3, -1))
# print(b)
# c = b.reshape((b.shape[0], 4, -1))
# print(c)
# print(np.all(a==c))

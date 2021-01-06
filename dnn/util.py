import wave
import numpy as np

from audiotools.util import get_dtype_from_width, load_audio_data
from dsptools.filter import butter_bandpass_filter
from dsptools.util import get_cos_IQ, get_phase
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
        data = origin_data[win-CHUNK:win+CHUNK]
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
            data_filter = butter_bandpass_filter(data, fc-250, fc+250)
            I, Q = get_cos_IQ(data_filter, fc, fs)
            unwrapped_phase = get_phase(I, Q)  # 这里的展开目前没什么效果
            # plt.plot(unwrapped_phase[0])
            # plt.show()
            # 通过标准差判断是否在运动
            assert unwrapped_phase.shape[1] > 1
            u_p_stds = np.std(unwrapped_phase, axis=1)
            # print(np.mean(u_p_stds))
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
            y_dataset_list.append(merged_u_p.shape[0]*[int(label)])
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
            y_all = merged_u_p.shape[0]*[int(label)]
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

# a = np.random.randint(1, 10, size=(3,4,5))
# print(a)
# b = a.reshape((3, -1))
# print(b)
# c = b.reshape((b.shape[0], 4, -1))
# print(c)
# print(np.all(a==c))
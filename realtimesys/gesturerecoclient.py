import socket
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from concurrent.futures import ThreadPoolExecutor
import threading
from tensorflow.keras import models
import time

from dsptools.filter import butter_bandpass_filter, butter_lowpass_filter, move_average_overlap_filter
from dsptools.util import get_cos_IQ_raw_offset, get_phase, get_cos_IQ_raw

F0 = 17000
STEP = 350
model_file = r'D:\projects\pyprojects\gestrecodemo\nn\models\mic_speaker_phase_234_5.h5'
model: models.Sequential = models.load_model(model_file)

def gesture_reco_detection_multithread(gesture_frames):

    '''
    可以做到config中去
    '''
    N_CHANNELS = 7
    DELAY_TIME = 1
    NUM_OF_FREQ = 8
    F0 = 17000
    STEP = 350  # 每个频率的跨度
    fs = 48000
    period = 10

    unwrapped_phase_list = [None] * NUM_OF_FREQ * 2

    def get_phase_and_diff(i):
        fc = F0 + i * STEP
        data_filter = butter_bandpass_filter(gesture_frames, fc - 150, fc + 150)
        I_raw, Q_raw = get_cos_IQ_raw(data_filter, fc, fs)
        # 滤波+下采样
        I = move_average_overlap_filter(I_raw)
        Q = move_average_overlap_filter(Q_raw)
        # denoise, 10可能太大了，但目前训练使用的都是10
        decompositionQ = seasonal_decompose(Q.T, period=period, two_sided=False)
        trendQ = decompositionQ.trend
        decompositionI = seasonal_decompose(I.T, period=period, two_sided=False)
        trendI = decompositionI.trend

        trendQ = trendQ.T
        trendI = trendI.T

        assert trendI.shape == trendQ.shape
        if len(trendI.shape) == 1:
            trendI = trendI.reshape((1, -1))
            trendQ = trendQ.reshape((1, -1))

        trendQ = trendQ[:, period:]
        trendI = trendI[:, period:]

        unwrapped_phase = get_phase(trendI, trendQ)  # 这里的展开目前没什么效果
        # plt.plot(unwrapped_phase[0])
        # plt.show()
        assert unwrapped_phase.shape[1] > 1
        # 用diff，和两次diff
        unwrapped_phase_list[2*i] = np.diff(unwrapped_phase)[:, :-1]
        unwrapped_phase_list[2*i+1] = (np.diff(np.diff(unwrapped_phase)))
    with ThreadPoolExecutor(max_workers=8) as pool:
        pool.map(get_phase_and_diff, [i for i in range(NUM_OF_FREQ)])

    merged_u_p = np.array(unwrapped_phase_list).reshape((NUM_OF_FREQ * N_CHANNELS * 2, -1))

    # 仿造（之后删除）
    # merged_u_p = np.tile(merged_u_p, (3,1))
    # merged_u_p = np.vstack((merged_u_p, merged_u_p[:16, :]))

    mean_len = 777  # 之后要改
    # 这里补0的策略可能要改
    detla_len = merged_u_p.shape[1] - mean_len
    if detla_len > 0:
        merged_u_p = merged_u_p[:, detla_len:]
    elif detla_len < 0:
        left_zero_padding_len = abs(detla_len) // 2
        right_zero_padding_len = abs(detla_len) - left_zero_padding_len
        left_zero_padding = np.zeros((NUM_OF_FREQ * 7 * 2, left_zero_padding_len))
        right_zero_padding = np.zeros((NUM_OF_FREQ * 7 * 2, right_zero_padding_len))
        merged_u_p = np.hstack((left_zero_padding, merged_u_p, right_zero_padding))

    y_predict = model.predict(merged_u_p.reshape((1, merged_u_p.shape[0], merged_u_p.shape[1], 1)))
    label = ['握紧', '张开', '左滑', '右滑', '上滑', '下滑', '前推', '后推', '顺时针转圈', '逆时针转圈']
    print(np.argmax(y_predict[0]))
    print(label[np.argmax(y_predict[0])])

if __name__ == '__main__':
    channels = 8
    frame_count = 2048
    frames_int = None

    offset = 0

    max_frame = 48000  # 对延迟影响很大

    # 运动检测参数
    THRESHOLD = 0.008  # 运动判断阈值0.008
    motion_start_index = -1
    motion_start_index_constant = -1  # 为了截取运动片段
    motion_stop_index = -1
    motion_start = False
    lower_than_threshold_count = 0  # 超过3次即运动停止
    higher_than_threshold_count = 0  # 超过3次即运动开始
    pre_frame = 2


    phase = [None] * max_frame
    # 画图参数
    fig, ax = plt.subplots()
    # ax.set_xlim([0, 48000])
    # ax.set_ylim([-2, 0])
    l_phase, = ax.plot(phase)
    motion_start_line = ax.axvline(0, color='r')
    motion_stop_line = ax.axvline(0, color='g')
    plt.pause(0.01)

    # socket
    address = ('127.0.0.1', 31500)
    tcp_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    tcp_socket.connect(address)
    while True:
        rdata = tcp_socket.recv(frame_count*channels*2)
        if len(rdata) == 0:
            break
        data = np.frombuffer(rdata, dtype=np.int16)
        data = data.reshape(-1, channels).T
        # assert data.shape[1] == frame_count
        # frames_int要定时清空
        frames_int = data if frames_int is None else np.hstack((frames_int, data))
        if frames_int.shape[1] > max_frame * 2:
            frames_int = frames_int[:, frame_count:]
            # print(frames_int.shape)
        if frames_int.shape[1] > 3 * frame_count:
            # 前后都多拿一个CHUNK
            data_segment = frames_int[:, -3 * frame_count:]
            # 运动检测+画图，只取一个freq
            # 可以改进，这里最后只用了一个mic，可以不用都处理
            fc = F0
            data_filter = butter_bandpass_filter(data_segment, fc - 150, fc + 150)
            I_raw, Q_raw = get_cos_IQ_raw_offset(data_filter, fc, offset)
            I = butter_lowpass_filter(I_raw, 200)
            Q = butter_lowpass_filter(Q_raw, 200)
            I = I[:, frame_count:frame_count * 2]
            Q = Q[:, frame_count:frame_count * 2]
            unwrapped_phase = get_phase(I, Q)
            phase = phase[frame_count:] + list(unwrapped_phase[0])

            # 运动判断
            std = np.std(unwrapped_phase[0])
            # print(std)
            # 为了截取运动部分
            if motion_start:
                motion_start_index_constant -= frame_count
            if motion_start_index > 0:
                motion_start_index -= frame_count
            if motion_stop_index > 0:
                motion_stop_index -= frame_count
            # 连续三次大于/小于阈值
            if motion_start:
                if std < THRESHOLD:
                    lower_than_threshold_count += 1
                    if lower_than_threshold_count >= 4:
                        # 运动开始，在前4CHUNK阈值已经低于，另外减去pre_frame*CHUNK的提前量(pre_frame大于lower_than_threshold_count则多出来的部分无法取到)
                        motion_stop_index = max_frame - frame_count * (lower_than_threshold_count - pre_frame)
                        motion_start = False
                        lower_than_threshold_count = 0
                        # 运动停止，手势判断
                        gesture_frames_len = motion_stop_index - motion_start_index_constant
                        gesture_frames = frames_int[:, -gesture_frames_len:]
                        threading.Thread(target=gesture_reco_detection_multithread, args=(gesture_frames[:7],)).start()
                else:
                    lower_than_threshold_count = 0
            else:
                if std > THRESHOLD:
                    higher_than_threshold_count += 1
                    if higher_than_threshold_count >= 4:
                        # 运动开始，在前4CHUNK阈值已经超过，另外减去pre_frame*CHUNK的提前量
                        motion_start_index = max_frame - frame_count * (higher_than_threshold_count + pre_frame)
                        motion_start_index_constant = motion_start_index
                        motion_start = True
                        higher_than_threshold_count = 0
                else:
                    higher_than_threshold_count = 0

            # 画图
            motion_start_line.set_xdata(motion_start_index)
            motion_stop_line.set_xdata(motion_stop_index)
            l_phase.set_ydata(phase)
            ax.relim()
            ax.autoscale()
            ax.figure.canvas.draw()
            plt.pause(0.001)

            offset += frame_count
            # # 不一定好，暂时先这样
            # if offset > max_frame:
            #     offset = 0

import socket
import numpy as np
import matplotlib.pyplot as plt

from dsptools.filter import butter_bandpass_filter, butter_lowpass_filter
from dsptools.util import get_cos_IQ_raw_offset, get_phase
F0 = 17000
STEP = 350

if __name__ == '__main__':
    channels = 2
    frame_count = 2048
    frames_int = None

    offset = 0

    max_frame = 48000 * 2  # 对延迟影响很大

    # 运动检测参数
    THRESHOLD = 0.008  # 运动判断阈值
    motion_start_index = -1
    motion_stop_index = -1
    motion_start = False
    lower_than_threshold_count = 0  # 超过3次即运动停止
    higher_than_threshold_count = 0  # 超过3次即运动开始
    pre_frame = 2
    # 画图参数
    fig, ax = plt.subplots()
    # ax.set_xlim([0, 48000])
    # ax.set_ylim([-2, 0])
    phase = [None] * max_frame
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
        if frames_int.shape[1] > max_frame * 3:
            frames_int = frames_int[:, frame_count:]
            # print(frames_int.shape)
        if frames_int.shape[1] > 3 * frame_count:
            # 前后都多拿一个CHUNK
            data_segment = frames_int[:, -3 * frame_count:]
            # 运动检测+画图，只取一个freq
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
                else:
                    lower_than_threshold_count = 0
            else:
                if std > THRESHOLD:
                    higher_than_threshold_count += 1
                    if higher_than_threshold_count >= 4:
                        # 运动开始，在前4CHUNK阈值已经超过，另外减去pre_frame*CHUNK的提前量
                        motion_start_index = max_frame - frame_count * (higher_than_threshold_count + pre_frame)
                        motion_start = True
                        higher_than_threshold_count = 0
                else:
                    higher_than_threshold_count = 0

            motion_start_line.set_xdata(motion_start_index)
            motion_stop_line.set_xdata(motion_stop_index)

            # 画图
            l_phase.set_ydata(phase)
            ax.relim()
            ax.autoscale()
            ax.figure.canvas.draw()
            plt.pause(0.001)
            offset += frame_count
            # 不一定好，暂时先这样
            if offset > max_frame:
                offset = 0

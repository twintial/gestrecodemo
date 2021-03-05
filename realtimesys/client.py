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

    max_frame = 48000  # 对延迟影响很大

    fig, ax = plt.subplots()
    phase = [None] * max_frame
    l_phase, = ax.plot(phase)
    plt.pause(0.01)

    address = ('127.0.0.1', 31500)
    tcp_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    tcp_socket.connect(address)
    while True:
        rdata = tcp_socket.recv(frame_count*channels*2)
        if (len(rdata) == 0):
            break
        data = np.frombuffer(rdata, dtype=np.int16)
        data = data.reshape(-1, channels).T
        assert data.shape[1] == frame_count
        frames_int = data if frames_int is None else np.hstack((frames_int, data))
        if frames_int.shape[1] > 3 * frame_count:
            # 前后都多拿一个CHUNK
            data_segment = frames_int[:, -3 * frame_count:]
            for i in range(1):
                fc = F0 + i * STEP
                data_filter = butter_bandpass_filter(data_segment, fc - 150, fc + 150)
                I_raw, Q_raw = get_cos_IQ_raw_offset(data_filter, fc, offset)
                I = butter_lowpass_filter(I_raw, 200)
                Q = butter_lowpass_filter(Q_raw, 200)
                I = I[:, frame_count:frame_count * 2]
                Q = Q[:, frame_count:frame_count * 2]
                unwrapped_phase = get_phase(I, Q)
                # 改成实时
                # print(unwrapped_phase.shape)
                # u_ps = unwrapped_phase if u_ps is None else np.hstack((u_ps, unwrapped_phase))
                phase = phase[frame_count:] + list(unwrapped_phase[0])
                l_phase.set_ydata(phase)
                ax.relim()
                ax.autoscale()
                ax.figure.canvas.draw()
                # plt.draw()
                plt.pause(0.001)
        offset += frame_count
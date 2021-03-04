import wave
import numpy as np

import pyaudio
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import QLCDNumber

from dsptools.filter import butter_bandpass_filter, butter_lowpass_filter
from dsptools.util import get_phase, get_cos_IQ_raw, get_cos_IQ_raw_offset
from nn.preprocess import F0, STEP


class Record:
    def __init__(self):
        self.wav_file = None

        self.signal = None
        self.cursor = 0

        self.t = 0

        self.frames = None
        self.u_ps = None
        self.chunk = 2048
        self.fs = 48000
        self.channels = 1
        self.format = pyaudio.paInt16  # int24在数据处理上不方便,改为int16,注意和input_callback中的np.int对于
        self._recording = False
        self._playing = False
        self.save_path = None
        self.input_device_index = None
        self.output_device_index = None

        self.p = pyaudio.PyAudio()
        self.record_stream = None
        self.play_stream = None

        self.offset = 0

        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(1, 1, 1)

    def set_param(self, input_id, output_id, channels, save_path):
        self.input_device_index = input_id
        self.output_device_index = output_id
        self.channels = channels
        self.save_path = save_path

    def input_callback(self, in_data, frame_count, time_info, status_flags):
        data = np.frombuffer(in_data, dtype=np.int16)
        # 多个声道，一个隔一个是一个声道
        data = data.reshape(-1, self.channels).T
        self.frames = data if self.frames is None else np.hstack((self.frames, data))
        if self.frames.shape[1] > 3 * frame_count:
            # 前后都多拿一个CHUNK
            data_segment = self.frames[:, -3*frame_count:]
            for i in range(1):
                fc = F0 + i * STEP
                data_filter = butter_bandpass_filter(data_segment, fc - 150, fc + 150)
                I_raw, Q_raw = get_cos_IQ_raw_offset(data_filter, fc, self.offset)
                I = butter_lowpass_filter(I_raw, 200)
                Q = butter_lowpass_filter(Q_raw, 200)
                I = I[:, frame_count:frame_count * 2]
                Q = Q[:, frame_count:frame_count * 2]
                unwrapped_phase = get_phase(I, Q)
                # 改成实时
                # print(unwrapped_phase.shape)
                self.u_ps = unwrapped_phase if self.u_ps is None else np.hstack((self.u_ps, unwrapped_phase))
                self.ax.plot(self.u_ps[0], c='m')
        self.offset += frame_count

        # 这里做处理
        self.t = self.t + frame_count / self.fs
        # self.time.display(self.t)
        return in_data, pyaudio.paContinue

    def output_callback(self, in_data, frame_count, time_info, status_flags):
        out_data = self.signal[self.cursor:self.cursor + frame_count]
        self.cursor = self.cursor + frame_count
        if self.cursor + frame_count > len(self.signal):
            self.cursor = 0
        return out_data, pyaudio.paContinue

    def wavfile_output_callback(self, in_data, frame_count, time_info, status_flags):
        out_data = self.wav_file.readframes(frame_count)
        return out_data, pyaudio.paContinue

    def play_signal(self, signal):
        # signal可能是一个wav文件名称或者是一串信号
        self._playing = True
        self.signal = signal
        if type(signal) == str:
            self.wav_file = wave.open(signal, "rb")
            self.play_stream = self.p.open(format=self.p.get_format_from_width(self.wav_file.getsampwidth()),
                                           channels=self.wav_file.getnchannels(),
                                           rate=self.wav_file.getframerate(),
                                           output=True,
                                           output_device_index=self.output_device_index,
                                           frames_per_buffer=self.chunk,
                                           stream_callback=self.wavfile_output_callback
                                           )
        else:
            # p = pyaudio.PyAudio()
            self.play_stream = self.p.open(format=pyaudio.paFloat32,
                                           channels=1,
                                           rate=self.fs,
                                           output=True,
                                           output_device_index=self.output_device_index,
                                           frames_per_buffer=self.chunk,
                                           stream_callback=self.output_callback)
        self.play_stream.start_stream()

    def record(self):
        self.record_stream = self.p.open(format=self.format,
                                         channels=self.channels,
                                         rate=self.fs,
                                         input=True,
                                         input_device_index=self.input_device_index,
                                         frames_per_buffer=self.chunk,
                                         stream_callback=self.input_callback)
        self.record_stream.start_stream()

    def play_and_record(self, signal):
        self.play_signal(signal)
        self.record()

    def stop(self):
        self.record_stream.stop_stream()
        self.record_stream.close()
        self.play_stream.stop_stream()
        self.play_stream.close()
        if type(self.signal) == str:
            self.wav_file.close()
        # self.p.terminate()

        self.save()
        self.frames = None
        self.t = 0

    def save(self):
        wf = wave.open(self.save_path, 'wb')
        wf.setnchannels(self.channels)
        wf.setsampwidth(self.p.get_sample_size(self.format))
        wf.setframerate(self.fs)
        wf.writeframes(b''.join(self.frames))
        wf.close()

import time

from realtimesys.recordtool import Record

from realtimesys.deviceinfo import get_drivers, get_devices

import os

from realtimesys.signalgenerator import cos_wave

if __name__ == '__main__':
    drivers = get_drivers()
    print([driver['name'] for driver in drivers])
    devices_name = get_devices(drivers, 'Input',driver=drivers[0]['name'])
    print(devices_name)
    devices_name = get_devices(drivers, 'Output',driver=drivers[0]['name'])
    print(devices_name)

    # 音频可以生成后保存
    t = 300
    # A = [1, 1, 1, 1, 1, 1, 1, 1]
    # alpha = 1 / sum(A)
    # y = A[0] * cos_wave(1, 17000, 48e3, t)
    # for i in range(1, 8):
    #     y = y + A[i] * cos_wave(1, 17000 + i * 350, 48e3, t)
    # signal = alpha * y

    address = ('127.0.0.1', 31500)
    print('wait for connect')
    recorder = Record(address)
    # 用麦克风阵列的音箱，超过20s就会有异响 ump-8:12 4 8/电脑:0 2 2
    recorder.set_param(12, 4, 8, os.path.join(os.getcwd(), '324.wav'))
    recorder.play_and_record('sinusoid2.wav')
    # recorder.play_and_record(None)
    time.sleep(t)
    recorder.stop()

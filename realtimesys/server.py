import time

from realtimesys.recordtool import Record

from realtimesys.deviceinfo import get_drivers, get_devices

import os

from realtimesys.sinsound import cos_wave

if __name__ == '__main__':
    drivers = get_drivers()
    print([driver['name'] for driver in drivers])
    devices_name = get_devices(drivers, 'Input',driver=drivers[0]['name'])
    print(devices_name)
    devices_name = get_devices(drivers, 'Output',driver=drivers[0]['name'])
    print(devices_name)

    # 音频可以生成后保存
    t = 20
    A = [1, 1, 1, 1, 1, 1, 1, 1]
    alpha = 1 / sum(A)
    y = A[0] * cos_wave(1, 17000, 48e3, t)
    for i in range(1, 8):
        y = y + A[i] * cos_wave(1, 17000 + i * 350, 48e3, t)
    signal = alpha * y

    address = ('127.0.0.1', 31500)
    print('wait for connect')
    recorder = Record(address)
    recorder.set_param(12, 3, 8, os.path.join(os.getcwd(), 'test.wav'))
    recorder.play_and_record(signal)
    time.sleep(t)
    recorder.stop()

from arlpy import bf, utils
import numpy as np


def ump_8_beamform(data, fs, angel):
    assert data.shape[0] == 7
    r = 0.043  # 43mm
    theta = np.pi / 3
    # 7个麦克风
    pos = [[0, 0, 0]]
    for i in range(6):
        pos.append([r * np.cos(theta * i), r * np.sin(theta * i), 0])
    pos = np.array(pos)
    # plt.plot(pos[:, 0], pos[:, 1],  '.')
    # plt.show()
    c = 343
    sd = bf.steering_plane_wave(pos, c, angel)

    y = bf.delay_and_sum(data, fs, sd)
    return y

def beamform_real(data, sd, fs=48000):
    """
    这种方法分辨率只有15度，越靠近15的倍数度数效果越好
    :param data:
    :param sd:
    :param fs:
    :return:
    """
    tailor_frames_nums = np.round(sd[0] * fs) + 6
    tailor_frames_nums = tailor_frames_nums.astype(np.int16)
    print(tailor_frames_nums)
    # 防止 max_tailor_frames - tailor_frames_num == 0
    max_tailor_frames = np.max(tailor_frames_nums) + 1
    beamformed_data = None
    for i, tailor_frames_num in enumerate(tailor_frames_nums):
        if beamformed_data is None:
            # copy()因为np.frombuffer得到的数组是read only的，而且赋值好像存在内存共享的情况，有待考证
            beamformed_data = data[i, tailor_frames_num:tailor_frames_num - max_tailor_frames].copy()
        else:
            beamformed_data += data[i, tailor_frames_num:tailor_frames_num - max_tailor_frames]
    beamformed_data = beamformed_data / len(tailor_frames_nums)
    return beamformed_data

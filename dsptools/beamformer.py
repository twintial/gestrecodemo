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

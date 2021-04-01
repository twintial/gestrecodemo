from nn.preprocess import generate_training_data_pcm, extract_phasedata_from_audio, phasedata_padding_labeling, \
    extract_magndata_from_audio, extract_magndata_from_audio_special_for_onemic, extract_magndata_from_beamformed_audio, \
    extract_phasedata_from_beamformed_audio, extract_phasedata_from_audio_special_for_onemic
import numpy as np
import os
import re
from pathconfig import *

def test():
    import os
    try:
        os.remove(r't.txt')
    except:
        pass
    # generate_training_data_pcm(r'D:\projects\pyprojects\andriodfaceidproject\temp\word1\shenjunjie\148.pcm', r't.txt')
    extract_phasedata_from_audio(r'D:\projects\pyprojects\andriodfaceidproject\temp\word1\shenjunjie\148.pcm', r't.txt')
    # a = np.array([[[1, 2, 3], [4, 5, 6]], [[1, 2, 3], [4, 5, 6]]])
    # print(a)
    # # x = np.hstack(([u_p for u_p in a]))
    # # print(x)
    # y = a.reshape((2, -1), order='A')
    # z = y.flatten()
    # print(z)
    d = np.loadtxt('t.txt')
    print(d.shape)

def generate_training_dataset(audio_dir, rawdata_dir, offset, audio_type):
    audio_file_names = os.listdir(audio_dir)
    for audio_file_name in audio_file_names:
        m = re.match(r'(\d*)\.'+audio_type, audio_file_name)
        if m:
            code = int(m.group(1))
            label = code // 10
            audio_file = os.path.join(audio_dir, audio_file_name)
            extract_phasedata_from_audio(audio_file, os.path.join(rawdata_dir, f'{code + offset}-{label}'), audio_type=audio_type, mic_array=True)
            # extract_magndata_from_audio(audio_file, os.path.join(rawdata_dir, f'{code + offset}-{label}'), audio_type=audio_type, mic_array=True)
            # extract_magndata_from_beamformed_audio(audio_file, os.path.join(rawdata_dir, f'{code + offset}-{label}'), audio_type=audio_type, mic_array=True)
            # extract_phasedata_from_beamformed_audio(audio_file, os.path.join(rawdata_dir, f'{code + offset}-{label}'), audio_type=audio_type, mic_array=True)
            # extract_magndata_from_audio_special_for_onemic(audio_file, os.path.join(rawdata_dir, f'{code + offset}-{label}'), audio_type=audio_type, mic_array=True)
            # extract_phasedata_from_audio_special_for_onemic(audio_file, os.path.join(rawdata_dir, f'{code + offset}-{label}'), audio_type=audio_type, mic_array=True)

def run():
    # 生成训练+测试数据, 10是因为每个动作做10次
    for index, audio_dir in enumerate(TRAINING_AUDIO_DIRS):
        generate_training_dataset(audio_dir, TRAINING_RAWDATA_DIR, index * 10, 'wav')
    mean_len = phasedata_padding_labeling(TRAINING_RAWDATA_DIR, TRAINING_PADDING_FILE, nchannels=7, mean_len_method=1400)
    # 生成新数据
    for index, audio_dir in enumerate(TEST_AUDIO_DIRS):
        generate_training_dataset(audio_dir, TEST_RAWDATA_DIR, index * 10, 'wav')
    phasedata_padding_labeling(TEST_RAWDATA_DIR, TEST_PADDING_FILE, nchannels=7, mean_len_method=1400)

def customized():
    import matplotlib.pyplot as plt
    dirs = [r'D:\实验数据\2021\毕设\siamese\zq\10']
    for index, audio_dir in enumerate(dirs):
        audio_file_names = os.listdir(audio_dir)
        for audio_file_name in audio_file_names:
            m = re.match(r'(\d*)\.' + 'wav', audio_file_name)
            if m:
                code = int(m.group(1))
                audio_file = os.path.join(audio_dir, audio_file_name)
                extract_phasedata_from_audio(audio_file, os.path.join(r'D:\实验数据\2021\毕设\siamese\zq\raw', audio_file_name),
                                             audio_type='wav', mic_array=True)
    # 要自己添加label
    phasedata_save_dir = r'D:\实验数据\2021\毕设\siamese\zq\raw'
    dataset_save_file = r'D:\实验数据\2021\毕设\siamese\zq\padding\dataset.npz'
    NUM_OF_FREQ = 8
    nchannels = 7
    label = 10
    mean_len = 1400

    phasedata_save_file_names = os.listdir(phasedata_save_dir)
    phasedata_list = []
    label_list = []
    for f_names in phasedata_save_file_names:
        phasedata_save_file = os.path.join(phasedata_save_dir, f_names)
        if phasedata_save_file.endswith('.npz'):
            data = np.load(phasedata_save_file)
            phase_data = data['phasedata']
            phase_data = phase_data.reshape((NUM_OF_FREQ * nchannels * 1, -1))
            phasedata_list.append(phase_data)
            # mean_len = mean_len + phase_data.shape[1] / len(phasedata_save_file_names)
            # label
            label_list.append(label)
        else:
            raise ValueError('unsupported file type')
    for i in range(len(phasedata_list)):

        # plt.figure()
        # plt.plot(phasedata_list[i][0])

        detla_len = phasedata_list[i].shape[1] - mean_len
        if detla_len > 0:
            phasedata_list[i] = phasedata_list[i][:, int((detla_len+1)/2):-int(detla_len/2)]
            assert phasedata_list[i].shape[1] == mean_len
        elif detla_len < 0:
            left_zero_padding_len = abs(detla_len) // 2
            right_zero_padding_len = abs(detla_len) - left_zero_padding_len
            left_zero_padding = np.zeros((NUM_OF_FREQ * nchannels * 1, left_zero_padding_len))
            right_zero_padding = np.zeros((NUM_OF_FREQ * nchannels * 1, right_zero_padding_len))
            phasedata_list[i] = np.hstack((left_zero_padding, phasedata_list[i], right_zero_padding))
        # plt.figure()
        # plt.plot(phasedata_list[i][0])
        # plt.show()
    phasedata_list = np.array(phasedata_list)
    label_list = np.array(label_list)

    np.savez_compressed(dataset_save_file, x=phasedata_list, y=label_list)

if __name__ == '__main__':
    run()
    # customized()
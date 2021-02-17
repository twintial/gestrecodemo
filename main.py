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

if __name__ == '__main__':
    # test()
    # generate_training_dataset(r'D:\projects\pyprojects\andriodfaceidproject\temp\gesture5\shenjunjie', r'data\gesture\raw', 0)
    # generate_training_dataset(r'D:\projects\pyprojects\andriodfaceidproject\temp\gesture2\shenjunjie', r'data\gesture\raw', 10)
    # generate_training_dataset(r'D:\projects\pyprojects\andriodfaceidproject\temp\gesture3\shenjunjie', r'data\gesture\raw', 20)
    # generate_training_dataset(r'D:\projects\pyprojects\andriodfaceidproject\temp\gesture4\shenjunjie', r'data\gesture\raw', 30)
    # generate_training_dataset(r'D:\projects\pyprojects\andriodfaceidproject\temp\gesture1\shenjunjie',
    #                           r'data/gesture/valraw', 0)

    for index, audio_dir in enumerate(TRAINING_AUDIO_DIRS):
        generate_training_dataset(audio_dir, TRAINING_RAWDATA_DIR, index * 10, 'wav')
    phasedata_padding_labeling(TRAINING_RAWDATA_DIR, TRAINING_PADDING_FILE, nchannels=7)


    for index, audio_dir in enumerate(TEST_AUDIO_DIRS):
        generate_training_dataset(audio_dir, TEST_RAWDATA_DIR, index * 10, 'wav')
    phasedata_padding_labeling(TEST_RAWDATA_DIR, TEST_PADDING_FILE, nchannels=7)

    # d = np.loadtxt('dataset/whole/14.txt')
    # print(d.shape)

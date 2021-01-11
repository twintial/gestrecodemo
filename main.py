from dnn.datapreprocess import generate_training_data_pcm, extract_phasedata_from_audio, phasedata_padding_labeling
import numpy as np
import os
import re


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

def generate_training_dataset(audio_dir, rawdata_dir, offset):
    audio_file_names = os.listdir(audio_dir)
    for audio_file_name in audio_file_names:
        m = re.match(r'(\d*)\.pcm', audio_file_name)
        if m:
            code = int(m.group(1))
            label = code // 10
            audio_file = os.path.join(audio_dir, audio_file_name)
            extract_phasedata_from_audio(audio_file, os.path.join(rawdata_dir, f'{code + offset}-{label}'))



if __name__ == '__main__':
    # test()
    # generate_training_dataset(r'D:\projects\pyprojects\andriodfaceidproject\temp\gesture1\shenjunjie', r'data\gesture\origin', 0)
    # generate_training_dataset(r'D:\projects\pyprojects\andriodfaceidproject\temp\gesture2\shenjunjie', r'data\gesture\origin', 10)
    # generate_training_dataset(r'D:\projects\pyprojects\andriodfaceidproject\temp\gesture3\shenjunjie', r'data\gesture\origin', 20)
    # generate_training_dataset(r'D:\projects\pyprojects\andriodfaceidproject\temp\gesture4\shenjunjie', r'data\gesture\origin', 30)
    generate_training_dataset(r'D:\projects\pyprojects\andriodfaceidproject\temp\gesture5\shenjunjie',
                              r'data/gesture/valraw', 0)
    phasedata_padding_labeling('data/gesture/valraw', 'data/gesture/valpadding/dataset', 1)
    # d = np.loadtxt('dataset/whole/14.txt')
    # print(d.shape)

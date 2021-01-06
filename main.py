from dnn.util import generate_training_data_pcm
import numpy as np
import os
import re


def test():
    import os
    try:
        os.remove(r't.txt')
    except:
        pass
    generate_training_data_pcm(r'D:\projects\pyprojects\andriodfaceidproject\temp\word1\shenjunjie\148.pcm', r't.txt')
    # a = np.array([[[1, 2, 3], [4, 5, 6]], [[1, 2, 3], [4, 5, 6]]])
    # print(a)
    # # x = np.hstack(([u_p for u_p in a]))
    # # print(x)
    # y = a.reshape((2, -1), order='A')
    # z = y.flatten()
    # print(z)
    d = np.loadtxt('t.txt')
    print(d.shape)

def generate_training_dataset():
    audio_dir = r'D:\projects\pyprojects\andriodfaceidproject\temp\word1\shenjunjie'
    audio_file_names = os.listdir(audio_dir)
    for audio_file_name in audio_file_names:
        m = re.match(r'(\d*)\.pcm', audio_file_name)
        if m:
            code = int(m.group(1))
            label = code // 10
            audio_file = os.path.join(audio_dir, audio_file_name)
            generate_training_data_pcm(audio_file, os.path.join('dataset', f'{str(label)}.txt'))



if __name__ == '__main__':
    # test()
    generate_training_dataset()
    d = np.loadtxt('dataset/14.txt')
    print(d.shape)

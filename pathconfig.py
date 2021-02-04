import os

EXPERIMENT_NAME = r'distant_mic_magn'  # only need to change this val

PROJECT_DIR = os.path.abspath(os.path.dirname(__file__))

BASE_DIR = os.path.join(PROJECT_DIR, r'data')

# train
TRAINING_AUDIO_DIRS = [
    # r'D:\projects\pyprojects\soundphase\gest\sjj\gesture1',
    # r'D:\projects\pyprojects\soundphase\gest\sjj\gesture2',
    # r'D:\projects\pyprojects\soundphase\gest\sjj\gesture3'
    # r'D:\实验数据\2021\毕设\micarray\sjj\gesture1',
    # r'D:\实验数据\2021\毕设\micarray\sjj\gesture2',
    # r'D:\实验数据\2021\毕设\micarray\sjj\gesture3'
    r'D:\实验数据\2021\毕设\micarrayspeaker\sjj\gesture2',
    r'D:\实验数据\2021\毕设\micarrayspeaker\sjj\gesture3',
    r'D:\实验数据\2021\毕设\micarrayspeaker\sjj\gesture4'
    # r'D:\实验数据\2021\毕设\distant_mic\sjj\gesture1',

]
TRAINING_DATA_DIR = os.path.join(BASE_DIR, EXPERIMENT_NAME, r'train')
TRAINING_RAWDATA_DIR = os.path.join(TRAINING_DATA_DIR, r'raw')
TRAINING_PADDING_FILE = os.path.join(TRAINING_DATA_DIR, r'padding', r'dataset.npz')  # npz ext
TRAINING_SPLIT_FILE = os.path.join(TRAINING_DATA_DIR, r'split', r'splitdata.npz')

# test
TEST_AUDIO_DIRS = [
    # r'D:\projects\pyprojects\soundphase\gest\sjj\gesture4',
    r'D:\实验数据\2021\毕设\micarrayspeaker\sjj\gesture5'
    # r'D:\实验数据\2021\毕设\distant_mic\sjj\gesture2'
]
TEST_DATA_DIR = os.path.join(BASE_DIR, EXPERIMENT_NAME, r'test')
TEST_RAWDATA_DIR = os.path.join(TEST_DATA_DIR, r'raw')
TEST_PADDING_FILE = os.path.join(TEST_DATA_DIR, r'padding', r'dataset.npz')  # npz ext


if __name__ == '__main__':
    if not os.path.exists(os.path.join(BASE_DIR, EXPERIMENT_NAME)):
        os.mkdir(os.path.join(BASE_DIR, EXPERIMENT_NAME))
        os.mkdir(TRAINING_DATA_DIR)
        os.mkdir(TRAINING_RAWDATA_DIR)
        os.mkdir(os.path.join(TRAINING_DATA_DIR, r'padding'))
        os.mkdir(os.path.join(TRAINING_DATA_DIR, r'split'))
        os.mkdir(TEST_DATA_DIR)
        os.mkdir(TEST_RAWDATA_DIR)
        os.mkdir(os.path.join(TEST_DATA_DIR, r'padding'))
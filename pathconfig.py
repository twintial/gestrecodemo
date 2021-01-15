import os

EXPERIMENT_NAME = r'ngesturemagn'  # only need to change this val

PROJECT_DIR = os.path.abspath(os.path.dirname(__file__))

BASE_DIR = os.path.join(PROJECT_DIR, r'data')

# train
TRAINING_AUDIO_DIRS = [
    r'D:\projects\pyprojects\soundphase\gest\sjj\gesture1',
    r'D:\projects\pyprojects\soundphase\gest\sjj\gesture2',
    r'D:\projects\pyprojects\soundphase\gest\sjj\gesture3'
]
TRAINING_DATA_DIR = os.path.join(BASE_DIR, EXPERIMENT_NAME, r'train')
TRAINING_RAWDATA_DIR = os.path.join(TRAINING_DATA_DIR, r'raw')
TRAINING_PADDING_FILE = os.path.join(TRAINING_DATA_DIR, r'padding', r'dataset.npz')  # npz ext
TRAINING_SPLIT_FILE = os.path.join(TRAINING_DATA_DIR, r'split', r'splitdata.npz')

# test
TEST_AUDIO_DIRS = [
    r'D:\projects\pyprojects\soundphase\gest\sjj\gesture4',
]
TEST_DATA_DIR = os.path.join(BASE_DIR, EXPERIMENT_NAME, r'test')
TEST_RAWDATA_DIR = os.path.join(TEST_DATA_DIR, r'raw')
TEST_PADDING_FILE = os.path.join(TEST_DATA_DIR, r'padding', r'dataset.npz')  # npz ext
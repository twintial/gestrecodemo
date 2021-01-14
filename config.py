import os


AUDIO_DIR = r'D:\projects\pyprojects\soundphase\gest\gesture1\sjj'

EXPERIMENT_NAME = r'ngesture'  # only need to change this val

PROJECT_DIR = os.path.abspath(os.path.dirname(__file__))

BASE_DIR = os.path.join(PROJECT_DIR, r'data')

# train
TRINGING_DATA_DIR = os.path.join(BASE_DIR, EXPERIMENT_NAME, r'train')
TRINGING_RAWDATA_DIR = os.path.join(TRINGING_DATA_DIR, r'raw')
TRINGING_PADDING_FILE = os.path.join(TRINGING_DATA_DIR, r'padding', r'dataset.npz')  # npz ext
TRINGING_SPLIT_FILE = os.path.join(TRINGING_DATA_DIR, r'split', r'splitdata.npz')

# test
TEST_DATA_DIR = os.path.join(BASE_DIR, EXPERIMENT_NAME, r'test')
TEST_RAWDATA_DIR = os.path.join(TEST_DATA_DIR, r'raw')
TEST_PADDING_FILE = os.path.join(TEST_DATA_DIR, r'padding', r'dataset.npz')  # npz ext
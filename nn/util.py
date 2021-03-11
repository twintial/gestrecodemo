import random

import numpy as np

from pathconfig import TRAINING_PADDING_FILE, TRAINING_SPLIT_FILE


def dataset_split(x: np.ndarray, y: np.ndarray, ratio=0.8):
    x_train = []
    y_train = []
    x_test = []
    y_test = []
    sorted_index = np.argsort(y)
    x_sorted = x[sorted_index]
    y_sorted = y[sorted_index]
    cursor = 0
    for cate in range(len(np.unique(y_sorted))):
        sample_num = np.sum(y_sorted == cate)
        random_index = np.random.permutation(sample_num)
        random_index = random_index + cursor  # 这里写的不好
        x_random = x_sorted[random_index]
        y_random = y_sorted[random_index]

        x_random = list(x_random)
        y_random = list(y_random)

        x_train = x_train + x_random[:int(sample_num * ratio)]
        x_test = x_test + x_random[int(sample_num * ratio):]

        y_train = y_train + y_random[:int(sample_num * ratio)]
        y_test = y_test + y_random[int(sample_num * ratio):]

        cursor = cursor + sample_num
    x_train = np.array(x_train)
    x_test = np.array(x_test)
    y_train = np.array(y_train)
    y_test = np.array(y_test)

    train_random = np.random.permutation(x_train.shape[0])
    test_random = np.random.permutation(x_test.shape[0])

    return x_train[train_random], x_test[test_random], y_train[train_random], y_test[test_random]


def dataset_split_two_dataset(random_index, train_random, test_random, x: np.ndarray, y: np.ndarray, ratio=0.8):
    x_train = []
    y_train = []
    x_test = []
    y_test = []
    sorted_index = np.argsort(y)
    x_sorted = x[sorted_index]
    y_sorted = y[sorted_index]
    cursor = 0
    for cate in range(len(np.unique(y_sorted))):
        sample_num = np.sum(y_sorted == cate)
        index = random_index + cursor
        x_random = x_sorted[index]
        y_random = y_sorted[index]

        x_random = list(x_random)
        y_random = list(y_random)

        x_train = x_train + x_random[:int(sample_num * ratio)]
        x_test = x_test + x_random[int(sample_num * ratio):]

        y_train = y_train + y_random[:int(sample_num * ratio)]
        y_test = y_test + y_random[int(sample_num * ratio):]

        cursor = cursor + sample_num
    x_train = np.array(x_train)
    x_test = np.array(x_test)
    y_train = np.array(y_train)
    y_test = np.array(y_test)

    return x_train[train_random], x_test[test_random], y_train[train_random], y_test[test_random]


def data_split_and_save(rawdata_path, splitdata_path):
    dataset = np.load(rawdata_path)
    x = dataset['x']
    x = x.reshape((x.shape[0], x.shape[1], x.shape[2], 1))
    print(x.shape)
    y = dataset['y']
    x_train, x_test, y_train, y_test = dataset_split(x, y, ratio=0.8)  # 最好保存一下
    np.savez_compressed(splitdata_path,
                        x_train=x_train, x_test=x_test, y_train=y_train, y_test=y_test)
    return x_train, x_test, y_train, y_test


def pair_data_split_and_save(rawdata_path, splitdata_path):
    dataset = np.load(rawdata_path)
    x = dataset['x']
    x = x.reshape((x.shape[0], x.shape[1], x.shape[2], 1))
    print(x.shape)
    y = dataset['y']
    x_train, x_test, y_train, y_test = dataset_split(x, y, ratio=0.8)
    num_classes = len(np.unique(y))

    def create_pairs(x, d_i):
        '''Positive and negative pair creation.
        Alternates between positive and negative pairs.
        '''
        pairs = []
        labels = []
        n = min([len(d_i[d]) for d in range(num_classes)]) - 1
        for d in range(num_classes):
            for i in range(n):
                z1, z2 = d_i[d][i], d_i[d][i + 1]
                pairs += [[x[z1], x[z2]]]
                inc = random.randrange(1, num_classes)
                dn = (d + inc) % num_classes
                z1, z2 = d_i[d][i], d_i[dn][i]
                pairs += [[x[z1], x[z2]]]
                labels += [1, 0]
        return np.array(pairs), np.array(labels)
    digit_indices_train = [np.where(y_train == i)[0] for i in range(num_classes)]
    digit_indices_test = [np.where(y_test == i)[0] for i in range(num_classes)]
    tr_pairs, tr_y = create_pairs(x_train, digit_indices_train)
    te_pairs, te_y = create_pairs(x_test, digit_indices_test)
    return tr_pairs, tr_y, te_pairs, te_y, num_classes
import random

import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
from pathconfig import TRAINING_PADDING_FILE, TRAINING_SPLIT_FILE


def normalize_max_min(x, axis=0):
    max = np.max(x, axis=axis, keepdims=True)
    min = np.min(x, axis=axis, keepdims=True)
    return (x - min) / (max - min)


# 不通用，不是随机打乱，以均衡为主
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
    x = normalize_max_min(x, axis=2)
    x = x.reshape((x.shape[0], x.shape[1], x.shape[2], 1))
    print(x.shape)
    y = dataset['y']
    x_train, x_test, y_train, y_test = dataset_split(x, y, ratio=0.8)  # 最好保存一下
    np.savez_compressed(splitdata_path,
                        x_train=x_train, x_test=x_test, y_train=y_train, y_test=y_test)
    return x_train, x_test, y_train, y_test

# siamese
def pair_data_split_and_save(rawdata_path, splitdata_path):
    dataset = np.load(rawdata_path)
    x = dataset['x']
    x = normalize_max_min(x, axis=2)
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
def create_one_shot_pair_data(rawdata_path, splitdata_path, gesture_code, balanced=True):
    dataset = np.load(rawdata_path)
    x = dataset['x']
    x = normalize_max_min(x, axis=2)
    x = x.reshape((x.shape[0], x.shape[1], x.shape[2], 1))
    print(x.shape)
    y = dataset['y']
    num_classes = len(np.unique(y))
    if gesture_code >= num_classes:
        raise ValueError('gesture don\'t exist')
    digit_indices_main = np.where(y == gesture_code)[0]
    digit_indices_other = np.where(y != gesture_code)[0]
    x_main = x[digit_indices_main]
    x_other = x[digit_indices_other]
    # pair, keep positive and negative samples balanced
    def create_pairs(x_m, x_other):
        '''Positive and negative pair creation.
        Alternates between positive and negative pairs.
        '''
        positive_pairs = []
        positive_labels = []
        negative_pairs = []
        negative_labels = []
        # positive samples
        for i in range(x_m.shape[0]):
            for j in range(i):
                positive_pairs.append([x_m[i], x_m[j]])
                positive_labels.append(1)
            for j in range(x_other.shape[0]):
                negative_pairs.append([x_m[i], x_other[j]])
                negative_labels.append(0)
        positive_pairs = np.array(positive_pairs)
        positive_labels = np.array(positive_labels)
        negative_pairs = np.array(negative_pairs)
        negative_labels = np.array(negative_labels)
        # 均衡正负样本
        if balanced:
            random_indice = np.random.permutation(len(negative_labels))
            negative_pairs = negative_pairs[random_indice[:len(positive_labels)]]
            negative_labels = negative_labels[random_indice[:len(positive_labels)]]

        pairs = np.concatenate((positive_pairs, negative_pairs), axis=0)
        labels = np.concatenate((positive_labels, negative_labels))
        return pairs, labels
    pairs, labels = create_pairs(x_main, x_other)
    return pairs, labels, num_classes
def one_shot_pair_data_split_and_save(rawdata_path, splitdata_path, gesture_code):
    pairs, labels, num_classes = create_one_shot_pair_data(rawdata_path, splitdata_path, gesture_code, balanced=True)
    tr_pairs, te_pairs, tr_y, te_y = train_test_split(pairs, labels, test_size=0.2, random_state=1123)
    return tr_pairs, tr_y, te_pairs, te_y, num_classes
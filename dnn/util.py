import numpy as np


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
        random_index = random_index + cursor
        x_random = x_sorted[random_index]
        y_random = y_sorted[random_index]

        x_random = list(x_random)
        y_random = list(y_random)

        x_train = x_train + x_random[:int(sample_num*ratio)]
        x_test = x_test + x_random[int(sample_num*ratio):]

        y_train = y_train + y_random[:int(sample_num*ratio)]
        y_test = y_test + y_random[int(sample_num*ratio):]

        cursor = cursor + sample_num
    x_train = np.array(x_train)
    x_test = np.array(x_test)
    y_train = np.array(y_train)
    y_test = np.array(y_test)

    train_random = np.random.permutation(x_train.shape[0])
    test_random = np.random.permutation(x_test.shape[0])

    return x_train[train_random], x_test[test_random], y_train[train_random], y_test[test_random]
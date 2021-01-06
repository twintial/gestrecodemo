from dnn.model import cons_model, train_model, train_model_v2
from dnn.util import load_dataset, load_dataset_v2
import numpy as np
import keras

num_classes = 3


def training_first_time():
    x_train, x_test, y_train, y_test = load_dataset_v2(r'../t', num_classes)
    # np.savetxt(r'../dataset/loaded/x_train.txt', x_train.flatten())
    # np.savetxt(r'../dataset/loaded/x_test.txt', x_test.flatten())
    # np.savetxt(r'../dataset/loaded/y_train.txt', y_train.flatten())
    # np.savetxt(r'../dataset/loaded/y_test.txt', y_test.flatten())

    model = cons_model(x_train.shape[1:], num_classes)
    train_model_v2(model, x_train, x_test, y_train, y_test, batch_size=32, epochs=400, save_path='models/1.h5')


def training_after():
    x_train = np.loadtxt(r'../dataset/loaded/x_train.txt').reshape((-1, 8, 2048, 1))
    x_test = np.loadtxt(r'../dataset/loaded/x_test.txt').reshape((-1, 8, 2048, 1))
    y_train = np.loadtxt(r'../dataset/loaded/y_train.txt').reshape((-1, 8, 2048, 1))
    y_test = np.loadtxt(r'../dataset/loaded/y_test.txt').reshape((-1, 8, 2048, 1))
    model = cons_model(x_train.shape[1:], num_classes)
    train_model_v2(model, x_train, x_test, y_train, y_test, batch_size=32, epochs=400, save_path='models/1.h5')



if __name__ == '__main__':
    # x, y = load_dataset(r'../dataset')
    # y_onehot = keras.utils.to_categorical(y, num_classes)
    # model = cons_model(x.shape[1:], num_classes)
    # train_model(model, x, y_onehot, batch_size=32, epochs=400, save_path='models/1.h5')
    training_first_time()
    # x_train, x_test, y_train, y_test = load_dataset_v2(r'../t', 3)
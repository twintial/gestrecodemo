from keras import models

from dnn.model import cons_model, train_model, train_model_v2, val_model
from dnn.datapreprocess import load_dataset, load_dataset_v2
import numpy as np
import keras

from dnn.util import dataset_split

num_classes = 10


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


def no_window_training_rawdata(rawdata_path, splitdata_path):
    dataset = np.load(rawdata_path)
    x = dataset['x']
    x = x.reshape((x.shape[0], x.shape[1], x.shape[2], 1))
    print(x.shape)
    y = dataset['y']
    x_train, x_test, y_train, y_test = dataset_split(x, y, ratio=0.8) # 最好保存一下
    np.savez_compressed(splitdata_path,
                        x_train=x_train, x_test=x_test, y_train=y_train, y_test=y_test)
    print(x_train.shape)
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)
    model = cons_model(x.shape[1:], num_classes)
    train_model_v2(model, x_train, x_test, y_train, y_test, batch_size=16, epochs=1000, save_path='models/1.h5')
    val_model(model, x_test, y_test, num_classes)
    # train_model(model, x, y_onehot, batch_size=1, epochs=20)


def no_window_training_splitdata(splitdata_path):
    rawdata = np.load(splitdata_path)
    x_train = rawdata['x_train']
    x_test = rawdata['x_test']
    y_train = rawdata['y_train']
    y_test = rawdata['y_test']
    print(x_train.shape)
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)
    model = cons_model(x_train.shape[1:], num_classes)
    train_model_v2(model, x_train, x_test, y_train, y_test, batch_size=16, epochs=1000, save_path='models/1.h5')
    val_model(model, x_test, y_test, num_classes)
    # train_model(model, x, y_onehot, batch_size=1, epochs=20)


# def analyze(model_file):
#     model = keras.models.load_model(model_file)
#     val_model(model)





if __name__ == '__main__':
    # x, y = load_dataset(r'../dataset')
    # y_onehot = keras.utils.to_categorical(y, num_classes)
    # model = cons_model(x.shape[1:], num_classes)
    # train_model(model, x, y_onehot, batch_size=32, epochs=400, save_path='models/best.h5')
    # training_first_time()
    # x_train, x_test, y_train, y_test = load_dataset_v2(r'../t', 3)

    # no_window_training_rawdata(r'../data/gesture/padding/dataset.npz', r'../data/gesture/split/splitdata.npz')
    # no_window_training_splitdata(r'../data/gesture/split/splitdata.npz')
    splitdata_path = r'../data/gesture/split/splitdata.npz'
    rawdata = np.load(splitdata_path)
    x_train = rawdata['x_train']
    x_test = rawdata['x_test']
    y_train = rawdata['y_train']
    y_test = rawdata['y_test']
    y_test = keras.utils.to_categorical(y_test, num_classes)
    model = models.load_model(r'models/best.h5')
    val_model(model, x_test, y_test, num_classes)
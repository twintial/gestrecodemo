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


def no_window_training():
    dataset = np.load(r'../data/gesture/padding/dataset.npz')
    x = dataset['x']
    x = x.reshape((x.shape[0], x.shape[1], x.shape[2], 1))
    print(x.shape)
    y = dataset['y']
    x_train, x_test, y_train, y_test = dataset_split(x, y, ratio=0.8) # 最好保存一下
    print(x_train.shape)
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)
    model = cons_model(x.shape[1:], num_classes)
    train_model_v2(model, x_train, x_test, y_train, y_test, batch_size=8, epochs=1400, save_path='models/1.h5')
    val_model(model, x_test, y_test, num_classes)
    # train_model(model, x, y_onehot, batch_size=1, epochs=20)


# def analyze(model_file):
#     model = keras.models.load_model(model_file)
#     val_model(model)





if __name__ == '__main__':
    # x, y = load_dataset(r'../dataset')
    # y_onehot = keras.utils.to_categorical(y, num_classes)
    # model = cons_model(x.shape[1:], num_classes)
    # train_model(model, x, y_onehot, batch_size=32, epochs=400, save_path='models/1.h5')
    # training_first_time()
    # x_train, x_test, y_train, y_test = load_dataset_v2(r'../t', 3)

    no_window_training()
    # analyze('models/1.h5')
import os
os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
import tensorflow as tf
from tensorflow.keras import Model, layers
from tensorflow.keras import backend as K

from nn.util import pair_data_split_and_save, one_shot_pair_data_split_and_save, create_one_shot_pair_data, \
    normalize_max_min
from pathconfig import TRAINING_PADDING_FILE, TRAINING_SPLIT_FILE, TEST_PADDING_FILE, EXPERIMENT_NAME

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

weights_path = rf'models/{EXPERIMENT_NAME}_64_siamese_weights.h5'


def euclidean_distance(vects):
    x, y = vects
    sum_square = K.sum(K.square(x - y), axis=1, keepdims=True)
    return K.sqrt(K.maximum(sum_square, K.epsilon()))

def contrastive_loss(y_true, y_pred):
    '''Contrastive loss from Hadsell-et-al.'06
    http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    '''
    y_true = tf.cast(y_true, tf.float32)
    margin = 1
    sqaure_pred = K.square(y_pred)
    margin_square = K.square(K.maximum(margin - y_pred, 0))
    return K.mean(y_true * sqaure_pred + (1 - y_true) * margin_square)

def compute_accuracy(y_true, y_pred):  # numpy上的操作
    '''Compute classification accuracy with a fixed threshold on distances.
    '''
    pred = y_pred.ravel() < 0.5
    return np.mean(pred == y_true)

def accuracy(y_true, y_pred):  # Tensor上的操作
    '''Compute classification accuracy with a fixed threshold on distances.
    '''
    return K.mean(K.equal(y_true, K.cast(y_pred < 0.5, y_true.dtype)))


def plot_train_history(history, train_metrics, val_metrics):
    plt.plot(history.history.get(train_metrics), '-o')
    plt.plot(history.history.get(val_metrics), '-o')
    plt.ylabel(train_metrics)
    plt.xlabel('Epochs')
    plt.legend(['train', 'validation'])

class BaseNet(Model):
    def __init__(self):
        super(BaseNet, self).__init__()
        self.flatten = layers.Flatten()
        self.dense1 = layers.Dense(128, activation=tf.nn.relu)
        self.dropout1 = layers.Dropout(0.1)
        self.dense2 = layers.Dense(128, activation=tf.nn.relu)
        self.dropout2 = layers.Dropout(0.1)
        self.dense3 = layers.Dense(128, activation=tf.nn.relu)

    def call(self, inputs, training=None, mask=None):
        x = self.flatten(inputs)
        x = self.dense1(x)
        x = self.dropout1(x)
        x = self.dense2(x)
        x = self.dropout2(x)
        return self.dense3(x)

class BaseCnnNet(Model):
    def __init__(self, num_classes):
        super(BaseCnnNet, self).__init__()
        self.conv2d_1 = layers.Conv2D(8,
                                      kernel_size=(3, 8),
                                      strides=(1, 1),
                                      bias_initializer=tf.initializers.Constant(value=0.1))
        self.batch_norm_1 = layers.BatchNormalization()
        self.relu_1 = layers.ReLU()
        self.max_pooling2d_1 = layers.MaxPooling2D(pool_size=(1, 3), strides=(1, 3), padding='same')

        self.conv2d_2 = layers.Conv2D(16,
                                      kernel_size=(3, 8),
                                      strides=(1, 1),
                                      bias_initializer=tf.initializers.Constant(value=0.1))
        self.batch_norm_2 = layers.BatchNormalization()
        self.relu_2 = layers.ReLU()
        self.max_pooling2d_2 = layers.MaxPooling2D(pool_size=(1, 4), strides=(1, 4), padding='same')

        self.conv2d_3 = layers.Conv2D(32,
                                      kernel_size=(3, 5),
                                      strides=(1, 1),
                                      bias_initializer=tf.initializers.Constant(value=0.1))
        self.batch_norm_3 = layers.BatchNormalization(trainable=True, scale=True)
        self.relu_3 = layers.ReLU()
        self.max_pooling2d_3 = layers.MaxPooling2D(pool_size=(2, 3), strides=(2, 3), padding='same')


        self.conv2d_4 = layers.Conv2D(32,
                                      kernel_size=(3, 3),
                                      strides=(1, 1),
                                      bias_initializer=tf.initializers.Constant(value=0.1))
        self.batch_norm_4 = layers.BatchNormalization(trainable=True, scale=True)
        self.relu_4 = layers.ReLU()
        self.max_pooling2d_4 = layers.MaxPooling2D(pool_size=(1, 3), strides=(1, 3), padding='same')

        self.conv2d_5 = layers.Conv2D(32,
                                      kernel_size=(3, 3),
                                      strides=(1, 1),
                                      bias_initializer=tf.initializers.Constant(value=0.1))
        self.batch_norm_5 = layers.BatchNormalization(trainable=True, scale=True)
        self.relu_5 = layers.ReLU()
        self.max_pooling2d_5 = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')

        self.flatten = layers.Flatten()
        self.dense1 = layers.Dense(128, activation='relu', bias_initializer=tf.initializers.Constant(value=0.1))
        self.dropout = layers.Dropout(0.2)
        # self.dense2 = layers.Dense(num_classes, activation='softmax')
        self.dense2 = layers.Dense(64, activation='softmax')

    def call(self, inputs, training=None, mask=None):
        x = self.conv2d_1(inputs)
        x = self.batch_norm_1(x)
        x = self.relu_1(x)
        x = self.max_pooling2d_1(x)

        x = self.conv2d_2(x)
        x = self.batch_norm_2(x)
        x = self.relu_2(x)
        x = self.max_pooling2d_2(x)

        x = self.conv2d_3(x)
        x = self.batch_norm_3(x)
        x = self.relu_3(x)
        x = self.max_pooling2d_3(x)

        x = self.conv2d_4(x)
        x = self.batch_norm_4(x)
        x = self.relu_4(x)
        x = self.max_pooling2d_4(x)

        x = self.conv2d_5(x)
        x = self.batch_norm_5(x)
        x = self.relu_5(x)
        x = self.max_pooling2d_5(x)

        # 这里有问题
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dropout(x)
        x = self.dense2(x)
        return x

class SiameseNet(Model):
    def __init__(self, num_classes):
        super(SiameseNet, self).__init__()
        self.distant = layers.Lambda(euclidean_distance)
        self.base_cnn_net = BaseCnnNet(num_classes)
    def call(self, inputs, training=None, mask=None):
        input_a, input_b = inputs
        processed_a = self.base_cnn_net(input_a)
        processed_b = self.base_cnn_net(input_b)
        distant_output = self.distant([processed_a, processed_b])
        return distant_output

# 将数据pair后放入siamese训练
def pre_training():
    tr_pairs, tr_y, te_pairs, te_y, num_classes = pair_data_split_and_save(TRAINING_PADDING_FILE, TRAINING_SPLIT_FILE)
    model = SiameseNet(num_classes)
    model.compile(loss=contrastive_loss, optimizer=tf.keras.optimizers.Adam(), metrics=[accuracy])
    history = model.fit([tr_pairs[:, 0], tr_pairs[:, 1]], tr_y,
                        batch_size=32, epochs=1000, verbose=2,
                        validation_data=([te_pairs[:, 0], te_pairs[:, 1]], te_y))
    # model.summary()
    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plot_train_history(history, 'loss', 'val_loss')
    plt.subplot(1, 2, 2)
    plot_train_history(history, 'accuracy', 'val_accuracy')
    plt.show()

    # compute final accuracy on training and test sets
    y_pred = model.predict([tr_pairs[:, 0], tr_pairs[:, 1]])
    tr_acc = compute_accuracy(tr_y, y_pred)
    y_pred = model.predict([te_pairs[:, 0], te_pairs[:, 1]])
    te_acc = compute_accuracy(te_y, y_pred)

    print('* Accuracy on training set: %0.2f%%' % (100 * tr_acc))
    print('* Accuracy on test set: %0.2f%%' % (100 * te_acc))

    model.save_weights(weights_path)

# 选择一个手势来做
def one_shot_training(gesture_code):
    one_shot_weights_path = rf'models/one_shot_{gesture_code}_weights.h5'
    tr_pairs, tr_y, te_pairs, te_y, num_classes = one_shot_pair_data_split_and_save(TRAINING_PADDING_FILE,
                                                                                    TRAINING_SPLIT_FILE, gesture_code)
    model = SiameseNet(num_classes)
    model.build(input_shape=[(None, 56, 777, 1), (None, 56, 777, 1)])
    model.load_weights(weights_path)
    model.compile(loss=contrastive_loss, optimizer=tf.keras.optimizers.Adam(), metrics=[accuracy])

    loss, acc = model.evaluate([tr_pairs[:, 0], tr_pairs[:, 1]], tr_y)


    history = model.fit([tr_pairs[:, 0], tr_pairs[:, 1]], tr_y,
                        batch_size=32, epochs=100, verbose=2,
                        validation_data=([te_pairs[:, 0], te_pairs[:, 1]], te_y))
    # model.summary()
    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plot_train_history(history, 'loss', 'val_loss')
    plt.subplot(1, 2, 2)
    plot_train_history(history, 'accuracy', 'val_accuracy')
    plt.show()

    # compute final accuracy on training and test sets
    y_pred = model.predict([tr_pairs[:, 0], tr_pairs[:, 1]])
    tr_acc = compute_accuracy(tr_y, y_pred)
    y_pred = model.predict([te_pairs[:, 0], te_pairs[:, 1]])
    te_acc = compute_accuracy(te_y, y_pred)
    print('loss on training set before one shot training: %0.6f' % loss)
    print('accuracy on training set before one shot training: %0.2f%%' % acc)
    print('* Accuracy on training set: %0.2f%%' % (100 * tr_acc))
    print('* Accuracy on test set: %0.2f%%' % (100 * te_acc))

    model.save_weights(one_shot_weights_path)

def one_shot_eval(gesture_code):
    one_shot_weights_path = rf'models/one_shot_{gesture_code}_weights.h5'
    pairs, labels, num_classes = create_one_shot_pair_data(TEST_PADDING_FILE, None, gesture_code, balanced=False)
    model = SiameseNet(num_classes)
    model.build(input_shape=[(None, 56, 777, 1), (None, 56, 777, 1)])
    model.load_weights(one_shot_weights_path)
    model.compile(loss=contrastive_loss, optimizer=tf.keras.optimizers.Adam(), metrics=[accuracy])
    loss, acc = model.evaluate([pairs[:, 0], pairs[:, 1]], labels)
    print('loss: %0.6f' % loss)
    print('accuracy: %0.2f%%' % acc)

# 一个通用的one_shot_training
def create_pairs(x_m, x_other, balanced):
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
def one_shot_training_general(x_real, x_fake, pre_trained_model_path,save_model_weights_path,balanced=True):
    pairs, labels = create_pairs(x_real, x_fake, balanced=balanced)
    tr_pairs, te_pairs, tr_y, te_y = train_test_split(pairs, labels, train_size=0.8, random_state=1123)
    model = SiameseNet(10)
    model.build(input_shape=[(None, 56, 1400, 1), (None, 56, 1400, 1)])
    model.load_weights(pre_trained_model_path)
    model.compile(loss=contrastive_loss, optimizer=tf.keras.optimizers.Adam(), metrics=[accuracy])

    loss, acc = model.evaluate([tr_pairs[:, 0], tr_pairs[:, 1]], tr_y)

    history = model.fit([tr_pairs[:, 0], tr_pairs[:, 1]], tr_y,
                        batch_size=32, epochs=50, verbose=2,
                        validation_data=([te_pairs[:, 0], te_pairs[:, 1]], te_y))
    # model.summary()
    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plot_train_history(history, 'loss', 'val_loss')
    plt.subplot(1, 2, 2)
    plot_train_history(history, 'accuracy', 'val_accuracy')
    plt.show()

    # compute final accuracy on training and test sets
    y_pred = model.predict([tr_pairs[:, 0], tr_pairs[:, 1]])
    tr_acc = compute_accuracy(tr_y, y_pred)
    y_pred = model.predict([te_pairs[:, 0], te_pairs[:, 1]])
    te_acc = compute_accuracy(te_y, y_pred)
    print('loss on training set before one shot training: %0.6f' % loss)
    print('accuracy on training set before one shot training: %0.2f%%' % acc)
    print('* Accuracy on training set: %0.2f%%' % (100 * tr_acc))
    print('* Accuracy on test set: %0.2f%%' % (100 * te_acc))

    model.save_weights(save_model_weights_path)

def one_shot_training_general_main():
    # load real data
    gesture_code = 10
    dataset = np.load(r'../data/siamese/sjj/10/padding/dataset.npz')
    x = dataset['x']
    x = normalize_max_min(x, axis=2)

    x = x.reshape((x.shape[0], x.shape[1], x.shape[2], 1))
    print(x.shape)
    y = dataset['y']
    digit_indices_main = np.where(y == gesture_code)[0]
    x_real = x[digit_indices_main]
    # load fake data
    dataset_save_file = r'../data/siamese/zq/10/padding/dataset.npz'
    fake_dataset = np.load(dataset_save_file)
    x = fake_dataset['x']
    x = normalize_max_min(x, axis=2)
    x = x.reshape((x.shape[0], x.shape[1], x.shape[2], 1))
    x_fake = x
    save_model_path = rf'models/one_shot_zq_{gesture_code}_weights.h5'
    one_shot_training_general(x_real, x_fake, weights_path, save_model_path)


if __name__ == '__main__':
    np.random.seed(1123)
    # pre_training()
    # one_shot_training(0)
    # one_shot_eval(6)
    one_shot_training_general_main()

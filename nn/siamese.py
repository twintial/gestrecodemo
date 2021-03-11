import tensorflow as tf
from tensorflow.keras import Model, layers
from tensorflow.keras import backend as K

from nn.util import pair_data_split_and_save
from pathconfig import TRAINING_PADDING_FILE, TRAINING_SPLIT_FILE

import numpy as np
import matplotlib.pyplot as plt


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
        self.dense2 = layers.Dense(num_classes, activation='softmax')

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


if __name__ == '__main__':
    tr_pairs, tr_y, te_pairs, te_y, num_classes = pair_data_split_and_save(TRAINING_PADDING_FILE, TRAINING_SPLIT_FILE)
    model = SiameseNet(num_classes)
    model.compile(loss=contrastive_loss, optimizer=tf.keras.optimizers.Adam(), metrics=[accuracy])
    history = model.fit([tr_pairs[:, 0], tr_pairs[:, 1]], tr_y,
              batch_size=32,
              epochs=1000, verbose=2,
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
import tensorflow as tf
from tensorflow.keras import Model, layers
from tensorflow.keras import backend as K


def euclidean_distance(vects):
    x, y = vects
    sum_square = K.sum(K.square(x - y), axis=1, keepdims=True)
    return K.sqrt(K.maximum(sum_square, K.epsilon()))

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

class SiameseNet(Model):
    def __init__(self):
        super(SiameseNet, self).__init__()
        self.distant = layers.Lambda(euclidean_distance)
        self.base_net = BaseNet()
    def call(self, inputs, training=None, mask=None):
        input_a, input_b = inputs
        processed_a = self.base_net(input_a)
        processed_b = self.base_net2(input_b)
        distant_output = self.distant([processed_a, processed_b])
        return distant_output
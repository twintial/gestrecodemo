import tensorflow as tf
from tensorflow.keras import initializers, losses, optimizers, layers
import numpy as np
import pandas as pd
from tensorflow.python.keras.callbacks import ModelCheckpoint

from nn import dataset_split
from nn.util import dataset_split_two_dataset
from pathconfig import PROJECT_DIR
import os


def integrated_model(phase_shape, magn_shape, num_classes):
    # phase
    phase_input = layers.Input(shape=phase_shape, name='phase_input')

    phase_cnn1 = layers.Conv2D(filters=8,
                               kernel_size=(3, 8),
                               bias_initializer=initializers.Constant(value=0.1))(phase_input)
    phase_batch_norm1 = layers.BatchNormalization()(phase_cnn1)
    phase_relu1 = layers.ReLU()(phase_batch_norm1)
    phase_maxpooling1 = layers.MaxPooling2D(pool_size=(1, 3), strides=(1, 3), padding='same')(phase_relu1)

    phase_cnn2 = layers.Conv2D(filters=16,
                               kernel_size=(3, 8),
                               bias_initializer=initializers.Constant(value=0.1))(phase_maxpooling1)
    phase_batch_norm2 = layers.BatchNormalization()(phase_cnn2)
    phase_relu2 = layers.ReLU()(phase_batch_norm2)
    phase_maxpooling2 = layers.MaxPooling2D(pool_size=(1, 4), strides=(1, 4), padding='same')(phase_relu2)

    phase_cnn3 = layers.Conv2D(filters=32,
                               kernel_size=(3, 5),
                               bias_initializer=initializers.Constant(value=0.1))(phase_maxpooling2)
    phase_batch_norm3 = layers.BatchNormalization()(phase_cnn3)
    phase_relu3 = layers.ReLU()(phase_batch_norm3)
    phase_maxpooling3 = layers.MaxPooling2D(pool_size=(1, 4), strides=(1, 4), padding='same')(phase_relu3)

    # magn
    magn_input = layers.Input(shape=magn_shape, name='magn_input')

    magn_cnn1 = layers.Conv2D(filters=8,
                              kernel_size=(3, 8),
                              bias_initializer=initializers.Constant(value=0.1))(magn_input)
    magn_batch_norm1 = layers.BatchNormalization()(magn_cnn1)
    magn_relu1 = layers.ReLU()(magn_batch_norm1)
    magn_maxpooling1 = layers.MaxPooling2D(pool_size=(1, 3), strides=(1, 3), padding='same')(magn_relu1)

    magn_cnn2 = layers.Conv2D(filters=16,
                              kernel_size=(3, 8),
                              bias_initializer=initializers.Constant(value=0.1))(magn_maxpooling1)
    magn_batch_norm2 = layers.BatchNormalization()(magn_cnn2)
    magn_relu2 = layers.ReLU()(magn_batch_norm2)
    magn_maxpooling2 = layers.MaxPooling2D(pool_size=(1, 4), strides=(1, 4), padding='same')(magn_relu2)

    magn_cnn3 = layers.Conv2D(filters=32,
                              kernel_size=(3, 5),
                              bias_initializer=initializers.Constant(value=0.1))(magn_maxpooling2)
    magn_batch_norm3 = layers.BatchNormalization()(magn_cnn3)
    magn_relu3 = layers.ReLU()(magn_batch_norm3)
    magn_maxpooling3 = layers.MaxPooling2D(pool_size=(1, 4), strides=(1, 4), padding='same')(magn_relu3)

    concat = layers.concatenate([phase_maxpooling3, magn_maxpooling3], axis=2)

    concat_cnn1 = layers.Conv2D(filters=32,
                                kernel_size=(3, 3),
                                bias_initializer=initializers.Constant(value=0.1))(concat)
    concat_batch_norm1 = layers.BatchNormalization()(concat_cnn1)
    concat_relu1 = layers.ReLU()(concat_batch_norm1)
    concat_maxpooling1 = layers.MaxPooling2D(pool_size=(2, 4), strides=(2, 4), padding='same')(concat_relu1)

    flatten = layers.Flatten()(concat_maxpooling1)
    dense1 = layers.Dense(128, activation='relu', bias_initializer=initializers.Constant(value=0.1))(flatten)
    dropout1 = layers.Dropout(rate=0.6)(dense1)
    output = layers.Dense(num_classes, activation='softmax')(dropout1)

    model = tf.keras.Model(inputs=[phase_input, magn_input], outputs=[output])

    model.compile(loss=losses.sparse_categorical_crossentropy,
                  optimizer=optimizers.Adam(),
                  metrics=['acc'])

    return model


def train():
    nclasses = 10
    ratio = 0.8
    random_index = np.random.permutation(30)

    phase_dataset = np.load(os.path.join(PROJECT_DIR, 'data/mic_speaker_phase/train/padding/dataset.npz'))
    x_phase = phase_dataset['x']
    x_phase = x_phase.reshape((x_phase.shape[0], x_phase.shape[1], x_phase.shape[2], 1))
    print(x_phase.shape)
    y_phase = phase_dataset['y']

    train_random = np.random.permutation(int(x_phase.shape[0] * ratio))
    test_random = np.random.permutation(int(x_phase.shape[0] * (1 - ratio)))

    x_phase_train, x_phase_test, y_phase_train, y_phase_test = dataset_split_two_dataset(random_index, train_random,
                                                                                         test_random, x_phase, y_phase,
                                                                                         ratio=0.8)  # 最好保存一下

    magn_dataset = np.load(os.path.join(PROJECT_DIR, 'data/mic_speaker_magn/train/padding/dataset.npz'))
    x_magn = magn_dataset['x']
    x_magn = x_magn.reshape((x_magn.shape[0], x_magn.shape[1], x_magn.shape[2], 1))
    print(x_magn.shape)
    y_magn = magn_dataset['y']
    x_magn_train, x_magn_test, y_magn_train, y_magn_test = dataset_split_two_dataset(random_index, train_random,
                                                                                     test_random, x_magn, y_magn,
                                                                                     ratio=0.8)  # 最好保存一下

    model = integrated_model(x_phase.shape[1:], x_magn.shape[1:], nclasses)

    assert np.all(y_magn_train == y_phase_train)
    assert np.all(y_magn_test == y_phase_test)

    save_path = 'temp.h5'
    checkpoint = ModelCheckpoint(save_path, monitor='val_acc', verbose=1, save_best_only=True, mode='max', period=2)
    callbacks_list = [checkpoint]
    history = model.fit((x_phase_train, x_magn_train), y_magn_train,
                        epochs=1000,
                        validation_data=((x_phase_test, x_magn_test), y_magn_test),
                        callbacks=callbacks_list,
                        verbose=1)


def val(model: tf.keras.Model):
    nclasses = 10

    phase_dataset = np.load(os.path.join(PROJECT_DIR, 'data/mic_speaker_phase/test/padding/dataset.npz'))
    x_phase = phase_dataset['x']
    x_phase = x_phase.reshape((x_phase.shape[0], x_phase.shape[1], x_phase.shape[2], 1))
    print(x_phase.shape)
    y_phase = phase_dataset['y']

    magn_dataset = np.load(os.path.join(PROJECT_DIR, 'data/mic_speaker_magn/test/padding/dataset.npz'))
    x_magn = magn_dataset['x']
    x_magn = x_magn.reshape((x_magn.shape[0], x_magn.shape[1], x_magn.shape[2], 1))
    print(x_magn.shape)
    y_magn = magn_dataset['y']

    assert np.all(y_phase == y_magn)
    # loss = model.evaluate((x_phase, x_magn), y_magn) ?? 这是什么意思
    # print(loss)
    val_model(model, x_phase, x_magn, y_phase, 10, 'test.csv')

def val_model(model: tf.keras.Model, x_phase_test, x_magn_test, y_test, nclasses, csv_file):
    analyze_mat = np.zeros((nclasses, nclasses))
    y_predict = model.predict((x_phase_test, x_magn_test))
    for i in range(y_predict.shape[0]):
        predict_class = np.argmax(y_predict[i])
        real_class = y_test[i]
        analyze_mat[real_class][predict_class] = analyze_mat[real_class][predict_class] + 1
    analyze_mat = analyze_mat / np.sum(analyze_mat, axis=1)
    print(analyze_mat)
    acc = np.diag(analyze_mat).mean()
    print(f"acc: {acc}")
    p = pd.DataFrame(analyze_mat)
    p.columns = ['握紧', '张开', '左滑', '右滑', '上滑', '下滑', '前推', '后推', '顺时针转圈', '逆时针转圈']
    p.index = ['握紧', '张开', '左滑', '右滑', '上滑', '下滑', '前推', '后推', '顺时针转圈', '逆时针转圈']
    p.to_csv(csv_file)
    return analyze_mat


if __name__ == '__main__':
    train()
    m = tf.keras.models.load_model(r'temp.h5')
    val(m)

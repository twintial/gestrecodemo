import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras import initializers, losses, optimizers
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, BatchNormalization, ReLU, Dropout, \
    SeparableConv2D
from tensorflow.keras.models import Sequential
from tensorflow.python.keras.callbacks import ModelCheckpoint

from nn.preprocess import print_history


def cons_cnn_model(input_shape, num_classes):
    model = Sequential()
    model.add(Conv2D(8,
                     kernel_size=(3, 8),
                     strides=(1, 1),
                     input_shape=input_shape,
                     bias_initializer=initializers.Constant(value=0.1)))
    model.add(BatchNormalization())
    model.add(ReLU())
    model.add(MaxPooling2D(pool_size=(1, 3), strides=(1, 3), padding='same'))

    model.add(Conv2D(16,
                     kernel_size=(3, 8),
                     strides=(1, 1),
                     bias_initializer=initializers.Constant(value=0.1)))
    model.add(BatchNormalization())
    model.add(ReLU())
    model.add(MaxPooling2D(pool_size=(1, 4), strides=(1, 4), padding='same'))

    model.add(Conv2D(32,
                     kernel_size=(3, 5),
                     strides=(1, 1),
                     bias_initializer=initializers.Constant(value=0.1)))
    model.add(BatchNormalization(trainable=True, scale=True))
    model.add(ReLU())
    model.add(MaxPooling2D(pool_size=(2, 3), strides=(2, 3), padding='same'))

    model.add(Conv2D(32,
                     kernel_size=(3, 3),
                     strides=(1, 1),
                     bias_initializer=initializers.Constant(value=0.1)))
    model.add(BatchNormalization(trainable=True, scale=True))
    model.add(ReLU())
    model.add(MaxPooling2D(pool_size=(1, 3), strides=(1, 3), padding='same'))

    model.add(Conv2D(32,
                     kernel_size=(3, 3),
                     strides=(1, 1),
                     bias_initializer=initializers.Constant(value=0.1)))
    model.add(BatchNormalization(trainable=True, scale=True))
    model.add(ReLU())
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))

    model.add(Flatten())
    model.add(Dense(128, activation='relu', bias_initializer=initializers.Constant(value=0.1)))
    model.add(Dropout(0.2))  # 表示丢弃,0.5还行
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss=losses.categorical_crossentropy,
                  optimizer=optimizers.Adam(),
                  metrics=['acc'])
    model.summary()
    return model


def cons_depthwise_separable_cnn_model(input_shape, num_classes):
    model = Sequential()
    model.add(SeparableConv2D(8,
                              kernel_size=(3, 8),
                              strides=(1, 1),
                              input_shape=input_shape,
                              bias_initializer=initializers.Constant(value=0.1),
                              depth_multiplier=1))
    model.add(BatchNormalization())
    model.add(ReLU())
    model.add(MaxPooling2D(pool_size=(1, 3), strides=(1, 3), padding='same'))

    model.add(SeparableConv2D(16,
                              kernel_size=(3, 8),
                              strides=(1, 1),
                              bias_initializer=initializers.Constant(value=0.1)))
    model.add(BatchNormalization())
    model.add(ReLU())
    model.add(MaxPooling2D(pool_size=(1, 4), strides=(1, 4), padding='same'))

    model.add(SeparableConv2D(32,
                              kernel_size=(3, 5),
                              strides=(1, 1),
                              bias_initializer=initializers.Constant(value=0.1)))
    model.add(BatchNormalization(trainable=True, scale=True))
    model.add(ReLU())
    model.add(MaxPooling2D(pool_size=(2, 3), strides=(2, 3), padding='same'))

    model.add(SeparableConv2D(32,
                              kernel_size=(3, 3),
                              strides=(1, 1),
                              bias_initializer=initializers.Constant(value=0.1)))
    model.add(BatchNormalization(trainable=True, scale=True))
    model.add(ReLU())
    model.add(MaxPooling2D(pool_size=(1, 3), strides=(1, 3), padding='same'))

    model.add(SeparableConv2D(32,
                              kernel_size=(3, 3),
                              strides=(1, 1),
                              bias_initializer=initializers.Constant(value=0.1)))
    model.add(BatchNormalization(trainable=True, scale=True))
    model.add(ReLU())
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))

    model.add(Flatten())
    model.add(Dense(128, activation='relu', bias_initializer=initializers.Constant(value=0.1)))
    model.add(Dropout(0.6))  # 0.4不错
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(loss=losses.categorical_crossentropy,
                  optimizer=optimizers.Adam(),
                  metrics=['acc'])
    model.summary()
    return model

# cons_depthwise_separable_cnn_model(input_shape=(112, 700, 1), num_classes=10)
# cons_cnn_model(input_shape=(112, 700, 1), num_classes=10)


@DeprecationWarning
def train_model(model: Sequential, x, y, batch_size=32, epochs=100, save_path=None):
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8)
    result = model.fit(x_train,
                       y_train,
                       batch_size=batch_size,
                       epochs=epochs,
                       validation_data=(x_test, y_test),
                       verbose=1)
    if save_path:
        model.save(save_path)
    print_history(result.history)
    return result


def train_model_v2(model: Sequential, x_train, x_test, y_train, y_test, batch_size=32, epochs=100, save_path=None):
    # my_callbacks = [
    #     keras.callbacks.EarlyStopping(monitor='val_acc', baseline=0.85),
    # ]
    # 保存最好模型
    checkpoint = ModelCheckpoint(save_path, monitor='val_acc', verbose=2, save_best_only=True, mode='max', save_freq='epoch')
    callbacks_list = [checkpoint]
    result = model.fit(x_train,
                       y_train,
                       batch_size=batch_size,
                       epochs=epochs,
                       validation_data=(x_test, y_test),
                       callbacks=callbacks_list,
                       verbose=2)
    # if save_path:
    #     model.save(save_path)
    # print_history(result.history)
    return result


def val_model(model: Sequential, x_test, y_test, nclasses, csv_file):
    analyze_mat = np.zeros((nclasses, nclasses))
    y_predict = model.predict(x_test)
    for i in range(y_predict.shape[0]):
        predict_class = np.argmax(y_predict[i])
        real_class = np.argmax(y_test[i])
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

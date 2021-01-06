from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, AveragePooling2D
from sklearn.model_selection import train_test_split
import keras

from dnn.util import print_history


def cons_model(input_shape, num_classes):
    model = Sequential()
    model.add(Conv2D(6, kernel_size=(1, 5), strides=(1, 1), activation='relu', input_shape=input_shape))
    model.add(AveragePooling2D(pool_size=(1, 2)))
    model.add(Conv2D(16, (1, 5), activation='relu'))
    model.add(AveragePooling2D(pool_size=(1, 2)))
    model.add(Conv2D(120, (1, 5), activation='relu'))
    model.add(Flatten())
    model.add(Dense(120, activation='relu'))
    model.add(Dense(84, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adam(),
                  metrics=['acc'])
    return model


def train_model(model: Sequential, x, y, batch_size=64, epochs=100, save_path=None):
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8)
    result = model.fit(x_train,
                       y_train,
                       batch_size=batch_size,
                       epochs=epochs,
                       validation_data=(y_train, y_test),
                       verbose=1)
    if save_path:
        model.save(save_path)
    print_history(result)
    return result
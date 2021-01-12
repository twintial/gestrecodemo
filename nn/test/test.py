from keras.models import Sequential
from keras.layers import Conv2D, \
    MaxPooling2D, Flatten, Dense, AveragePooling2D,\
    BatchNormalization, ReLU, Dropout
from sklearn.model_selection import train_test_split
import keras

from nn.datapreprocess import print_history
from nn.test import cnn_input
import numpy as np


my_data = cnn_input.Data_Control(

    '../data/datapre/'
    )
n_class = 8

X = my_data.traindata
X = X.reshape(-1, my_data.traindata.shape[1], my_data.traindata.shape[2],1)
Y = my_data.trainlabel
Y = keras.utils.to_categorical(Y)


#测试数据
Xtest = my_data.testdata
Xtest = Xtest.reshape(-1, my_data.testdata.shape[1], my_data.testdata.shape[2], 1)
Ytest = my_data.testlabel
Ytest = keras.utils.to_categorical(Ytest)
Ytestuser = my_data.testuser

model = Sequential()
model.add(Conv2D(8,
                 kernel_size=(8, 3),
                 strides=(1, 1),
                 input_shape=X.shape[1:],
                 bias_initializer=keras.initializers.Constant(value=0.1)))
model.add(BatchNormalization())
model.add(ReLU())
model.add(MaxPooling2D(pool_size=(3, 1), strides=(3, 1), padding='same'))

model.add(Conv2D(16,
                 kernel_size=(8, 3),
                 strides=(1, 1),
                 bias_initializer=keras.initializers.Constant(value=0.1)))
model.add(BatchNormalization())
model.add(ReLU())
model.add(MaxPooling2D(pool_size=(4, 1), strides=(4, 1), padding='same'))

model.add(Conv2D(32,
                 kernel_size=(5, 3),
                 strides=(1, 1),
                 bias_initializer=keras.initializers.Constant(value=0.1)))
model.add(BatchNormalization(trainable=True, scale=True))
model.add(ReLU())
model.add(MaxPooling2D(pool_size=(3, 2), strides=(3, 2), padding='same'))

model.add(Conv2D(32,
                 kernel_size=(3, 3),
                 strides=(1, 1),
                 bias_initializer=keras.initializers.Constant(value=0.1)))
model.add(BatchNormalization(trainable=True, scale=True))
model.add(ReLU())
model.add(MaxPooling2D(pool_size=(3, 1), strides=(3, 1), padding='same'))

model.add(Conv2D(32,
                 kernel_size=(3, 3),
                 strides=(1, 1),
                 bias_initializer=keras.initializers.Constant(value=0.1)))
model.add(BatchNormalization(trainable=True, scale=True))
model.add(ReLU())
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))
model.add(Flatten())
model.add(Dense(128, activation='relu', bias_initializer=keras.initializers.Constant(value=0.1)))
model.add(Dropout(0.6))
model.add(Dense(n_class, activation='softmax'))

# model = Sequential()
# model.add(Conv2D(32, (5, 1), activation='relu'))
# model.add(AveragePooling2D(pool_size=(2, 2)))
# model.add(Conv2D(64, (5, 1), activation='relu'))
# model.add(AveragePooling2D(pool_size=(2, 2)))
# model.add(Conv2D(128, (5, 1), activation='relu'))
# model.add(Flatten())
# model.add(Dense(120, activation='relu'))
# model.add(Dense(84, activation='relu'))
# model.add(Dense(n_class, activation='softmax'))
#
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adam(),
              metrics=['acc'])
batch_size = 32
epochs = 10000
result = model.fit(X,
                   Y,
                   batch_size=batch_size,
                   epochs=epochs,
                   validation_data=(Xtest, Ytest),
                   verbose=1)
print_history(result.history)


from dnn.model import cons_model, train_model
from dnn.util import load_dataset
import keras

if __name__ == '__main__':
    x, y = load_dataset(r'../t')
    num_classes = 3
    y_onehot = keras.utils.to_categorical(y, num_classes)
    model = cons_model((x.shape[1], x.shape[2], 1), num_classes)
    train_model(model, x, y_onehot, save_path='models/1.h5')

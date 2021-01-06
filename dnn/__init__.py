from dnn.model import cons_model, train_model
from dnn.util import load_dataset
import keras

if __name__ == '__main__':
    x, y = load_dataset(r'../dataset')
    num_classes = 20
    y_onehot = keras.utils.to_categorical(y, num_classes)
    model = cons_model(x.shape[1:], num_classes)
    train_model(model, x, y_onehot, batch_size=32, epochs=400, save_path='models/1.h5')

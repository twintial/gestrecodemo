from dnn.model import cons_model, train_model
from dnn.util import load_dataset
import keras

if __name__ == '__main__':
    x, y = load_dataset(r'../dataset')
    num_classes = 10
    y_hardmax = keras.utils.to_categorical(y, num_classes)
    model = cons_model(x.shape[1:], 10)
    train_model(model, x, y_hardmax, save_path='models/1.h5')

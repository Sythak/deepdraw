from tensorflow.keras.utils import to_categorical
import os

def image_preprocess(X_train, X_test):
    image_size=28
    num_classes = os.environ.get("NUM_CLASSES")

    X_train = X_train.reshape(X_train.shape[0], image_size, image_size, 1).astype('float32')
    X_test = X_test.reshape(X_test.shape[0], image_size, image_size, 1).astype('float32')

    X_train /= 255.0
    X_test /= 255.0


def y_to_categorical(y_train, y_test):
    # Convert class vectors to class matrices, one hot encoded
    num_classes = os.environ.get("NUM_CLASSES")
    y_train = to_categorical(y_train, num_classes)
    y_test = to_categorical(y_test, num_classes)

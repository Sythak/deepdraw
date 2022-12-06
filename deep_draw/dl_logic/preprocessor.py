from tensorflow.keras.utils import to_categorical
import os

def image_preprocess(X_train, X_test):
    image_size=28

    X_train_processed = X_train.reshape(X_train.shape[0], image_size, image_size, 1).astype('float32')
    X_test_processed = X_test.reshape(X_test.shape[0], image_size, image_size, 1).astype('float32')

    X_train_processed /= 255.0
    X_test_processed /= 255.0

    return X_train_processed, X_test_processed


def y_to_categorical(y_train, y_test):
    # Convert class vectors to class matrices, one hot encoded, for cnn non-tfrecords only
    num_classes = os.environ.get("NUM_CLASSES")
    y_train_cat = to_categorical(y_train, num_classes)
    y_test_cat = to_categorical(y_test, num_classes)

    return y_train_cat, y_test_cat

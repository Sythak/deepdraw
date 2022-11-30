from colorama import Fore, Style

import numpy as np
import pandas as pd
import os
import glob

from deep_draw.dl_logic.preprocessor import image_preprocess, y_to_categorical
from deep_draw.dl_logic.cnn import initialize_cnn, compile_cnn, train_cnn_npy, evaluate_cnn, train_cnn_tfrecords, initialize_cnn_tfrecords, compile_cnn_tfrecords, evaluate_cnn_tfrecords
from deep_draw.dl_logic.data import load_data_npy
from deep_draw.dl_logic.params import format_data, root, max_items_per_class, NUM_CLASSES, test_size, learning_rate, patience, batch_size, epochs, validation_split
from deep_draw.dl_logic.data import load_tfrecords_dataset
from deep_draw.dl_logic.params import LOCAL_REGISTRY_PATH
from deep_draw.dl_logic.registry import save_model, load_model, get_model_version
import yaml
from yaml.loader import SafeLoader

def preprocess_train_eval():
# Load & preprocess
    if format_data == 'npy':
        X_train, X_test, y_train, y_test, class_names = load_data_npy(root= root, test_size=test_size, max_items_per_class= max_items_per_class)
        X_train_processed, X_test_processed = image_preprocess(X_train, X_test)
        y_train_cat, y_test_cat = y_to_categorical(y_train, y_test)
        model = None
        if model is None:
            model = initialize_cnn()
        model = compile_cnn(model)
        model, history = train_cnn_npy(model,
                                    X=X_train_processed,
                                    y=y_train_cat,
                                    validation_split=validation_split,
                                    epochs = epochs,
                                    batch_size=batch_size,
                                    patience=patience)
        params_train = dict(
            # Model parameters
            learning_rate=learning_rate,
            batch_size=batch_size,
            patience=patience,
            epochs=epochs,
            validation_split=validation_split,

            # Package behavior
            context="train",

            # Data source
            model_version=get_model_version(),
            )

        res = history.history['val_accuracy'][-1]
        save_model(model, params=params_train, metrics=dict(accuracy=res))

        params_test = dict(
            # Model parameters
            learning_rate=learning_rate,
            batch_size=batch_size,
            patience=patience,
            epochs=epochs,
            validation_split=validation_split,
            # Package behavior
            context="test",
            # Data source
            model_version=get_model_version(),
            )
        accuracy_test = evaluate_cnn(model, X_test_processed, y_test_cat, batch_size=batch_size)['accuracy']
        save_model(params=params_test, metrics=dict(accuracy=accuracy_test))

    if format_data == 'tfrecords':
        dataset_train = load_tfrecords_dataset(source_type = 'train', batch_size=batch_size)
        dataset_val = load_tfrecords_dataset(source_type = 'val', batch_size=batch_size)
        dataset_test = load_tfrecords_dataset(source_type = 'test', batch_size=batch_size)

        # Open the file and load the file
        with open('../dl_logic/categories.yaml') as f:
            class_names = yaml.load(f, Loader=SafeLoader)

        model = None
        if model is None:
            model = initialize_cnn_tfrecords()
        model = compile_cnn_tfrecords(model)

        model, history = train_cnn_tfrecords(model,
                                            dataset_train,
                                            dataset_val,
                                            batch_size=batch_size,
                                            patience=patience,
                                            epochs = epochs)
        params_train = dict(
            # Model parameters
            learning_rate=learning_rate,
            batch_size=batch_size,
            patience=patience,
            epochs=epochs,

            # Package behavior
            context="train",

            # Data source
            model_version=get_model_version(),
            )

        res = history.history['val_accuracy'][-1]
        save_model(model, params=params_train, metrics=dict(accuracy=res))

        params_test = dict(
            # Model parameters
            learning_rate=learning_rate,
            batch_size=batch_size,
            patience=patience,
            epochs=epochs,

            # Package behavior
            context="test",

            # Data source
            model_version=get_model_version(),
            )

        accuracy_test = evaluate_cnn_tfrecords(model, dataset_test, batch_size=batch_size)['accuracy']
        save_model(params=params_test, metrics=dict(accuracy=accuracy_test))

    return class_names

def pred(X_pred):
    model = load_model()
    y_pred = model.predict(X_pred)
    index = np.argmax(y_pred, axis=1)

    # Open the file and load the file
    with open('deep_draw/dl_logic/categories.yaml') as f:
        class_names = yaml.load(f, Loader=SafeLoader)

    prediction = class_names[index[0]]

    #print("\n✅ prediction done: ", y_pred, y_pred.shape)
    return prediction

if __name__ == '__main__':
    class_names = preprocess_train_eval()

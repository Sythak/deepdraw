from colorama import Fore, Style

from tensorflow.keras import Model, Sequential, layers, regularizers, optimizers
from tensorflow.keras.layers import Dense, Masking, Conv1D, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from deep_draw.dl_logic.params import NUM_CLASSES, batch_size, patience, learning_rate, epochs
from deep_draw.dl_logic.registry import load_model, save_model, get_model_version
from deep_draw.dl_logic.data import load_tfrecords_dataset
import numpy as np
from typing import Tuple
import os


def initialize_rnn_tfrecords() -> Model:
    """
    Initialize the Recurrent Neural Network
    Note: Padding & Normalisation already done at the creation of tfrecords files
    """
    print(Fore.BLUE + "\nInitialize model..." + Style.RESET_ALL)

    num_classes = NUM_CLASSES

    model = Sequential()

    model.add(layers.Masking(mask_value=1000, input_shape=(1102,3)))

    model.add(layers.Conv1D(48, (5), activation = 'relu'))
    model.add(Dropout(0.3))
    model.add(layers.Conv1D(64, (5), activation = 'relu'))
    model.add(Dropout(0.3))
    model.add(layers.Conv1D(96, (3), activation = 'relu'))
    model.add(Dropout(0.3))

    model.add(layers.LSTM(units = 128, activation= 'tanh', return_sequences= True))
    model.add(Dropout(0.3))
    model.add(layers.LSTM(units = 128, activation= 'tanh', return_sequences= False))
    model.add(Dropout(0.3))

    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(num_classes, activation = 'softmax'))

    print("\n✅ model initialized")

    return model


def compile_rnn_tfrecords(model: Model) -> Model:
    """
    Compile the RNN model with y not one-hot encoded
    """
    model.compile(
        optimizer='rmsprop',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'])

    print("\n✅ model compiled")
    return model


def train_rnn_tfrecords(model: Model,
                dataset_train,
                dataset_val,
                batch_size=32,
                patience=10,
                epochs = 100) -> Tuple[Model, dict]:
    """
    Fit model and return a the tuple (fitted_model, history)
    """

    print(Fore.BLUE + "\nTrain model..." + Style.RESET_ALL)

    es = EarlyStopping(monitor="val_loss",
                       patience=patience,
                       restore_best_weights=True,
                       verbose=0)

    history = model.fit(dataset_train,
                        validation_data=dataset_val,
                        epochs=epochs,
                        batch_size=batch_size,
                        callbacks=[es],
                        verbose=1)

    print(f"\n✅ model trained")

    return model, history

def evaluate_rnn_tfrecords(model: Model,
                   dataset_test,
                   batch_size=64) -> Tuple[Model, dict]:
    """
    Evaluate trained model performance on dataset
    """

    if model is None:
        print(f"\n❌ no model to evaluate")
        return None

    metrics = model.evaluate(
        dataset_test,
        batch_size=batch_size,
        verbose=1,
    #   callbacks=None,
        return_dict=True)

    loss = metrics["loss"]
    accuracy = metrics["accuracy"]

    print(f"\n✅ model evaluated: loss {round(loss, 2)} accuracy {round(accuracy, 2)}")

    return metrics

if __name__ == "__main__" :
    model=load_model()
    dataset_test = load_tfrecords_dataset(source_type = 'test', batch_size=batch_size)
    params_test = dict(
                # Model parameters
                learning_rate=learning_rate,
                batch_size=batch_size,
                patience=patience,
                epochs=epochs,

                # Package behavior
                context="test_rnn",

                # Data source
                model_version=get_model_version(),
                )

    accuracy_test = evaluate_rnn_tfrecords(model, dataset_test, batch_size=batch_size)['accuracy']
    save_model(params=params_test, metrics=dict(accuracy=accuracy_test))

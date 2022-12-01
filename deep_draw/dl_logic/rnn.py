from colorama import Fore, Style

from tensorflow.keras import Model, Sequential, layers, regularizers, optimizers
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dropout, Dense
from tensorflow.keras.callbacks import EarlyStopping
from deep_draw.dl_logic.params import NUM_CLASSES
import numpy as np
from typing import Tuple
import os


def initialize_rnn() -> Model:
    """
    Initialize the Recurrent Neural Network with random weights
    """
    print(Fore.BLUE + "\nInitialize model..." + Style.RESET_ALL)

    num_classes = NUM_CLASSES

    model = Sequential()

    model.add(layers.Embedding())#Embedding content : input_dim, output_dim

    model.add(layers.LSTM(16, activation= 'tanh', return_sequences= True, recurrent_dropout= 0.3))

    model.add(layers.LSTM(32, activation= 'tanh', return_sequences= True, recurrent_dropout= 0.3))

    model.add(layers.LSTM(64, activation= 'tanh', return_sequences= True, recurrent_dropout= 0.3))

    model.add(Dense(128, activation='relu'))

    model.add(Dense(num_classes, activation = 'softmax'))

    print("\n✅ model initialized")

    return model

def initialize_rnn_tfrecords() -> Model:
    """
    Initialize the Recurrent Neural Network with random weights
    """
    pass


def compile_rnn(model: Model) -> Model:
    """
    Compile the RNN model
    """
    model.compile(
        optimizer='rmsprop',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'])

    print("\n✅ model compiled")
    return model

def compile_rnn_tfrecords(model: Model) -> Model:
    """
    Compile the RNN model
    """
    pass

def train_rnn(model: Model,
                X: np.ndarray,
                y: np.ndarray,
                batch_size=64,
                patience=5,
                validation_split=0.3,
                epochs = 100) -> Tuple[Model, dict]:
    """
    Fit model and return a the tuple (fitted_model, history)
    """

    print(Fore.BLUE + "\nTrain model..." + Style.RESET_ALL)

    es = EarlyStopping(monitor="val_loss",
                       patience=patience,
                       restore_best_weights=True,
                       verbose=0)

    history = model.fit(X,
                        y,
                        validation_split=validation_split,
                        epochs=epochs,
                        batch_size=batch_size,
                        callbacks=[es],
                        verbose=1)

    print(f"\n✅ model trained ({len(X)} rows)")

    return model, history

def train_rnn_tfrecords(model: Model,
                dataset_train,
                dataset_val,
                batch_size=32,
                patience=10,
                epochs = 100) -> Tuple[Model, dict]:
    """
    Fit model and return a the tuple (fitted_model, history)
    """
    pass

def evaluate_rnn(model: Model,
                   X: np.ndarray,
                   y: np.ndarray,
                   batch_size=64) -> Tuple[Model, dict]:
    """
    Evaluate trained model performance on dataset
    """

    print(Fore.BLUE + f"\nEvaluate model on {len(X)} rows..." + Style.RESET_ALL)

    if model is None:
        print(f"\n❌ no model to evaluate")
        return None

    metrics = model.evaluate(
        x=X,
        y=y,
        batch_size=batch_size,
        verbose=1,
    #   callbacks=None,
        return_dict=True)

    loss = metrics["loss"]
    accuracy = metrics["accuracy"]

    print(f"\n✅ model evaluated: loss {round(loss, 2)} accuracy {round(accuracy, 2)}")

    return metrics

def evaluate_rnn_tfrecords(model: Model,
                   dataset_test,
                   batch_size=64) -> Tuple[Model, dict]:
    """
    Evaluate trained model performance on dataset
    """
    pass

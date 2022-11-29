from colorama import Fore, Style

from tensorflow.keras import Model, Sequential, layers, regularizers, optimizers
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dropout, Dense
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np
from typing import Tuple
import os


def initialize_cnn(X: np.ndarray) -> Model:
    """
    Initialize the Convolutional Neural Network with random weights
    """
    print(Fore.BLUE + "\nInitialize model..." + Style.RESET_ALL)

    num_classes = os.environ.get("NUM_CLASSES")

    model = Sequential()

    model.add(Conv2D(16, (3,3), activation='relu', input_shape=(28,28,1)))
    model.add(MaxPooling2D((2,2)))

    model.add(Conv2D(32, (3,3), activation='relu', padding='same'))
    model.add(MaxPooling2D((2,2)))

    model.add(Conv2D(64, (3,3), activation='relu', padding='same'))
    model.add(MaxPooling2D((2,2)))

    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    #model.add(Dropout(0.4))
    model.add(Dense(num_classes, activation = 'softmax'))

    print("\n✅ model initialized")

    return model


def compile_cnn(model: Model) -> Model:
    """
    Compile the CNN
    """
    model.compile(
        optimizer='adam',
        loss='categorical_ crossentropy',
        metrics=['accuracy'])

    print("\n✅ model compiled")
    return model


def train_cnn(model: Model,
                X: np.ndarray,
                y: np.ndarray,
                batch_size=64,
                patience=10,
                validation_split=0.3) -> Tuple[Model, dict]:
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
                        epochs=100,
                        batch_size=batch_size,
                        callbacks=[es],
                        verbose=1)

    print(f"\n✅ model trained ({len(X)} rows)")

    return model, history


def evaluate_cnn(model: Model,
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
        # callbacks=None,
        return_dict=True)

    loss = metrics["loss"]
    accuracy = metrics["accuracy"]

    print(f"\n✅ model evaluated: loss {round(loss, 2)} mae {round(accuracy, 2)}")

    return metrics

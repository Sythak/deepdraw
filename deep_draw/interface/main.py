import numpy as np
import pandas as pd

from deep_draw.dl_logic.preprocessor import image_preprocess, y_to_categorical
from deep_draw.dl_logic.cnn import initialize_cnn, compile_cnn, train_cnn_npy, evaluate_cnn
from deep_draw.dl_logic.data import load_data_npy
from deep_draw.dl_logic.params import format_data, root, max_items_per_class, NUM_CLASSES, test_size, learning_rate, patience, batch_size, epochs, validation_split


# Load & preprocess
if format_data == 'npy':
    X_train, X_test, y_train, y_test, class_names = load_data_npy(root= root, test_size=test_size, max_items_per_class= max_items_per_class)

X_train_processed, X_test_processed = image_preprocess(X_train, X_test)
y_train_cat, y_test_cat = y_to_categorical(y_train, y_test)


# Train
model = None

## here maybe option to load model

if model is None:
    model = initialize_cnn(X_train_processed)
model = compile_cnn(model)
model, history = train_cnn_npy(model,
                               X=X_train_processed,
                               y=y_train_cat,
                               validation_split=validation_split,
                               epochs = epochs,
                               batch_size=batch_size,
                               patience=patience)

# Evaluate
metrics = evaluate_cnn(model, X_test_processed, y_test_cat, batch_size=batch_size)
accuracy = metrics["accuracy"]

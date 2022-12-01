
import numpy as np
import os
import glob
import tensorflow as tf


def load_data_ndjson(root, test_size, max_items_per_class):

    all_files = glob.glob(os.path.join(root, '*.ndjson'))

    #initialize variables
    X = np.empty([0, 784])
    y = np.empty([0])
    class_names = []

    #load a subset of the data to memory
    for idx, file in enumerate(sorted(all_files)):
        data = np.load(file)
        print("\n✅ ", file, " loaded")
        data = data[0: max_items_per_class, :]
        labels = np.full(data.shape[0], idx)

        X = np.concatenate((X, data), axis=0)
        y = np.append(y, labels)

        class_name, ext = os.path.splitext(os.path.basename(file))
        class_names.append(class_name.replace("full_numpy_bitmap_", "").replace(".npy", ""))

    data = None
    labels = None

    #shuffle (to be sure)
    permutation = np.random.permutation(y.shape[0])
    X = X[permutation, :]
    y = y[permutation]

    #separate into training and testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=True)

    return X_train, X_test, y_train, y_test, class_names

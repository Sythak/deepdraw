import numpy as np
import os
import glob
from sklearn.model_selection import train_test_split
import tensorflow as tf

def load_data_npy(root, test_size, max_items_per_class):

    all_files = glob.glob(os.path.join(root, '*.npy'))

    #initialize variables
    X = np.empty([0, 784])
    y = np.empty([0])
    class_names = []

    #load a subset of the data to memory
    for idx, file in enumerate(all_files):
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

def load_shard(root, shard, test_size=0.2, max_items_per_class= 5000):

    all_files = glob.glob(os.path.join(root, '*.npy'))

    #initialize variables
    X = np.empty([0, 784])
    y = np.empty([0])
    class_names = []

    #load a subset of the data to memory
    for idx, file in enumerate(all_files):
        print(file)
        data = np.load(file)
        data = data[shard*max_items_per_class: (shard+1)*max_items_per_class, :]
        labels = np.full(data.shape[0], idx)

        X = np.concatenate((X, data), axis=0)
        y = np.append(y, labels)

        class_name, ext = os.path.splitext(os.path.basename(file))
        class_names.append(class_name)

    data = None
    labels = None

    #shuffle (to be sure)
    permutation = np.random.permutation(y.shape[0])
    X = X[permutation, :]
    y = y[permutation]

    #separate into training and testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=True)

    return X_train.astype(int), X_test.astype(int), y_train.astype(int), y_test.astype(int), class_names

def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))): # if value ist tensor
        value = value.numpy() # get value of tensor
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
    """Returns a floast_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def serialize_array(array):
    array = tf.io.serialize_tensor(array)
    return array

def parse_single_image(image, label):

    #define the dictionary -- the structure -- of our single example
    data = {
        'height' : _int64_feature(image.shape[0]),
        'width' : _int64_feature(image.shape[1]),
        'depth' : _int64_feature(image.shape[2]),
        'raw_image' : _bytes_feature(serialize_array(image)),
        'label' : _int64_feature(label)
    }
    #create an Example, wrapping the single features
    out = tf.train.Example(features=tf.train.Features(feature=data))

    return out

def write_images_to_tfr_short(images, labels, filename:str="images"):
    filename= filename+".tfrecords"
    writer = tf.io.TFRecordWriter(filename) #create a writer that'll store our data to disk
    count = 0

    for index in range(len(images)):

        #get the data we want to write
        current_image = images[index]
        current_label = labels[index][0]

        out = parse_single_image(image=current_image, label=current_label)
        writer.write(out.SerializeToString())
        count += 1

    writer.close()
    print(f"Wrote {count} elements to TFRecord")
    return count

def parse_tfr_element(element):
    #use the same structure as above; it's kinda an outline of the structure we now want to create
    data = {
      'height': tf.io.FixedLenFeature([], tf.int64),
      'width':tf.io.FixedLenFeature([], tf.int64),
      'label':tf.io.FixedLenFeature([], tf.int64),
      'raw_image' : tf.io.FixedLenFeature([], tf.string),
      'depth':tf.io.FixedLenFeature([], tf.int64),
    }


    content = tf.io.parse_single_example(element, data)

    height = content['height']
    width = content['width']
    depth = content['depth']
    label = content['label']
    raw_image = content['raw_image']


    #get our 'feature'-- our image -- and reshape it appropriately
    feature = tf.io.parse_tensor(raw_image, out_type=tf.int64)
    feature = tf.reshape(feature, shape=[height,width,depth])
    return (feature, label)

def get_dataset_multi(tfr_dir: str = "/content/", pattern: str = "*.tfrecords"):
    files = glob.glob(os.path.join(tfr_dir, pattern), recursive=False)
    print(files)

    #create the dataset
    dataset = tf.data.TFRecordDataset(files)

    #pass every single feature through our mapping function
    dataset = dataset.map(parse_tfr_element)

    return dataset

# Load train dataset
dataset_train = get_dataset_multi(tfr_dir='../raw_data/tfrecords/', pattern="*_train.tfrecords")
dataset_train = dataset_train.batch(32)
dataset_train = dataset_train.map(lambda x, y:(tf.cast(x, tf.float32)/255.0, y))

# Load val dataset
dataset_val = get_dataset_multi(tfr_dir='../raw_data/tfrecords/', pattern="*_val.tfrecords")
dataset_val = dataset_val.batch(32)
dataset_val = dataset_val.map(lambda x, y:(tf.cast(x, tf.float32)/255.0, y))

# Load test dataset
dataset_test = get_dataset_multi(tfr_dir='../raw_data/tfrecords/', pattern="*_test.tfrecords")
dataset_test = dataset_test.batch(32)
dataset_test = dataset_test.map(lambda x, y:(tf.cast(x, tf.float32)/255.0, y))

# Create tfrecords
for shard in range(10):
    print(shard)
    X_train, X_test, y_train, y_test, class_names = load_shard('./npy/', shard, test_size=0.3, max_items_per_class=10000)
    X_train = X_train.reshape(-1,28,28,1)
    X_test = X_test.reshape(-1,28,28,1)
    y_train = y_train.reshape(-1,1)
    y_test = y_test.reshape(-1,1)
    write_images_to_tfr_short(X_train, y_train, filename=f"shard_{shard}_train")
    write_images_to_tfr_short(X_test, y_test, filename=f"shard_{shard}_test")

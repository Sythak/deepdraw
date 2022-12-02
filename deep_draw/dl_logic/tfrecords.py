from deep_draw.dl_logic.params import model_selection

def load_shard(root, shard, test_size=0.2, max_items_per_class= 5000):

    all_files = glob.glob(os.path.join(root, '*.npy'))

    #initialize variables
    X = np.empty([0, 784])
    y = np.empty([0])
    class_names = []

    #load a subset of the data to memory
    for idx, file in enumerate(sorted(all_files)):
        print(file)
        data = np.load(file)
        data = data[shard*max_items_per_class: (shard+1)*max_items_per_class, :]
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

    for index in range(len(images)

        #get the data we want to write
        current_image = images[index]
        current_label = labels[index][0]

        out = parse_single_image(image=current_image, label=current_label)
        writer.write(out.SerializeToString())
        count += 1

    writer.close()
    print(f"Wrote {count} elements to TFRecord")
    return count


def padding(stroke, max_len=500):
    """Converts from stroke-3 to stroke-5 format and pads to given length."""
    # (But does not insert special start token).

    result = np.ones((max_len, 3), dtype=float)*1000
    length = len(stroke)
    assert length <= max_len
    result[0:length, 0:3] = stroke[:, 0:3]
    return result

def parse_line(ndjson_line):
    """Parse an ndjson line and return ink (as np array) and classname."""
    sample = json.loads(ndjson_line)
    class_name = sample["word"]
    if not class_name:
        print ("Empty classname")
        return None, None
    inkarray = sample["drawing"]
    stroke_lengths = [len(stroke[0]) for stroke in inkarray]
    total_points = sum(stroke_lengths)
    np_ink = np.zeros((total_points, 3), dtype=np.float32)
    current_t = 0
    if not inkarray:
        print("Empty inkarray")
        return None, None
    for stroke in inkarray:
        if len(stroke[0]) != len(stroke[1]):
            print("Inconsistent number of x and y coordinates.")
            return None, None
        for i in [0, 1]:
            np_ink[current_t:(current_t + len(stroke[0])), i] = stroke[i]
        current_t += len(stroke[0])
        np_ink[current_t - 1, 2] = 1  # stroke_end
    # Preprocessing.
    # 1. Size normalization.
    lower = np.min(np_ink[:, 0:2], axis=0)
    upper = np.max(np_ink[:, 0:2], axis=0)
    scale = upper - lower
    scale[scale == 0] = 1
    np_ink[:, 0:2] = (np_ink[:, 0:2] - lower) / scale
    # 2. Compute deltas.
    np_ink[1:, 0:2] -= np_ink[0:-1, 0:2]
    np_ink = np_ink[1:, :]
    return padding(np_ink, max_len=max_point), class_name

def convert_data(trainingdata_dir,
                 observations_per_class,
                 output_file,
                 classnames,
                 output_shards=10,
                 offset=0):
    """Convert training data from ndjson files into tf.Example in tf.Record.

    Args:
    trainingdata_dir: path to the directory containin the training data.
     The training data is stored in that directory as ndjson files.
    observations_per_class: the number of items to load per class.
    output_file: path where to write the output.
    classnames: array with classnames - is auto created if not passed in.
    output_shards: the number of shards to write the output in.
    offset: the number of items to skip at the beginning of each file.

    Returns:
    classnames: the class names as strings. classnames[classes[i]] is the
      textual representation of the class of the i-th data point.
    """

    def _pick_output_shard():
        return random.randint(0, output_shards - 1)

    file_handles = []
    # Open all input files.
    for filename in sorted(tf.compat.v1.gfile.ListDirectory(trainingdata_dir)):
        if not filename.endswith(".ndjson"):
            print("Skipping", filename)
            continue
        file_handles.append(
            tf.io.gfile.GFile(os.path.join(trainingdata_dir, filename), "r"))
        if offset != 0:  # Fast forward all files to skip the offset.
            count = 0
            for _ in file_handles[-1]:
                count += 1
                if count == offset:
                    break

    writers = []
    for i in range(output_shards):
        writers.append(
            tf.io.TFRecordWriter("%s-%05i-of-%05i" % (output_file, i,
                                                         output_shards)))

    reading_order = list(range(len(file_handles))) * observations_per_class
    random.shuffle(reading_order)

    for c in reading_order:
        line = file_handles[c].readline()
        ink = None
        while ink is None:
            ink, class_name = parse_line(line)
            if ink is None:
                print ("Couldn't parse ink from '" + line + "'.")
        if class_name not in classnames:
            classnames.append(class_name)
        features = {}
        features["class_index"] = _int64_feature(classnames.index(class_name))
        features["ink"] = _bytes_feature(serialize_array(ink))
        features["height"] = _int64_feature(ink.shape[0])
        features["width"] = _int64_feature(ink.shape[1])
        f = tf.train.Features(feature=features)
        example = tf.train.Example(features=f)
        writers[_pick_output_shard()].write(example.SerializeToString())

    # Close all files
    for w in writers:
        w.close()
    for f in file_handles:
        f.close()
    # Write the class list.
    with tf.io.gfile.GFile(output_file + ".classes", "w") as f:
        for class_name in classnames:
            f.write(class_name + "\n")
    return classnames

if __main__ == '__name__':

    if model_selection == 'cnn' :
        for shard in range(7):
            print(f"Computing shard {shard} out of 10 :")
            X_train, X_test, y_train, y_test, class_names = load_shard('../../raw_data/npy/', shard, test_size=0.2, max_items_per_class=10000)
            X_train = X_train.reshape(-1,28,28,1)
            X_test = X_test.reshape(-1,28,28,1)
            y_train = y_train.reshape(-1,1)
            y_test = y_test.reshape(-1,1)
            write_images_to_tfr_short(X_train, y_train, filename=f"./tfrecords/shard_{shard}_train")
            write_images_to_tfr_short(X_test, y_test, filename=f"./tfrecords/shard_{shard}_test")

        for shard in range(7, 10):
            print(f"Computing shard {shard} out of 10 :")
            X_train, X_test, y_train, y_test, class_names = load_shard('../../raw_data/npy/', shard, test_size=0.2, max_items_per_class=10000)
            X_train = X_train.reshape(-1,28,28,1)
            X_test = X_test.reshape(-1,28,28,1)
            y_train = y_train.reshape(-1,1)
            y_test = y_test.reshape(-1,1)
            write_images_to_tfr_short(X_train, y_train, filename=f"./tfrecords/shard_{shard}_val")
            write_images_to_tfr_short(X_test, y_test, filename=f"./tfrecords/shard_{shard}_test")

    if model_selection == 'rnn' :
        files = glob.glob(os.path.join('../../raw_data/ndjson', '*.ndjson'), recursive=False)
        max_point = 0
        class_names = []
        for file in sorted(files) :
            with open(file, 'r') as f :
                simp_ndjson_lines = f.readlines()
                class_name = json.loads(simp_ndjson_lines[0])['word']
                class_names.append(class_name)
                for i in range(len(simp_ndjson_lines)) :
                    strokes = json.loads(simp_ndjson_lines[i])['drawing']
                    stroke_lengths = [len(stroke[0]) for stroke in strokes]
                    total_points = sum(stroke_lengths)
                    max_point = max(max_point, total_points)

        classnames = convert_data('../../raw_data/ndjson', 80000, "./tfrecords/train.tfrecords", class_names, output_shards=10, offset=0)
        convert_data('../../raw_data/ndjson', 20000, "./tfrecords/test.tfrecords", class_names, output_shards=10, offset=80000)




# main params
format_data = 'npy' # 'npy' or 'tfrecords'
root = '../../raw_data/npy/'
max_items_per_class= 1000
NUM_CLASSES = 10
test_size=0.2

# model params
learning_rate = 0.001
batch_size = 64
patience = 2
epochs = 10
validation_split=0.3

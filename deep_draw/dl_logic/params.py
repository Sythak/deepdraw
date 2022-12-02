import os
from dotenv import load_dotenv, find_dotenv

#env_path = join(dirname(dirname(__file__)), '.env')
env_path = find_dotenv()
load_dotenv(env_path)

model_selection = 'rnn' # 'rnn' or 'cnn'
format_data = 'tfrecords' # 'npy' or 'tfrecords'

# path params
LOCAL_REGISTRY_PATH = os.getenv('LOCAL_REGISTRY_PATH')
root = '../../raw_data/npy'

# storage input data
source_npy = 'quickdraw' # 'local' or 'quickdraw'
storage_tfr = 'local' # 'local' or 'gcs'

# size params
max_items_per_class= 100000
NUM_CLASSES = 10
test_size=0.2
val_size = 0.3

# model params
learning_rate = 0.001
batch_size = 32
patience = 5
epochs = 100
validation_split=0.3

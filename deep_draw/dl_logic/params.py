import os
from dotenv import load_dotenv, find_dotenv

#env_path = join(dirname(dirname(__file__)), '.env')
env_path = find_dotenv()
load_dotenv(env_path)

LOCAL_REGISTRY_PATH = os.getenv('LOCAL_REGISTRY_PATH')

# main params
format_data = 'tfrecords' # 'npy' or 'tfrecords'
root = '../../raw_data/npy'
max_items_per_class= 100000
NUM_CLASSES = 50
test_size=0.2

# model params
learning_rate = 0.001
batch_size = 32
patience = 5
epochs = 100
validation_split=0.3

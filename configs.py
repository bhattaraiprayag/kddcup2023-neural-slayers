import os

# Configuration
SEED = 183
NUM_RECOMMENDATIONS = 100
N_COMPONENTS = 32  # New dimension for reduced embeddings
TASK = 'task1'
LOCALES = ['DE', 'JP', 'UK']
DATA_PATH = 'data/raw/'
OUTPUT_PATH = 'outputs/'
TRAIN_PATH = os.path.join(DATA_PATH, 'train/')
TEST_PATH = os.path.join(DATA_PATH, 'test/')
EMBED_PATH = os.path.join(OUTPUT_PATH, 'embedding/')
INDEX_PATH = os.path.join(OUTPUT_PATH, 'index/')
COMBINED_FEATURES = ['title', 'brand', 'color', 'size', 'model', 'material', 'price', 'desc', 'locale']
BATCH_SIZE = 1024
PROD_DTYPES = {'id':'object', 'locale':'object', 'title':'object', 'price':'float64', 'brand':'object', 'color':'object', 'size':'object', 'model':'object', 'material':'object', 'author':'object', 'desc':'object'}
SESS_DTYPES = {'session_id': 'int32'}
# configs.py

import os

# Configuration
SEED = 183
NUM_RECOMMENDATIONS = 100
N_COMPONENTS = 64  # Dimension of product embeddings
TASK = 'task1'
LOCALES = ['DE', 'JP', 'UK']

DATA_PATH = 'data/raw/'
OUTPUT_PATH = 'outputs/'
TRAIN_PATH = os.path.join(DATA_PATH, 'train/')
TEST_PATH = os.path.join(DATA_PATH, 'test/')
EMBED_PATH = os.path.join(OUTPUT_PATH, 'embedding/')
INDEX_PATH = os.path.join(OUTPUT_PATH, 'index/')
MODELS_PATH = os.path.join(OUTPUT_PATH, 'models/')
NEGATIVE_SAMPLES_PATH = os.path.join(OUTPUT_PATH, 'negative_samples/')
P2P_GRAPH_PATH = os.path.join(OUTPUT_PATH, 'graphs/')

COMBINED_FEATURES = ['title', 'locale', 'brand', 'color', 'price', 'model', 'material', 'desc',]
PROD_DTYPES = {'id':'object', 'locale':'object', 'title':'object', 'price':'float64', 'brand':'object', 'color':'object', 'size':'object', 'model':'object', 'material':'object', 'author':'object', 'desc':'object'}
SESS_DTYPES = {'session_id': 'int32'}


# --- Model & Training Hyperparameters ---
BATCH_SIZE = 1024
EPOCHS = 1
MAX_SESSION_LENGTH = 40
LEARNING_RATE = 0.0001

# Model #1: GRU-based parameters
GRU_HIDDEN_UNITS = 512  # Query Tower (GRU)
GRU_NUM_LAYERS = 10

# Model #2: Transformer-based parameters
NUM_HEADS = 8
ENC_LAYERS = 8
DIM_FFN = N_COMPONENTS * 16


# Negative Sampling Parameters
TOTAL_NEGATIVE_SAMPLES = 5000
HARD_NEGATIVE_RATIO = 0.4
NUM_NEGATIVES = 5000
TRIPLET_MARGIN = 0.75


# Prototyping Parameters
SLICER = 20000
PRED_SLICER = SLICER
# USE_SLICER = USE_PRED_SLICER = True
USE_SLICER = USE_PRED_SLICER = False

# Import python libraries
import os
import ast
import time
import faiss
import psutil
import pickle
import importlib

import numpy as np
import pandas as pd
import multiprocessing as mp

from tqdm import tqdm, TqdmWarning
from collections import Counter
from IPython.display import clear_output
from joblib import Parallel, delayed, dump, load

# from category_encoders import OrdinalEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import TruncatedSVD
from sentence_transformers import SentenceTransformer


# Ignore warnings
import warnings
warnings.filterwarnings('ignore', category=TqdmWarning)

# Housekeeping ==> THIS GENERALLY STAYS THE SAME
SEED, NUM_RECOMMENDATIONS = 183, 100
task, task1_locales = 'task1', ['DE', 'JP', 'UK']
NUM_CORES, NUM_THREADS = psutil.cpu_count(logical=False), psutil.cpu_count(logical=True)
data_path, output_path = 'data/raw/', 'outputs/'
train_path, test_path = data_path + 'train/', data_path + 'test/'
output_file = output_path + task + '_predictions.parquet'
project_files = ['products_train.csv', 'sessions_train.csv', f'sessions_test_{task}.csv']
prod_dtypes = {
    'id': 'object',
    'locale': 'object',
    'title': 'object',
    'price': 'float64',
    'brand': 'object',
    'color': 'object',
    'size': 'object',
    'model': 'object',
    'material': 'object',
    'author': 'object',
    'desc': 'object'
}
sess_dtypes = {
    'session_id': 'int32'
}

# Import project modules
import data_processor
importlib.reload(data_processor)
from data_processor import handle_data

import data_utils
importlib.reload(data_utils)
from data_utils import scale_prices, split_locales, split_sessions

# LOAD DATA HERE ==> THIS GENERALLY STAYS THE SAME
products_train, sessions_train, leaderboard_test = handle_data(
    project_files, [train_path, train_path, test_path],
    task, task1_locales, 1, SEED, prod_dtypes, sess_dtypes
)

# Scale prices
products_train = scale_prices(products_train, task1_locales)

# Split products and sessions by locale
products_de, products_jp, products_uk = split_locales(products_train, task1_locales)
sessions_de, sessions_jp, sessions_uk = split_locales(sessions_train, task1_locales)

from gen_emb import load_locale_embeddings

# Define parameters for embedding generation
combined_features = ['title', 'brand', 'color', 'size', 'model', 'material', 'price', 'desc', 'locale']
batch_size = 1024

# Create a dictionary to map locale strings to their respective product DataFrames
locale_products = {
    'DE': products_de,
    'JP': products_jp,
    'UK': products_uk
}

# Define paths for saving embeddings for each locale
locale_embed_path = {
    'DE': os.path.join(output_path, 'embedding/', 'products_DE.npy'),
    'JP': os.path.join(output_path, 'embedding/', 'products_JP.npy'),
    'UK': os.path.join(output_path, 'embedding/', 'products_UK.npy')
}

# # EXAMPLE
# print(locale_embed_path['DE'])

# Set slicer for smaller datasets
slicer = 1000  # Set to None for full dataset, or a number for a smaller subset
use_slicer = False  # Set to True to use a smaller dataset for testing

# Automate embedding generation for all three locales
embeddings_by_locale = {}
for locale in task1_locales:
    current_products = locale_products[locale]
    embeddings = load_locale_embeddings(
        locale,
        current_products if not use_slicer else current_products.head(slicer),
        combined_features,
        batch_size,
        locale_embed_path
    )
    embeddings_by_locale[locale] = embeddings
    print(f"Embeddings for {locale} generated/loaded with shape: {embeddings.shape}")
print("\nAll locale embeddings processed successfully!")

from sklearn.preprocessing import normalize
def load_embeddings(locale):
    embedding_path = output_path + 'embedding/' + f'products_{locale}.npy'
    embeddings = np.load(embedding_path)
    embeddings = normalize(embeddings)
    return embeddings

def reduce_embeddings(embeddings, n_components):
    svd = TruncatedSVD(n_components=n_components, random_state=SEED)
    embeddings_reduced = svd.fit_transform(embeddings)
    embeddings_reduced = normalize(embeddings_reduced)
    return embeddings_reduced

locale_data = {
    'DE': {
        'load_embeddings': lambda: load_embeddings('DE'),
        'products': eval(f'products_{task1_locales[0].lower()}'),
        'sessions': eval(f'sessions_{task1_locales[0].lower()}')
    },
    'JP': {
        'load_embeddings': lambda: load_embeddings('JP'),
        'products': eval(f'products_{task1_locales[1].lower()}'),
        'sessions': eval(f'sessions_{task1_locales[1].lower()}')
    },
    'UK': {
        'load_embeddings': lambda: load_embeddings('UK'),
        'products': eval(f'products_{task1_locales[2].lower()}'),
        'sessions': eval(f'sessions_{task1_locales[2].lower()}')
    }
}




# --- FAISS Index Integration Start ---

import faiss_index # Import your faiss_index.py script

# Create the directory for FAISS indices if it doesn't exist
faiss_index_dir = os.path.join(output_path, 'index/')
os.makedirs(faiss_index_dir, exist_ok=True)

# Define a dictionary to store paths for locale FAISS indices
locale_faiss_index_paths = {}
for locale in task1_locales:
    locale_faiss_index_paths[locale] = os.path.join(faiss_index_dir, f'products_{locale}.faiss')

# Define FAISS index parameters for Flat index (hyperparameters are not used for Flat index)
index_type = 'Flat'
# For Flat index, hyperparameters like n_list, m, n_bits, m_refine, n_bits_refine are not applicable.
# We'll pass a placeholder or empty tuple.
hyperparameters_flat = (0, 0, 0, 0, 0) # Placeholder

print("\n--- Creating FAISS Flat Indices for Each Locale ---")
for locale in task1_locales:
    print(f"\nProcessing locale: {locale}")
    current_embeddings = embeddings_by_locale[locale]

    # Create and save the FAISS Flat index
    faiss_index_object = faiss_index.create_locale_indices(
        locale,
        embeddings_by_locale, # Pass the dictionary of all embeddings to create_locale_indices
        index_type,
        hyperparameters_flat,
        locale_faiss_index_paths,
        batch_size # Use the batch_size defined earlier
    )

    # Add the FAISS index object to locale_data
    locale_data[locale]['faiss_index'] = faiss_index_object

print("\nAll FAISS Flat indices created and added to locale_data successfully!")

# You can now access the FAISS index for 'DE' for example:
# print(locale_data['DE']['faiss_index'])
# print(locale_data['DE']['faiss_index'].ntotal)

# --- FAISS Index Integration End ---
# Imports
import os
import faiss
import importlib
import numpy as np
import pandas as pd
import psutil
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize
import warnings
from tqdm import TqdmWarning

# Project Modules
import data_processor
import data_utils
import faiss_index
import gen_emb
importlib.reload(data_processor)
importlib.reload(data_utils)
importlib.reload(faiss_index)
importlib.reload(gen_emb)
from data_processor import handle_data
from data_utils import scale_prices, split_locales
from gen_emb import load_locale_embeddings
from faiss_index import create_locale_indices
from configs import (
    SEED, NUM_RECOMMENDATIONS, N_COMPONENTS, TASK, LOCALES,
    DATA_PATH, OUTPUT_PATH, TRAIN_PATH, TEST_PATH,
    EMBED_PATH, INDEX_PATH, COMBINED_FEATURES, BATCH_SIZE,
    PROD_DTYPES, SESS_DTYPES
)

def prepare_locale_artefacts():
    # Configuration
    os.makedirs(EMBED_PATH, exist_ok=True)
    os.makedirs(INDEX_PATH, exist_ok=True)

    # Utility Functions
    def reduce_embeddings(embeddings, n_components, random_state):
        """Reduces embedding dimensionality using TruncatedSVD and normalizes the result."""
        svd = TruncatedSVD(n_components=n_components, random_state=random_state)
        embeddings_reduced = svd.fit_transform(embeddings)
        embeddings_reduced = normalize(embeddings_reduced)
        return embeddings_reduced

    # Data Loading
    project_files = ['products_train.csv', 'sessions_train.csv', f'sessions_test_{TASK}.csv']
    products_train, sessions_train, _ = handle_data(
        project_files, [TRAIN_PATH, TRAIN_PATH, TEST_PATH],
        TASK, LOCALES, 1, SEED, PROD_DTYPES, SESS_DTYPES
    )

    # Preprocessing
    products_train = scale_prices(products_train, LOCALES)
    products_list = split_locales(products_train, LOCALES)
    sessions_list = split_locales(sessions_train, LOCALES)
    products_by_locale = dict(zip(LOCALES, products_list))
    sessions_by_locale = dict(zip(LOCALES, sessions_list))

    # Processing and FAISS Indexing
    locale_data = {}
    # print("\n--- Processing Locales, Reducing Embeddings, and Creating FAISS Indices ---")
    for locale in LOCALES:
        print(f"\nProcessing locale: {locale}")
        
        # Define paths for the current locale
        locale_embed_path = os.path.join(EMBED_PATH, f'products_{locale}.npy')
        locale_faiss_path = os.path.join(INDEX_PATH, f'products_{locale}.faiss')

        # Load original embeddings
        full_embeddings = load_locale_embeddings(
            locale=locale,
            products=products_by_locale[locale],
            combined_features=COMBINED_FEATURES,
            batch_size=BATCH_SIZE,
            locale_embed_path={locale: locale_embed_path}
        )
        # print(f"Loaded full embeddings for {locale} with shape: {full_embeddings.shape}")

        # Reduce embedding dimensionality
        reduced_embeddings = reduce_embeddings(full_embeddings, N_COMPONENTS, SEED)
        # print(f"Reduced embeddings for {locale} to shape: {reduced_embeddings.shape}")
        
        # Create and save FAISS index using reduced embeddings
        embeddings_dict = {locale: reduced_embeddings}
        faiss_paths_dict = {locale: locale_faiss_path}
        
        faiss_index_object = create_locale_indices(
            locale=locale,
            locale_embeddings=embeddings_dict,
            index_type='Flat',
            hyperparameters=(0, 0, 0, 0, 0), # Not used for Flat index
            index_files=faiss_paths_dict,
            batch_size=BATCH_SIZE
        )
        # print(f"FAISS index for {locale} created with {faiss_index_object.ntotal} vectors.")

        # Store all data for the locale
        locale_data[locale] = {
            'products': products_by_locale[locale],
            'sessions': sessions_by_locale[locale],
            'embeddings': reduced_embeddings,
            'faiss_index': faiss_index_object
        }

    print("\n--- All locales processed successfully ---")
    return locale_data


if __name__ == "__main__":
    # Suppress TqdmWarning for cleaner output
    warnings.filterwarnings("ignore", category=TqdmWarning)
    
    # Run the main function
    locale_data = prepare_locale_artefacts()
    
    # Example of accessing the generated data for a locale
    print(f"\nData for DE locale: {locale_data['DE'].keys()}")
    print(f"Products shape: {locale_data['DE']['products'].shape}")
    print(f"Embeddings shape: {locale_data['DE']['embeddings'].shape}")
    print(f"FAISS index total: {locale_data['DE']['faiss_index'].ntotal}")

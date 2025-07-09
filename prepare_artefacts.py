# prepare_artefacts.py

import importlib
import os
import pickle
import warnings

import faiss
import numpy as np
import pandas as pd
import psutil
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize
from tqdm import TqdmWarning

import data_processor
import data_utils
import faiss_index
import gen_emb

importlib.reload(data_processor)
importlib.reload(data_utils)
importlib.reload(faiss_index)
importlib.reload(gen_emb)
from configs import (
    BATCH_SIZE, COMBINED_FEATURES, DATA_PATH, EMBED_PATH, P2P_GRAPH_PATH, INDEX_PATH, LOCALES, N_COMPONENTS, NUM_RECOMMENDATIONS, OUTPUT_PATH, PRED_SLICER, PROD_DTYPES, SEED, SESS_DTYPES, SLICER, TASK, TEST_PATH, TRAIN_PATH, USE_PRED_SLICER, USE_SLICER
)
from data_processor import handle_data
from data_utils import scale_prices, split_locales
from faiss_index import create_locale_indices
from gen_emb import load_locale_embeddings
from session_graph_builder import ProductGraphBuilder


def reduce_embeddings(embeddings, n_components, random_state):
    svd = TruncatedSVD(n_components=n_components, random_state=random_state)
    embeddings_reduced = svd.fit_transform(embeddings)
    embeddings_reduced = normalize(embeddings_reduced)
    return embeddings_reduced


def prepare_locale_artefacts():
    os.makedirs(EMBED_PATH, exist_ok=True)
    os.makedirs(INDEX_PATH, exist_ok=True)

    project_files = ['products_train.csv', 'sessions_train.csv', f'sessions_test_{TASK}.csv']
    products_train, sessions_train, sessions_test = handle_data(
        project_files, [TRAIN_PATH, TRAIN_PATH, TEST_PATH],
        TASK, LOCALES, 1, SEED, PROD_DTYPES, SESS_DTYPES
    )

    products_train = scale_prices(products_train, LOCALES)
    products_list = split_locales(products_train, LOCALES)
    sessions_list = split_locales(sessions_train, LOCALES)
    sessions_test_list = split_locales(sessions_test, LOCALES)

    if USE_SLICER:
        products_list = [products[:SLICER] for products in products_list]
        sessions_list = [sessions[:SLICER] for sessions in sessions_list]
        sessions_test_list = [sessions_test[:SLICER] for sessions_test in sessions_test_list]
    products_by_locale = dict(zip(LOCALES, products_list))
    sessions_by_locale = dict(zip(LOCALES, sessions_list))
    sessions_test_by_locale = dict(zip(LOCALES, sessions_test_list))

    locale_data = {}
    for locale in LOCALES:
        locale_embed_path = os.path.join(EMBED_PATH, f'products_{locale}.npy')
        locale_faiss_path = os.path.join(INDEX_PATH, f'products_{locale}.faiss')
        locale_p2p_graph_path = os.path.join(P2P_GRAPH_PATH, f'graph_pmi_{locale}.gpickle')

        full_embeddings = load_locale_embeddings(
            locale=locale,
            products=products_by_locale[locale],
            combined_features=COMBINED_FEATURES,
            batch_size=BATCH_SIZE,
            locale_embed_path={locale: locale_embed_path}
        )
        id2embidx_path = locale_embed_path.replace('.npy', '_id2embidx.pkl')
        os.makedirs(os.path.dirname(id2embidx_path), exist_ok=True)
        with open(id2embidx_path, 'rb') as f:
            prod_id_to_emb_idx_map = pickle.load(f)

        reduced_embeddings = reduce_embeddings(full_embeddings, N_COMPONENTS, SEED)

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

        if not os.path.exists(locale_p2p_graph_path):
            graph_builder = ProductGraphBuilder(sessions_by_locale[locale])
            p2p_graph = graph_builder.build_graph(
                weight_type='pmi',
                num_workers=-1,
                session_slice=SLICER if USE_SLICER else None
            )
            ProductGraphBuilder.save_graph(p2p_graph, locale_p2p_graph_path)
        else:
            p2p_graph = ProductGraphBuilder.load_graph(locale_p2p_graph_path)

        locale_data[locale] = {
            'products': products_by_locale[locale],
            'sessions': sessions_by_locale[locale],
            'sessions_test': sessions_test_by_locale[locale],
            'embeddings': reduced_embeddings,
            'prod_id_to_emb_idx_map': prod_id_to_emb_idx_map,
            'faiss_index': faiss_index_object,
            'p2p_graph': p2p_graph
        }
    return locale_data


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=TqdmWarning)

    locale_data = prepare_locale_artefacts()

    print(f"\nData for DE locale: {locale_data['DE'].keys()}")
    print(f"Products shape: {locale_data['DE']['products'].shape}")
    print(f"Sessions shape: {locale_data['DE']['sessions'].shape}")
    print(f"Test Sessions shape: {locale_data['DE']['sessions_test'].shape}")
    print(f"Embeddings shape: {locale_data['DE']['embeddings'].shape}")
    print(f"FAISS index total: {locale_data['DE']['faiss_index'].ntotal}")

    # # Check first item of the embeddings
    # print(f"First embedding for DE locale: {locale_data['DE']['embeddings'][0]}")
    # Check first 10 similar items of the first product embedding
    first_product_id = locale_data['DE']['products']['id'].iloc[0]
    first_product_emb_idx = locale_data['DE']['prod_id_to_emb_idx_map'][first_product_id]
    first_product_emb_idx = locale_data['DE']['prod_id_to_emb_idx_map'][first_product_id]
    distances, indices = locale_data['DE']['faiss_index'].search(
        np.expand_dims(locale_data['DE']['embeddings'][first_product_emb_idx], axis=0), 10
    )
    print(f"First 10 similar items for product ID {first_product_id}:")
    for idx, dist in zip(indices[0], distances[0]):
        if idx != first_product_emb_idx:
            similar_product_id_idx = int(idx)
            emb_idx_to_prod_id_map = {v: k for k, v in locale_data['DE']['prod_id_to_emb_idx_map'].items()}
            if similar_product_id_idx in emb_idx_to_prod_id_map:
                similar_product_id = emb_idx_to_prod_id_map[similar_product_id_idx]
                print(f"Product ID: {similar_product_id}, Distance: {dist:.4f}")
            else:
                print(f"Product ID not found for index {similar_product_id_idx}")

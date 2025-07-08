# gen_emb.py

import os
import faiss
import pickle

import numpy as np

from sentence_transformers import SentenceTransformer
from tqdm import tqdm


def create_prod_embeddings(products_train, combined_features, output_path, batch_size):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
    embeddings = []
    prod_id_to_emb_idx_map = {}
    texts_to_embed = products_train[combined_features].astype(str).agg(' '.join, axis=1).tolist()
    product_ids = products_train['id'].astype(str).tolist()
    num_batches = len(texts_to_embed) // batch_size + (1 if len(texts_to_embed) % batch_size != 0 else 0)
    global_idx = 0
    for i in tqdm(range(num_batches), desc="Creating Embeddings..."):
        start = i * batch_size
        end = min((i + 1) * batch_size, len(texts_to_embed))
        batch_texts = texts_to_embed[start:end]
        batch_embeddings = model.encode(batch_texts, show_progress_bar=False, batch_size=batch_size)
        embeddings.append(batch_embeddings)
        for j in range(start, end):
            prod_id_to_emb_idx_map[product_ids[j]] = global_idx
            global_idx += 1
    np.save(output_path, np.vstack(embeddings))
    map_output_path = output_path.replace('.npy', '_id2embidx.pkl')
    with open(map_output_path, 'wb') as f:
        pickle.dump(prod_id_to_emb_idx_map, f)


def load_locale_embeddings(locale, products, combined_features, batch_size, locale_embed_path):
    npy_path = locale_embed_path[locale]
    pkl_path = npy_path.replace('.npy', '_id2embidx.pkl')
    if os.path.exists(locale_embed_path[locale]) and os.path.exists(pkl_path):
        embeddings = np.load(npy_path)
        faiss.normalize_L2(embeddings)
    else:
        create_prod_embeddings(products, combined_features, npy_path, batch_size)
        embeddings = np.load(npy_path)
        faiss.normalize_L2(embeddings)
    return embeddings

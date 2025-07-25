# faiss_index.py

import os

import faiss
import numpy as np
from tqdm import tqdm


def load_index(index_path):
    return faiss.read_index(index_path)


def train_index(index_path, data):
    index = faiss.read_index(index_path)
    index.train(data)
    faiss.write_index(index, index_path)
    return index


def initiate_index(embeddings, index_type, hyperparameters, output_path):
    d = embeddings.shape[1]
    index = faiss.IndexFlatL2(d)
    n_list, m, n_bits, m_refine, n_bits_refine = hyperparameters
    if index_type == 'Flat':
        faiss.write_index(index, output_path)
    elif index_type == 'IVFFlat':
        quantizer = index
        index = faiss.IndexIVFFlat(quantizer, d, n_list)
        faiss.write_index(index, output_path)
    elif index_type == 'IVFPQ':
        quantizer = index
        index = faiss.IndexIVFPQ(quantizer, d, n_list, m, n_bits)
        faiss.write_index(index, output_path)
    elif index_type == 'IVFPQR':
        quantizer = index
        index = faiss.IndexIVFPQR(quantizer, d, n_list, m, n_bits, m_refine, n_bits_refine)
        faiss.write_index(index, output_path)
    return index


def add_batches_to_index(index_path, embeddings, batch_size):
    index = faiss.read_index(index_path)
    num_batches = len(embeddings) // batch_size + (1 if len(embeddings) % batch_size != 0 else 0)
    for i in tqdm(range(num_batches), desc="Adding batches to FAISS index..."):
        start = i * batch_size
        end = min((i + 1) * batch_size, len(embeddings))
        batch_embeddings = embeddings[start:end]
        index.add(batch_embeddings)
        faiss.write_index(index, index_path)
    return index


def create_locale_indices(locale, locale_embeddings, index_type, hyperparameters, index_files, batch_size):
    if os.path.exists(index_files[locale]):
        index = load_index(index_files[locale])
    else:
        index = initiate_index(locale_embeddings[locale], index_type, hyperparameters, index_files[locale])
        index = train_index(index_files[locale], locale_embeddings[locale])
        index = add_batches_to_index(index_files[locale], locale_embeddings[locale], batch_size)
    return index

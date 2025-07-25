# gen_emb_new.py

import os
import pickle
import faiss
import torch

import numpy as np

from tqdm import tqdm
from transformers import BertModel, BertTokenizer


def create_prod_embeddings(products_train, combined_features, output_path, batch_size):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-uncased')
    model = BertModel.from_pretrained('bert-base-multilingual-uncased')
    model.to(device)
    model.eval()    
    prod_id_to_emb_idx_map = {}
    texts_to_embed = products_train[combined_features].astype(str).agg(' '.join, axis=1).tolist()
    product_ids = products_train['id'].astype(str).tolist()    
    num_batches = len(texts_to_embed) // batch_size + (1 if len(texts_to_embed) % batch_size != 0 else 0)
    global_idx = 0    
    for i in tqdm(range(num_batches), desc="Creating Embeddings..."):
        start = i * batch_size
        end = min((i + 1) * batch_size, len(texts_to_embed))
        batch_texts = texts_to_embed[start:end]        
        with torch.no_grad():
            encoded_input = tokenizer(
                batch_texts, 
                padding=True, 
                truncation=True, 
                return_tensors='pt'
            )
            encoded_input = {k: v.to(device) for k, v in encoded_input.items()}
            output = model(**encoded_input)
            batch_embeddings = output.pooler_output.cpu().numpy()
        if i == 0:
            np.save(output_path, batch_embeddings)
        else:
            existing_embeddings = np.load(output_path)
            updated_embeddings = np.vstack([existing_embeddings, batch_embeddings])
            np.save(output_path, updated_embeddings)
        for j in range(start, end):
            prod_id_to_emb_idx_map[product_ids[j]] = global_idx
            global_idx += 1
    map_output_path = output_path.replace('.npy', '_id2embidx.pkl')
    with open(map_output_path, 'wb') as f:
        pickle.dump(prod_id_to_emb_idx_map, f)


def load_locale_embeddings(locale, products, combined_features, batch_size, locale_embed_path):
    npy_path = locale_embed_path[locale]
    pkl_path = npy_path.replace('.npy', '_id2embidx.pkl')
    if os.path.exists(locale_embed_path[locale]) and os.path.exists(pkl_path):
        embeddings = np.load(npy_path)
        faiss.normalize_L2(embeddings)
        with open(pkl_path, 'rb') as f:
            prod_id_to_emb_idx_map = pickle.load(f)
    else:
        create_prod_embeddings(products, combined_features, npy_path, batch_size)
        embeddings = np.load(npy_path)
        faiss.normalize_L2(embeddings)
        with open(pkl_path, 'rb') as f:
            prod_id_to_emb_idx_map = pickle.load(f)
    return embeddings, prod_id_to_emb_idx_map

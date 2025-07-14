# negative_sampler.py

import os
import pickle
from concurrent.futures import ThreadPoolExecutor
from functools import partial

import faiss
import numpy as np
import psutil
from tqdm import tqdm

from configs import HARD_NEGATIVE_RATIO, TOTAL_NEGATIVE_SAMPLES

NUM_HARD_NEGATIVES = int(TOTAL_NEGATIVE_SAMPLES * HARD_NEGATIVE_RATIO)
NUM_RANDOM_NEGATIVES = TOTAL_NEGATIVE_SAMPLES - NUM_HARD_NEGATIVES

def _generate_for_product(
    product_id, faiss_index, prod_id_to_emb_idx_map,
    emb_idx_to_prod_id, all_indices_array
):
    try:
        query_idx = prod_id_to_emb_idx_map.get(product_id)
        if query_idx is None:
            return None, None # Product not in catalog
        query_vector = np.expand_dims(faiss_index.reconstruct(int(query_idx)), axis=0).astype('float32')
        _, hard_indices = faiss_index.search(query_vector, NUM_HARD_NEGATIVES + 1)
        hard_indices = hard_indices[0]
        final_negatives_indices = {idx for idx in hard_indices if idx != query_idx}
        num_random_needed = TOTAL_NEGATIVE_SAMPLES - len(final_negatives_indices)
        if num_random_needed > 0:
            random_candidates = np.random.choice(all_indices_array, size=num_random_needed + len(final_negatives_indices) + 1, replace=False)
            valid_random = [idx for idx in random_candidates if idx != query_idx and idx not in final_negatives_indices]
            final_negatives_indices.update(valid_random[:num_random_needed])
        negative_product_ids = [emb_idx_to_prod_id[idx] for idx in list(final_negatives_indices)]
        return product_id, negative_product_ids[:TOTAL_NEGATIVE_SAMPLES]
    except Exception as e:
        print(f"Error processing product {product_id}: {e}")
        return None, None

def create_negative_samples_for_locale(locale, sessions_df, faiss_index, prod_id_to_emb_idx_map, output_path):
    os.makedirs(output_path, exist_ok=True)
    output_file = os.path.join(output_path, f'negative_samples_{locale}.pkl')
    if os.path.exists(output_file) and os.path.getsize(output_file) > 10:
        # print(f"[{locale}] Negative samples file already exists. Skipping creation.")
        return
    target_product_ids = sessions_df['next_item'].unique()
    valid_target_ids = [pid for pid in target_product_ids if pid in prod_id_to_emb_idx_map]
    if not valid_target_ids:
        print(f"[{locale}] No valid 'next_item' products found. Skipping creation.")
        return
    emb_idx_to_prod_id = {v: k for k, v in prod_id_to_emb_idx_map.items()}
    all_indices_array = np.array(list(emb_idx_to_prod_id.keys()), dtype=np.int32)
    negative_samples_map = {}
    num_workers = psutil.cpu_count(logical=True)
    # reduce_workers = 0.5
    # num_workers = max(1, int(num_workers * reduce_workers) if reduce_workers < 1 else max(1, int(num_workers)))
    print(f"[{locale}] Using {num_workers} CPU cores for parallel processing.")

    worker_fn = partial(_generate_for_product, 
                        faiss_index=faiss_index, 
                        prod_id_to_emb_idx_map=prod_id_to_emb_idx_map,
                        emb_idx_to_prod_id=emb_idx_to_prod_id,
                        all_indices_array=all_indices_array)
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        results = list(tqdm(executor.map(worker_fn, valid_target_ids), total=len(valid_target_ids), desc=f"[{locale}] Generating session negatives..."))
    for product_id, negatives in results:
        if product_id and negatives:
            negative_samples_map[product_id] = negatives
    with open(output_file, 'wb') as f:
        pickle.dump(negative_samples_map, f)

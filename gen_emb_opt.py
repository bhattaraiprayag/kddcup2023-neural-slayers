# gen_emb_opt.py

import json
import os
import pickle
from typing import Dict, List, Tuple

import faiss
import numpy as np
import torch
from tqdm import tqdm
from transformers import BertModel, BertTokenizerFast


def _get_device() -> str:
    return os.getenv("EMB_DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")


def _init_model_and_tokeniser(device: str):
    tokenizer = BertTokenizerFast.from_pretrained(
        "bert-base-multilingual-uncased", use_fast=True
    )
    model = BertModel.from_pretrained("bert-base-multilingual-uncased")
    model.to(device)
    model.eval()
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")
    return tokenizer, model


def _open_memmap(path: str, shape: Tuple[int, int], dtype="float32", allow_resume=True):
    mode = "r+" if (allow_resume and os.path.exists(path)) else "w+"
    return np.memmap(path, dtype=dtype, mode=mode, shape=shape), mode == "r+"


def _progress_path(array_path: str) -> str:
    return f"{array_path}.progress"


def _read_progress(progress_file: str) -> int:
    try:
        with open(progress_file) as f:
            return int(f.read().strip())
    except FileNotFoundError:
        return 0


def _write_progress(progress_file: str, batch_idx: int):
    with open(progress_file, "w") as f:
        f.write(str(batch_idx))


def create_prod_embeddings(
    products_train,
    combined_features: List[str],
    output_path: str,
    batch_size: int,
):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    device = _get_device()
    print(f"[gen_emb_opt] Using device: {device}")
    tokenizer, model = _init_model_and_tokeniser(device)
    texts_to_embed = (
        products_train[combined_features].astype("string").agg(" ".join, axis=1).tolist()
    )
    product_ids = products_train["id"].astype("string").tolist()
    n_rows = len(texts_to_embed)
    hidden = model.config.hidden_size
    num_batches = (n_rows + batch_size - 1) // batch_size
    emb_mmap, resuming = _open_memmap(output_path, shape=(n_rows, hidden))
    progress_file = _progress_path(output_path)
    start_batch = _read_progress(progress_file) if resuming else 0
    if resuming:
        print(
            f"[gen_emb_opt] Resuming from batch {start_batch}/{num_batches} "
            f"(≈ {start_batch * batch_size} / {n_rows} rows)"
        )
    else:
        print(f"[gen_emb_opt] Fresh run – will write {n_rows} rows")
    for batch_idx in tqdm(
        range(start_batch, num_batches), desc="Creating embeddings", unit="batch"
    ):
        s = batch_idx * batch_size
        e = min((batch_idx + 1) * batch_size, n_rows)
        batch_texts = texts_to_embed[s:e]
        encoded = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        encoded = {k: v.to(device) for k, v in encoded.items()}
        with torch.cuda.amp.autocast(dtype=torch.float16):
            with torch.no_grad():
                out = model(**encoded)
        emb_mmap[s:e] = out.pooler_output.float().cpu().numpy()
        emb_mmap.flush()
        _write_progress(progress_file, batch_idx + 1)
    if os.path.exists(progress_file):
        os.remove(progress_file)
    map_output_path = output_path.replace(".npy", "_id2embidx.pkl")
    prod_id_to_emb_idx_map = dict(zip(product_ids, np.arange(n_rows, dtype=int)))
    with open(map_output_path, "wb") as f:
        pickle.dump(prod_id_to_emb_idx_map, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"[gen_emb_opt] Done. Wrote {n_rows} vectors → {output_path}")


def load_locale_embeddings(
    locale: str,
    products_df,
    text_features: List[str],
    batch_size: int,
    locale_embed_path: Dict[str, str],
):
    npy_path = locale_embed_path[locale]
    pkl_path = npy_path.replace(".npy", "_id2embidx.pkl")
    if not os.path.exists(pkl_path):
        create_prod_embeddings(
            products_train=products_df,
            combined_features=text_features,
            output_path=npy_path,
            batch_size=batch_size,
        )
    embeddings = np.memmap(npy_path, dtype='float32', mode='r+', shape=(len(products_df), 768))
    faiss.normalize_L2(embeddings)
    with open(pkl_path, "rb") as f:
        prod_id_to_emb_idx_map = pickle.load(f)
    return embeddings, prod_id_to_emb_idx_map

# train_lightning.py

import importlib
import os
import pickle

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

torch.set_float32_matmul_precision('medium')

import configs
import dataset
import faiss_index
import negative_sampler
import prepare_artefacts
import utils
from configs import (BATCH_SIZE, DIM_FFN, ENC_LAYERS,
                     EPOCHS, LEARNING_RATE, LOCALES,
                     MAX_SESSION_LENGTH, MODELS_PATH,
                     NEGATIVE_SAMPLES_PATH, NUM_HEADS, NUM_NEGATIVES,
                     NUM_RECOMMENDATIONS, SEED, TRIPLET_MARGIN)
from lightning_model import LightningTwoTower
from negative_sampler import create_negative_samples_for_locale
from utils import model_tuner

importlib.reload(configs)
importlib.reload(prepare_artefacts)
importlib.reload(negative_sampler)
importlib.reload(dataset)
importlib.reload(utils)
importlib.reload(faiss_index)


def main():
    pl.seed_everything(SEED, workers=True)

    print("\n--- Preparing artefacts (data, embeddings, FAISS indices) ---")
    all_locale_data = prepare_artefacts.prepare_locale_artefacts()

    print("\n--- Generating/retrieving negative samples ---")
    for locale in LOCALES:
        create_negative_samples_for_locale(
            locale,
            all_locale_data[locale]['sessions'],
            all_locale_data[locale]['faiss_index'],
            all_locale_data[locale]['prod_id_to_emb_idx_map'],
            NEGATIVE_SAMPLES_PATH
        )

    all_predictions = []
    print("\n--- Training a model and predicting for each locale ---")
    for locale in LOCALES:
        print(f"\n===== Processing locale: {locale} =====")
        locale_data = all_locale_data[locale]

        enhanced_embed_path = os.path.join(configs.EMBED_PATH, f'enhanced_embeddings_{locale}.npy')
        if os.path.exists(enhanced_embed_path):
            print(f"Found and loading GNN-enhanced embeddings from {enhanced_embed_path}")
            locale_data['embeddings'] = np.load(enhanced_embed_path)
            locale_faiss_path = os.path.join(configs.INDEX_PATH, f'products_{locale}_enhanced.faiss')
            if os.path.exists(locale_faiss_path):
                print(f"Loading existing FAISS index from {locale_faiss_path}")
                faiss_index_object = faiss_index.load_index(
                    locale_faiss_path
                )
                locale_data['faiss_index'] = faiss_index_object
            else:
                faiss_index_object = faiss_index.create_locale_indices(
                    locale=locale,
                    locale_embeddings={locale: locale_data['embeddings']},
                    index_type='Flat',
                    hyperparameters=(0, 0, 0, 0, 0),
                    index_files={locale: locale_faiss_path},
                    batch_size=configs.BATCH_SIZE
                )
                locale_data['faiss_index'] = faiss_index_object
                print(f"New FAISS index created with {locale_data['faiss_index'].ntotal} vectors.")
        else:
            print("GNN-enhanced embeddings not found. Using SVD-reduced embeddings as fallback.")

        product_df = locale_data['products']
        all_product_ids = product_df['id'].tolist()
        id_to_idx = {pid: i for i, pid in enumerate(all_product_ids)}
        emb_idx_to_id = {v: k for k, v in locale_data['prod_id_to_emb_idx_map'].items()}

        neg_samples_path = os.path.join(NEGATIVE_SAMPLES_PATH, f'negative_samples_{locale}.pkl')
        with open(neg_samples_path, 'rb') as f:
            neg_samples_map = pickle.load(f)

        embedding_dim = locale_data['embeddings'].shape[1]
        precomputed_embeddings = torch.FloatTensor(locale_data['embeddings'])

        model = LightningTwoTower(
            embedding_dim=embedding_dim, nhead=NUM_HEADS, num_encoder_layers=ENC_LAYERS,
            dim_feedforward=DIM_FFN, precomputed_embeddings=precomputed_embeddings,
            learning_rate=LEARNING_RATE, triplet_margin=TRIPLET_MARGIN,
            train_sessions_df=locale_data['sessions'], id_to_idx=locale_data['prod_id_to_emb_idx_map'],
            neg_samples_map=neg_samples_map, max_session_length=MAX_SESSION_LENGTH,
            num_negatives=NUM_NEGATIVES, batch_size=BATCH_SIZE
        )

        trainer = pl.Trainer(max_epochs=EPOCHS, accelerator="auto", devices="auto", strategy="auto")
        tuned_model = model_tuner(model=model, pl_trainer=trainer, orig_lr=LEARNING_RATE, init_bs=BATCH_SIZE)

        print(f"[{locale}] Training model for {EPOCHS} epochs...")
        trainer.fit(tuned_model)

        os.makedirs(MODELS_PATH, exist_ok=True)
        model_save_path = os.path.join(MODELS_PATH, f'query_tower_{locale}.pt')
        torch.save(model.query_tower.state_dict(), model_save_path)

        print(f"[{locale}] Generating predictions...")
        query_tower = model.query_tower.to('cpu')
        query_tower.eval()
        test_sessions_df = locale_data['sessions_test']
        test_sessions = test_sessions_df['prev_items'].str.split(',').tolist()

        locale_predictions = []
        with torch.no_grad():
            for session in tqdm(test_sessions, desc=f"[{locale}] Predicting"):
                session_indices = [locale_data['prod_id_to_emb_idx_map'].get(item, 0) for item in session]
                if len(session_indices) > MAX_SESSION_LENGTH:
                    session_indices = session_indices[-MAX_SESSION_LENGTH:]
                else:
                    session_indices = [0] * (MAX_SESSION_LENGTH - len(session_indices)) + session_indices
                session_tensor = torch.tensor([session_indices], dtype=torch.long)
                session_emb = query_tower(session_tensor).cpu().numpy()
                _, I = locale_data['faiss_index'].search(session_emb, NUM_RECOMMENDATIONS)
                predicted_product_ids = [emb_idx_to_id[i] for i in I[0] if i in emb_idx_to_id]
                locale_predictions.append(predicted_product_ids)

        all_predictions.extend([(idx, preds) for idx, preds in zip(test_sessions_df.index, locale_predictions)])

    print("\n--- Combining all predictions and saving submission file ---")
    if not all_predictions:
        print("No predictions were generated. Exiting.")
        return

    all_predictions.sort(key=lambda x: x[0])
    sorted_predictions = [p[1] for p in all_predictions]
    
    submission_df = pd.DataFrame({'next_item_prediction': sorted_predictions})
    submission_file = os.path.join(configs.OUTPUT_PATH, 'submission.parquet')
    submission_df.to_parquet(submission_file)
    print(f"Submission file saved to: {submission_file}")


if __name__ == '__main__':
    main()

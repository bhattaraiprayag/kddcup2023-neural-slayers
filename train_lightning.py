# train_lightning.py

import os
import torch
import pickle
import importlib
import numpy as np
import pandas as pd
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from tqdm import tqdm

torch.set_float32_matmul_precision('medium')

import configs
importlib.reload(configs)
from configs import (
    SEED, NUM_RECOMMENDATIONS, N_COMPONENTS, TASK, LOCALES,
    MODELS_PATH, NEGATIVE_SAMPLES_PATH,
    BATCH_SIZE, EPOCHS, MAX_SESSION_LENGTH, LEARNING_RATE,
    GRU_HIDDEN_UNITS, GRU_NUM_LAYERS,   # Model #1
    NUM_HEADS, ENC_LAYERS, DIM_FFN, # Model #2
    TOTAL_NEGATIVE_SAMPLES, HARD_NEGATIVE_RATIO, NUM_NEGATIVES, TRIPLET_MARGIN,
    SLICER, PRED_SLICER, USE_SLICER, USE_PRED_SLICER
)

import prepare_artefacts
import negative_sampler
import dataset
importlib.reload(prepare_artefacts)
importlib.reload(negative_sampler)
importlib.reload(dataset)

from negative_sampler import create_negative_samples_for_locale
from dataset import SessionDataset
from lightning_model import LightningTwoTower


def main():
    pl.seed_everything(SEED, workers=True)

    print("\n--- Preparing artefacts (data, embeddings, FAISS indices) ---")
    all_locale_data = prepare_artefacts.prepare_locale_artefacts()

    print("\n--- Generating/retrieving negative samples ---")
    for locale in LOCALES:
        create_negative_samples_for_locale(
            locale,
            all_locale_data[locale]['sessions'] if not USE_SLICER else all_locale_data[locale]['sessions'][:SLICER],
            all_locale_data[locale]['faiss_index'],
            all_locale_data[locale]['prod_id_to_emb_idx_map'],
            NEGATIVE_SAMPLES_PATH
        )

    all_predictions = []
    print("\n--- Training a model and predicting for each locale ---")
    for locale in LOCALES:
        print(f"\n===== Processing locale: {locale} =====")
        locale_data = all_locale_data[locale]
        product_df = locale_data['products']
        product_ids = product_df['id'].values
        id_to_idx = {pid: i for i, pid in enumerate(product_ids)}
        idx_to_id = {i: pid for i, pid in enumerate(product_ids)}

        neg_samples_path = os.path.join(NEGATIVE_SAMPLES_PATH, f'negative_samples_{locale}.pkl')
        if not os.path.exists(neg_samples_path):
            print(f"Warning: Negative samples file not found for locale {locale}. Skipping training.")
            continue
        with open(neg_samples_path, 'rb') as f:
            neg_samples_map = pickle.load(f)

        train_dataset = SessionDataset(locale_data['sessions'], id_to_idx, neg_samples_map, MAX_SESSION_LENGTH, NUM_NEGATIVES)
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=os.cpu_count())

        embedding_dim = locale_data['embeddings'].shape[1]
        precomputed_embeddings = torch.FloatTensor(locale_data['embeddings'])

        model = LightningTwoTower(
            embedding_dim=embedding_dim,
            nhead=NUM_HEADS,
            num_encoder_layers=ENC_LAYERS,
            dim_feedforward=DIM_FFN,
            precomputed_embeddings=precomputed_embeddings,
            learning_rate=LEARNING_RATE,
            triplet_margin=TRIPLET_MARGIN,
            num_negatives=NUM_NEGATIVES
        )

        # # # DEBUG:
        # trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        # total_params = sum(p.numel() for p in model.parameters())
        # print(f"Total parameters: {total_params}")
        # print(f"Trainable parameters: {trainable_params}, {trainable_params / total_params * 100:.2f}%")

        trainer = pl.Trainer(
            max_epochs=EPOCHS,
            accelerator="auto",
            devices="auto",
            strategy="auto",
            logger=True,
        )

        print(f"[{locale}] Training model for {EPOCHS} epochs...")
        trainer.fit(model, train_loader)
        
        os.makedirs(MODELS_PATH, exist_ok=True)
        model_save_path = os.path.join(MODELS_PATH, f'query_tower_{locale}.pt')
        torch.save(model.query_tower.state_dict(), model_save_path)

        # PREDICTION
        print(f"[{locale}] Generating predictions...")
        query_tower = model.query_tower.to('cpu')
        query_tower.eval()
        test_sessions_df = locale_data['sessions_test']
        test_sessions = test_sessions_df['prev_items'].str.split(',').tolist()
        
        if USE_PRED_SLICER:
            test_sessions = test_sessions[:PRED_SLICER]

        locale_predictions = []
        with torch.no_grad():
            for session in tqdm(test_sessions, desc=f"[{locale}] Predicting"):
                session_indices = [id_to_idx.get(item, 0) for item in session]
                if len(session_indices) > MAX_SESSION_LENGTH:
                    session_indices = session_indices[-MAX_SESSION_LENGTH:]
                else:
                    session_indices = [0] * (MAX_SESSION_LENGTH - len(session_indices)) + session_indices
                
                session_tensor = torch.tensor([session_indices], dtype=torch.long)
                session_emb = query_tower(session_tensor).cpu().numpy()
                _, I = locale_data['faiss_index'].search(session_emb, NUM_RECOMMENDATIONS)
                predicted_product_ids = [idx_to_id[i] for i in I[0] if i in idx_to_id]
                locale_predictions.append(predicted_product_ids)

        all_predictions.extend([(idx, preds, locale) for idx, preds in zip(test_sessions_df.index, locale_predictions)])

    print("\n--- Combining all predictions and saving submission file ---")
    if not all_predictions:
        print("No predictions were generated. Exiting.")
        return
    
    all_predictions.sort(key=lambda x: x[0])
    sorted_predictions = [p[1] for p in all_predictions]
    
    if USE_PRED_SLICER:
        original_test_df = pd.concat([all_locale_data[loc]['sessions_test'][:PRED_SLICER] for loc in LOCALES]).sort_index()
    else:
        original_test_df = pd.concat([all_locale_data[loc]['sessions_test'] for loc in LOCALES]).sort_index()
    
    locales_sorted = [p[2] for p in all_predictions]
    submission_df = pd.DataFrame({
        'locale': locales_sorted,
        'next_item_prediction': sorted_predictions
    }, index=original_test_df.index)

    submission_file = os.path.join(configs.OUTPUT_PATH, 'submission.parquet')
    submission_df.to_parquet(submission_file)
    print(f"Submission file saved to: {submission_file}")
    print(f"Submission file shape: {submission_df.shape}")

if __name__ == '__main__':
    main()

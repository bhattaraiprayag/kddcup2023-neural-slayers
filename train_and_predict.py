# train_and_predict.py

import os
import torch
import pickle
import importlib
import numpy as np
import pandas as pd
import torch.nn as nn
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
import model
import dataset
importlib.reload(prepare_artefacts)
importlib.reload(negative_sampler)
importlib.reload(model)
importlib.reload(dataset)

from negative_sampler import create_negative_samples_for_locale
from model import TwoTowerModel
from dataset import SessionDataset


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
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
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

        num_items = len(product_ids)
        embedding_dim = locale_data['embeddings'].shape[1]
        precomputed_embeddings = torch.FloatTensor(locale_data['embeddings'])
        # two_tower_model = TwoTowerModel(  # Model #1
        #     num_items=num_items,
        #     embedding_dim=embedding_dim,
        #     hidden_dim=GRU_HIDDEN_UNITS,
        #     n_layers=GRU_NUM_LAYERS,
        #     precomputed_embeddings=precomputed_embeddings
        # ).to(device)
        two_tower_model = TwoTowerModel(    # Model #2
            embedding_dim=embedding_dim,
            nhead=NUM_HEADS,
            num_encoder_layers=ENC_LAYERS,
            dim_feedforward=DIM_FFN,
            precomputed_embeddings=precomputed_embeddings
        ).to(device)
        optimizer = torch.optim.AdamW(two_tower_model.parameters(), lr=LEARNING_RATE)
        loss_fn = nn.TripletMarginLoss(margin=TRIPLET_MARGIN)
        print(two_tower_model)

        trainable_params = sum(p.numel() for p in two_tower_model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in two_tower_model.parameters())
        print(f"Total parameters: {total_params}")
        print(f"Trainable parameters: {trainable_params}, {trainable_params / total_params * 100:.2f}%")

        # TRAINING
        print(f"[{locale}] Training model for {EPOCHS} epochs...")
        print(f"Total size of data: {len(train_dataset)} sessions")
        print(f"Batch size: {BATCH_SIZE}")
        print(f"Number of batches per epoch: {len(train_loader)}")
        two_tower_model.train()
        for epoch in range(EPOCHS):
            total_loss = 0
            for session, pos_item, neg_items in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
                session, pos_item, neg_items = session.to(device), pos_item.to(device), neg_items.to(device)
                optimizer.zero_grad()
                session_emb, pos_emb, neg_emb_batch = two_tower_model(session, pos_item, neg_items)
                loss = 0
                for neg_emb in torch.unbind(neg_emb_batch, dim=1):
                    loss += loss_fn(session_emb, pos_emb, neg_emb)
                loss /= NUM_NEGATIVES
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            print(f"[{locale}] Epoch {epoch+1} | Average Loss: {total_loss / len(train_loader):.4f}")

        os.makedirs(MODELS_PATH, exist_ok=True)
        model_save_path = os.path.join(MODELS_PATH, f'query_tower_{locale}.pt')
        torch.save(two_tower_model.query_tower.state_dict(), model_save_path)

        # PREDICTION
        print(f"[{locale}] Generating predictions...")
        query_tower = two_tower_model.query_tower
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
                session_tensor = torch.tensor([session_indices], dtype=torch.long).to(device)
                session_emb = query_tower(session_tensor).cpu().numpy()
                _, I = locale_data['faiss_index'].search(session_emb, NUM_RECOMMENDATIONS)
                predicted_product_ids = [idx_to_id[i] for i in I[0] if i in idx_to_id]
                locale_predictions.append(predicted_product_ids)
        all_predictions.extend([(idx, preds, locale) for idx, preds in zip(test_sessions_df.index, locale_predictions)])

    # Prepare submission file
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
    locales = [loc for _, _, loc in sorted(all_predictions, key=lambda x: x[0])]
    submission_df = pd.DataFrame({
        'next_item_prediction': sorted_predictions,
        'locale': locales
    }, index=original_test_df.index)
    submission_file = os.path.join(configs.OUTPUT_PATH, 'submission.parquet')
    submission_df.to_parquet(submission_file)
    print(f"Submission file saved to: {submission_file}")
    print(f"Submission file shape: {submission_df.shape}")


if __name__ == '__main__':
    main()

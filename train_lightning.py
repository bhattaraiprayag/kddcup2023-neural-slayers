# train_lightning.py

import importlib
import os
import pickle

import faiss
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
                     NUM_RECOMMENDATIONS, SEED, TRIPLET_MARGIN,
                     PRED_SLICER, SLICER, USE_PRED_SLICER, USE_SLICER)
from lightning_model import LightningTwoTower
from negative_sampler import create_negative_samples_for_locale
from utils import model_tuner
from dataset import PredictionDataset

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

    print("\n--- Training a model and predicting for each locale ---")
    all_predictions = []
    for locale in LOCALES:
        print(f"\n===== Processing locale: {locale} =====")
        locale_data = all_locale_data[locale]

        enhanced_embed_path = os.path.join(configs.EMBED_PATH, f'enhanced_embeddings_{locale}.npy')
        if os.path.exists(enhanced_embed_path):
            locale_data['embeddings'] = np.load(enhanced_embed_path)
            faiss.normalize_L2(locale_data['embeddings'])
            locale_faiss_path = os.path.join(configs.INDEX_PATH, f'products_{locale}_enhanced.faiss')
            if os.path.exists(locale_faiss_path):
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
        else:
            print("GNN-enhanced embeddings not found. Using SVD-reduced embeddings as fallback.")

        product_df = locale_data['products']
        all_product_ids = product_df['id'].tolist()
        id_to_idx = {pid: i for i, pid in enumerate(all_product_ids)}
        emb_idx_to_id = {v: k for k, v in locale_data['prod_id_to_emb_idx_map'].items()}

        neg_samples_path = os.path.join(NEGATIVE_SAMPLES_PATH, f'negative_samples_{locale}.pkl')
        if not os.path.exists(neg_samples_path):
            print("\n--- Generating negative samples ---")
            create_negative_samples_for_locale(
                locale,
                locale_data['sessions'] if not USE_SLICER else locale_data['sessions'][:SLICER],
                locale_data['faiss_index'],
                locale_data['prod_id_to_emb_idx_map'],
                NEGATIVE_SAMPLES_PATH
            )
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

        query_tower = model.query_tower.to('cpu')
        query_tower.eval()
        test_sessions_df = locale_data['sessions_test']
        test_sessions = test_sessions_df['prev_items'].str.split(',').tolist()
        if USE_PRED_SLICER:
            test_sessions = test_sessions[:PRED_SLICER]

        pred_dataset = PredictionDataset(
            sessions=test_sessions,
            id_map=locale_data['prod_id_to_emb_idx_map'],
            max_len=MAX_SESSION_LENGTH
        )
        pred_loader = DataLoader(
            pred_dataset, batch_size=BATCH_SIZE*4, shuffle=False,
            num_workers=min(16, int(os.cpu_count() // 2)), pin_memory=True
        )

        locale_predictions = []
        with torch.no_grad():
            for session_batch in tqdm(pred_loader, desc=f"[{locale}] Predicting"):
                session_tensor = session_batch.to('cpu')
                session_emb_batch = query_tower(session_tensor).cpu().numpy()
                _, I_batch = locale_data['faiss_index'].search(session_emb_batch, NUM_RECOMMENDATIONS)
                for recommendations_indices in I_batch:
                    predicted_ids = [emb_idx_to_id[i] for i in recommendations_indices if i in emb_idx_to_id]
                    locale_predictions.append(predicted_ids)
        all_predictions.extend([(idx, preds, locale) for idx, preds in zip(test_sessions_df.index, locale_predictions)])

    if not all_predictions:
        print("No predictions were generated. Exiting.")
        return

    all_predictions.sort(key=lambda x: x[0])
    sorted_predictions = [p[1] for p in all_predictions]
    locales = [p[2] for p in all_predictions]
    if USE_PRED_SLICER:
        original_test_df = pd.concat([all_locale_data[loc]['sessions_test'][:PRED_SLICER] for loc in LOCALES]).sort_index()
    else:
        original_test_df = pd.concat([all_locale_data[loc]['sessions_test'] for loc in LOCALES]).sort_index()
    submission_df = pd.DataFrame({
        'next_item_prediction': sorted_predictions,
        'locale': locales
    }, index=original_test_df.index)
    submission_file = os.path.join(configs.OUTPUT_PATH, 'submission_new.parquet')
    submission_df.to_parquet(submission_file)
    print(f"Submission file saved to: {submission_file}")
    print(f"Submission file shape: {submission_df.shape}")


if __name__ == '__main__':
    main()

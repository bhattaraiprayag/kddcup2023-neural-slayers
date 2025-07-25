# tune_gnn_multi.py

import itertools
import os
import pickle
import time
from datetime import datetime

import multiprocessing as mp
from multiprocessing import Queue
import queue

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.optim.lr_scheduler import (CosineAnnealingWarmRestarts,
                                      OneCycleLR, ReduceLROnPlateau)
from torch_geometric.transforms import RandomLinkSplit
from torch_geometric.utils import from_networkx
from tqdm.auto import tqdm

from configs import (EMBED_PATH, GRAPH_TYPE, LOCALES, N_COMPONENTS,
                     P2P_GRAPH_PATH, SEED)
from gnn_model import GraphSAGE
from prepare_artefacts import prepare_locale_artefacts


def worker(job_queue, results_queue, gpu_id, all_locale_data):
    device = torch.device(f'cuda:{gpu_id}')
    print(f"Worker started on GPU: {gpu_id}")
    while True:
        try:
            locale, params = job_queue.get_nowait()
        except queue.Empty:
            print(f"Worker on GPU {gpu_id} found no more jobs. Exiting.")
            break
        except Exception as e:
            print(f"Error getting job on GPU {gpu_id}: {e}")
            continue
        try:
            data_on_device = {k: v.to(device) if hasattr(v, 'to') else v for k, v in all_locale_data[locale].items()}
            trial_result = run_trial(locale, params, data_on_device, device)
            results_queue.put(trial_result)
        except Exception as e:
            print(f"Error during trial run on GPU {gpu_id} for locale {locale}: {e}")


def load_all_data(locales, device):
    all_data = {}
    initial_artefacts = prepare_locale_artefacts()

    for locale in locales:
        locale_data = initial_artefacts[locale]
        initial_embeddings = locale_data['embeddings']
        prod_id_to_emb_idx = locale_data['prod_id_to_emb_idx_map']
        graph_path = os.path.join(P2P_GRAPH_PATH, f'graph_{GRAPH_TYPE}_{locale}.gpickle')
        if not os.path.exists(graph_path):
            print(f"Graph for {locale} not found. Skipping.")
            continue
        nx_graph = pickle.load(open(graph_path, 'rb'))
        node_ids = sorted(list(nx_graph.nodes()))
        prod_id_to_node_idx = {pid: i for i, pid in enumerate(node_ids)}
        num_nodes = len(node_ids)
        node_features = torch.zeros((num_nodes, N_COMPONENTS), dtype=torch.float)
        for i, pid in enumerate(node_ids):
            if pid in prod_id_to_emb_idx:
                emb_idx = prod_id_to_emb_idx[pid]
                node_features[i] = torch.tensor(initial_embeddings[emb_idx])
        pyg_data = from_networkx(nx_graph)
        pyg_data.x = node_features
        edge_index = torch.tensor([
            [prod_id_to_node_idx[u], prod_id_to_node_idx[v]]
            for u, v in nx_graph.edges()
        ], dtype=torch.long).t().contiguous()
        edge_weight = torch.tensor([d.get('weight', 1.0) for _, _, d in nx_graph.edges(data=True)], dtype=torch.float)
        pyg_data.edge_index, pyg_data.edge_attr = edge_index, edge_weight
        pyg_data.num_nodes = num_nodes
        transform = RandomLinkSplit(
            is_undirected=True, num_val=0.15, num_test=0.15, add_negative_train_samples=True, neg_sampling_ratio=1.0
        )
        train_data, val_data, test_data = transform(pyg_data)
        all_data[locale] = {
            'train_data': train_data,
            'val_data': val_data,
            'test_data': test_data,
            'pyg_data': pyg_data,
            'node_ids': node_ids,
            'prod_id_to_emb_idx': prod_id_to_emb_idx
        }
        print(f"Data for {locale} prepared and loaded.")
    return all_data


def plot_and_save_history(history, locale, trial_id, scheduler_name):
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 9), sharex=True)
    epochs = range(1, len(history['train_loss']) + 1)
    fig.suptitle(f'GraphSAGE Training Metrics of {trial_id} w/ {scheduler_name}', fontsize=16)
    ax1.plot(epochs, history['train_loss'], 
             color='dodgerblue',
             marker='o',
             linestyle='-',
             linewidth=1.5,
             markersize=3)
    ax1.set_title('Training Loss over Epochs')
    ax1.set_ylabel('BCE Loss')
    ax1.legend(['Training Loss'])
    ax2.plot(epochs, history['train_auc'],
             color='green',
             marker='o',
             linestyle='-',
             linewidth=1.5,
             markersize=3,
             label='Training AUC')
    ax2.plot(epochs, history['val_auc'],
             color='red',
             marker='o',
             linestyle='--',
             linewidth=1.5,
             markersize=3,
             label='Validation AUC')
    ax2.set_title('Area Under Curve (AUC) Performance')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('AUC Score')
    ax2.legend()
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plot_save_path = os.path.join("outputs", "tuning", "plots", f'gnn_training_history_{trial_id}_{scheduler_name}.png')
    os.makedirs(os.path.dirname(plot_save_path), exist_ok=True)
    plt.savefig(plot_save_path, dpi=600)
    plt.close(fig)


def train_step(model, optimizer, criterion, data):
    model.train()
    optimizer.zero_grad()
    z = model(data.x, data.edge_index)
    out = model.decode(z, data.edge_label_index).view(-1)
    loss = criterion(out, data.edge_label)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    return float(loss), roc_auc_score(data.edge_label.detach().cpu().numpy(), out.detach().cpu().numpy())


@torch.no_grad()
def test_step(model, data):
    model.eval()
    z = model(data.x, data.edge_index)
    out = model.decode(z, data.edge_label_index).view(-1)
    return roc_auc_score(data.edge_label.cpu().numpy(), out.cpu().numpy())


def run_trial(locale, params, data, device):
    trial_id = f"{locale}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
    print(f"\n--- Starting Trial: {trial_id} ---")
    print(f"Params: {params}")
    train_data = data['train_data'].to(device)
    val_data = data['val_data'].to(device)
    test_data = data['test_data'].to(device)
    pyg_data = data['pyg_data'].to(device)
    model = GraphSAGE(
        in_channels=N_COMPONENTS,
        hidden_channels=N_COMPONENTS * params['hidden_channels_factor'],
        out_channels=N_COMPONENTS
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=params['lr'], weight_decay=params['weight_decay']
    )
    gnn_epochs = 100
    if params['scheduler'] == 'ReduceLROnPlateau':
        scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5, min_lr=1e-10)
    elif params['scheduler'] == 'OneCycleLR':
        scheduler = OneCycleLR(optimizer, max_lr=params['lr'], epochs=gnn_epochs, steps_per_epoch=1)
    elif params['scheduler'] == 'CosineAnnealingWarmRestarts':
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=50, T_mult=2, eta_min=1e-8)
    else:
        raise ValueError("Invalid scheduler specified")
    criterion = nn.BCEWithLogitsLoss().to(device)
    best_val_auc = 0
    patience, max_patience = 0, 50
    history = {'train_loss': [], 'train_auc': [], 'val_auc': []}
    model_save_path = os.path.join("outputs", "tuning", "models", f'best_gnn_{locale}_{trial_id}.pt')
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    start_time = time.time()
    with tqdm(range(1, gnn_epochs + 1), unit="epoch", desc=f"Training {trial_id}") as pbar:
        for epoch in pbar:
            loss, train_auc = train_step(model, optimizer, criterion, train_data)
            val_auc = test_step(model, val_data)
            history['train_loss'].append(loss)
            history['train_auc'].append(train_auc)
            history['val_auc'].append(val_auc)
            if isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step(val_auc)
            else:
                scheduler.step()
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                torch.save(model.state_dict(), model_save_path)
                patience = 0
            else:
                patience += 1
            if patience >= max_patience:
                print(f"Early stopping at epoch {epoch}.")
                break
            pbar.set_postfix(train_auc=f"{train_auc:.4f}", val_auc=f"{val_auc:.4f}", best_val_auc=f"{best_val_auc:.4f}")
    training_time = time.time() - start_time
    print(f"Trial trained in {training_time:.2f} seconds.")
    plot_path = plot_and_save_history(history, locale, trial_id, params['scheduler'])
    model.load_state_dict(torch.load(model_save_path, weights_only=True))
    test_auc = test_step(model, test_data)
    print(f"Final AUC -> Validation: {best_val_auc:.4f}, Test: {test_auc:.4f}")
    with torch.no_grad():
        final_embeddings_gpu = model.encode(pyg_data.x, pyg_data.edge_index)
        final_embeddings_cpu = final_embeddings_gpu.cpu().numpy()
    num_products_original = len(data['prod_id_to_emb_idx'])
    enhanced_embeddings_ordered = np.zeros((num_products_original, N_COMPONENTS), dtype=np.float32)
    for i, pid in enumerate(data['node_ids']):
        if pid in data['prod_id_to_emb_idx']:
            original_emb_idx = data['prod_id_to_emb_idx'][pid]
            enhanced_embeddings_ordered[original_emb_idx] = final_embeddings_cpu[i]
    embed_save_path = os.path.join("outputs", "tuning", "embeddings", f'enhanced_embeddings_{trial_id}.npy')
    # os.makedirs(os.path.dirname(embed_save_path), exist_ok=True)
    # np.save(embed_save_path, enhanced_embeddings_ordered)
    return {
        'trial_id': trial_id,
        'locale': locale,
        'best_val_auc': best_val_auc,
        'test_auc': test_auc,
        'training_time_s': training_time,
        'model_path': model_save_path,
        'plot_path': plot_path,
        'embeddings_path': embed_save_path,
        **params
    }


def main():
    pl.seed_everything(SEED, workers=True)
    num_gpus = torch.cuda.device_count()
    print(f"Found {num_gpus} available GPUs.")

    # --- Hyperparameter Search Space ---
    param_grid = {
        'lr': [1e-1, 1e-2],
        'weight_decay': [1e-1, 1e-2, 1e-3, 1e-4],
        'hidden_channels_factor': [2],
        'scheduler': ['ReduceLROnPlateau', 'OneCycleLR', 'CosineAnnealingWarmRestarts']
    }

    all_locale_data = load_all_data(LOCALES, torch.device('cpu')) # Load data on CPU first
    keys, values = zip(*param_grid.items())
    hyperparameter_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    results = []
    if num_gpus > 1:
        job_queue = Queue()
        results_queue = Queue()
        for locale in LOCALES:
            if locale not in all_locale_data:
                print(f"No data for {locale}, skipping.")
                continue
            for params in hyperparameter_combinations:
                job_queue.put((locale, params))
        processes = []
        for gpu_id in range(num_gpus):
            p = mp.Process(target=worker, args=(job_queue, results_queue, gpu_id, all_locale_data))
            processes.append(p)
            p.start()
        for p in processes:
            p.join()
        while not results_queue.empty():
            results.append(results_queue.get())
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")
        total_trials = len(LOCALES) * len(hyperparameter_combinations)
        print(f"\nStarting hyperparameter tuning for {len(LOCALES)} locales.")
        print(f"Total trials to run: {total_trials}")
        for locale in LOCALES:
            if locale not in all_locale_data:
                print(f"No data for {locale}, skipping.")
                continue
            print(f"\n===== Tuning for {locale} =====")
            locale_data = {k: v.to(device) if hasattr(v, 'to') else v for k, v in all_locale_data[locale].items()}
            for params in hyperparameter_combinations:
                trial_result = run_trial(locale, params, locale_data, device)
                results.append(trial_result)
                results_df = pd.DataFrame(results)
                results_df.to_csv(os.path.join("outputs", "tuning", "tuning_results.csv"), index=False)

    print("\n--- Hyperparameter Tuning Complete ---")
    results_df = pd.DataFrame(results)
    results_df.sort_values(by=['locale', 'best_val_auc'], ascending=[True, False], inplace=True)
    print(results_df)
    results_df.to_csv(os.path.join("outputs", "tuning", "tuning_results_final.csv"), index=False)
    print("\nFinal results saved to 'outputs/tuning/tuning_results_final.csv'")


if __name__ == '__main__':
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass
    main()

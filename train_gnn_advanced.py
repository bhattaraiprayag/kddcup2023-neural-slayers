# train_gnn_advanced.py

import time
import os
import pickle
import numpy as np
import torch
import torch.nn as nn
import pytorch_lightning as pl
from sklearn.metrics import roc_auc_score
from torch.optim.lr_scheduler import ReduceLROnPlateau, OneCycleLR, CosineAnnealingWarmRestarts
from torch_geometric.transforms import RandomLinkSplit
from torch_geometric.utils import from_networkx
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

from configs import (EMBED_PATH, LOCALES, N_COMPONENTS, P2P_GRAPH_PATH, SEED, GRAPH_TYPE)
from gnn_model import GraphSAGE
from prepare_artefacts import prepare_locale_artefacts


def plot_and_save_history(history, locale, scheduler_name):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    epochs = range(1, len(history['train_loss']) + 1)
    ax1.plot(epochs, history['train_loss'], 'bo-', label='Train Loss')
    ax1.set_ylabel('BCE Loss')
    ax1.set_title(f'Training and Validation Metrics for {locale}')
    ax1.legend()
    ax1.grid(True)
    ax2.plot(epochs, history['train_auc'], 'go-', label='Train AUC')
    ax2.plot(epochs, history['val_auc'], 'ro-', label='Validation AUC')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('AUC')
    ax2.legend()
    ax2.grid(True)
    plt.tight_layout()
    plot_save_path = os.path.join("outputs", "plots", f'gnn_training_history_{locale}_{scheduler_name}.png')
    os.makedirs(os.path.dirname(plot_save_path), exist_ok=True)
    plt.savefig(plot_save_path)
    plt.close()


def train(model, optimizer, criterion, data):
    model.train()
    optimizer.zero_grad()
    # z = model.encode(data.x, data.edge_index)
    z = model(data.x, data.edge_index, data.edge_weight)
    out = model.decode(z, data.edge_label_index).view(-1)
    loss = criterion(out, data.edge_label)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    return float(loss), roc_auc_score(data.edge_label.detach().cpu().numpy(), out.detach().cpu().numpy())


@torch.no_grad()
def test(model, data):
    model.eval()
    # z = model.encode(data.x, data.edge_index)
    z = model(data.x, data.edge_index, data.edge_weight)
    out = model.decode(z, data.edge_label_index).view(-1)
    return roc_auc_score(data.edge_label.cpu().numpy(), out.cpu().numpy())


def main():
    pl.seed_everything(SEED, workers=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    all_locale_data = prepare_locale_artefacts()
    for locale in LOCALES:
        print(f"\n===== Processing {locale} =====")
        locale_data = all_locale_data[locale]
        initial_embeddings = locale_data['embeddings']
        prod_id_to_emb_idx = locale_data['prod_id_to_emb_idx_map']
        graph_path = os.path.join(P2P_GRAPH_PATH, f'graph_{GRAPH_TYPE}_{locale}.gpickle')
        if not os.path.exists(graph_path):
            print(f"Graph for {locale} not found at {graph_path}. Skipping.")
            continue
        nx_graph = pickle.load(open(graph_path, 'rb'))

        timer = time.time()
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
        # pyg_data.edge_weight = pyg_data.edge_attr
        edge_index = torch.tensor([
            [prod_id_to_node_idx[u], prod_id_to_node_idx[v]]
            for u, v in nx_graph.edges()
        ], dtype=torch.long).t().contiguous()
        edge_weight = torch.tensor([d.get('weight', 1.0) for _, _, d in nx_graph.edges(data=True)], dtype=torch.float)
        # pyg_data.edge_index, pyg_data.edge_attr = edge_index, edge_weight
        pyg_data.edge_index, pyg_data.edge_attr = edge_index, pyg_data.edge_weight
        pyg_data.num_nodes = num_nodes

        transform = RandomLinkSplit(
            is_undirected=True, num_val=0.1, num_test=0.1, add_negative_train_samples=True, neg_sampling_ratio=1.0
        )
        train_data, val_data, test_data = transform(pyg_data)
        train_data, val_data, test_data = train_data.to(device), val_data.to(device), test_data.to(device)
        print(f"Data prep: {time.time() - timer:.2f} seconds | Number of nodes: {num_nodes}, edges: {pyg_data.num_edges}")

        model = GraphSAGE(
            in_channels=N_COMPONENTS,
            hidden_channels=N_COMPONENTS * 2,
            out_channels=N_COMPONENTS
        ).to(device)

        gnn_epochs = 1000
        # gnn_epochs = 10

        optimizer = torch.optim.AdamW(
            params=model.parameters(),
            lr=1e-3,
            weight_decay=1e-2
        )
        scheduler_patience = 5
        scheduler = ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=scheduler_patience, min_lr=1e-10
        )
        # scheduler = OneCycleLR(
        #     optimizer, max_lr=1e-3, epochs=gnn_epochs, steps_per_epoch=1,
        # )
        # scheduler = CosineAnnealingWarmRestarts(
        #     optimizer,
        #     T_0=50,  # Number of epochs for the first restart
        #     T_mult=2, # Factor to increase T_0 after each restart
        #     eta_min=1e-8 # Minimum learning rate
        # )
        criterion = nn.BCEWithLogitsLoss().to(device)
        scheduler_name = scheduler.__class__.__name__
        print(f"Using scheduler: {scheduler_name}")

        best_val_auc = 0
        patience, max_patience = 0, 50
        history = {'train_loss': [], 'train_auc': [], 'val_auc': []}
        model_save_path = os.path.join("outputs", "models", f'best_gnn_{locale}.pt')
        os.makedirs(os.path.dirname(model_save_path), exist_ok=True)

        timer = time.time()
        with tqdm(range(1, gnn_epochs + 1), unit="epoch", desc=f"Training {locale}") as pbar:
            for epoch in pbar:
                loss, train_auc = train(model, optimizer, criterion, train_data)
                val_auc = test(model, val_data)
                history['train_loss'].append(loss)
                history['train_auc'].append(train_auc)
                history['val_auc'].append(val_auc)
                scheduler.step(val_auc) if isinstance(scheduler, ReduceLROnPlateau) else scheduler.step()
                # print(scheduler.get_last_lr())
                pbar.set_postfix(train_auc=f"{train_auc:.4f}", val_auc=f"{val_auc:.4f}", best_val_auc=f"{best_val_auc:.4f}")
                if val_auc > best_val_auc:
                    best_val_auc = val_auc
                    torch.save(model.state_dict(), model_save_path)
                    patience = 0
                else:
                    patience += 1
                if patience >= max_patience:
                    print(f"Early stopping at epoch {epoch}...")
                    break
        print(f"Trained in {time.time() - timer:.2f} seconds.")
        plot_and_save_history(history, locale, scheduler_name)
        model.load_state_dict(torch.load(model_save_path, weights_only=True))
        test_auc = test(model, test_data)
        print(f"Final AUC: Validation: {best_val_auc:.4f}, Test: {test_auc:.4f}")

        with torch.no_grad():
            # final_embeddings_gpu = model.encode(pyg_data.x.to(device), pyg_data.edge_index.to(device))
            final_embeddings_gpu = model.encode(pyg_data.x.to(device), pyg_data.edge_index.to(device), pyg_data.edge_weight.to(device))
            final_embeddings_cpu = final_embeddings_gpu.cpu().numpy()

        num_products_original = len(prod_id_to_emb_idx)
        enhanced_embeddings_ordered = np.zeros((num_products_original, N_COMPONENTS), dtype=np.float32)
        for i in range(num_nodes):
            pid = node_ids[i]
            if pid in prod_id_to_emb_idx:
                original_emb_idx = prod_id_to_emb_idx[pid]
                enhanced_embeddings_ordered[original_emb_idx] = final_embeddings_cpu[i]

        save_path = os.path.join(EMBED_PATH, f'enhanced_embeddings_{locale}.npy')
        np.save(save_path, enhanced_embeddings_ordered)


if __name__ == '__main__':
    main()

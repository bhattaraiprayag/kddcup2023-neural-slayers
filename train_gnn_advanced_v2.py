# train_gnn_advanced_v2.py - This version uses a different data splitting strategy using RandomLinkSplit and LinkNeighborLoader

import time
import os
import pickle
import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from sklearn.metrics import roc_auc_score
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.loader import LinkNeighborLoader
from torch_geometric.transforms import RandomLinkSplit
from torch_geometric.utils import from_networkx
from tqdm import tqdm
import matplotlib.pyplot as plt

from configs import (EMBED_PATH, LOCALES, N_COMPONENTS, P2P_GRAPH_PATH, SEED, GRAPH_TYPE, LEARNING_RATE, BATCH_SIZE)
from gnn_model import GraphSAGE
from prepare_artefacts import prepare_locale_artefacts


def plot_and_save_history(history, locale):
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
    plot_save_path = os.path.join("outputs", "plots", f'training_history_{locale}.png')
    os.makedirs(os.path.dirname(plot_save_path), exist_ok=True)
    plt.savefig(plot_save_path)
    print(f"Saved training plot to {plot_save_path}")
    plt.close()


def train(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    for batch in tqdm(loader, desc="Training Batches"):
        batch = batch.to(device)
        optimizer.zero_grad()
        z = model.encode(batch.x, batch.edge_index)
        out = model.decode(z, batch.edge_label_index).view(-1)
        loss = criterion(out, batch.edge_label)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
        all_preds.append(out.detach().cpu())
        all_labels.append(batch.edge_label.cpu())
    avg_loss = total_loss / len(loader)
    y_true = torch.cat(all_labels).int()
    y_pred = torch.cat(all_preds)
    # return avg_loss, roc_auc_score(torch.cat(all_labels), torch.cat(all_preds))
    return avg_loss, roc_auc_score(y_true, y_pred)


@torch.no_grad()
def test(model, loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    for batch in loader:
        batch = batch.to(device)
        z = model.encode(batch.x, batch.edge_index)
        out = model.decode(z, batch.edge_label_index).view(-1)
        all_preds.append(out.cpu())
        all_labels.append(batch.edge_label.cpu())
    y_true = torch.cat(all_labels).int()
    y_pred = torch.cat(all_preds)
    # return roc_auc_score(torch.cat(all_labels), torch.cat(all_preds))
    return roc_auc_score(y_true, y_pred)


def main():
    pl.seed_everything(SEED, workers=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    all_locale_data = prepare_locale_artefacts()
    for locale in LOCALES:
        locale_data = all_locale_data[locale]
        initial_embeddings = locale_data['embeddings']
        prod_id_to_emb_idx = locale_data['prod_id_to_emb_idx_map']
        graph_path = os.path.join(P2P_GRAPH_PATH, f'graph_{GRAPH_TYPE}_{locale}.gpickle')
        if not os.path.exists(graph_path):
            print(f"Graph for {locale} not found at {graph_path}. Skipping.")
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
        train_data, val_data, test_data = train_data.to(device), val_data.to(device), test_data.to(device)
        train_loader = LinkNeighborLoader(
            data=train_data, num_neighbors=[15, 10], batch_size=BATCH_SIZE,
            edge_label_index=train_data.edge_label_index, edge_label=train_data.edge_label,
            shuffle=True, neg_sampling_ratio=1.0,
        )
        val_loader = LinkNeighborLoader(
            data=val_data, num_neighbors=[-1, -1], batch_size=BATCH_SIZE,
            edge_label_index=val_data.edge_label_index, edge_label=val_data.edge_label,
            shuffle=False,
        )
        test_loader = LinkNeighborLoader(
            data=test_data, num_neighbors=[-1, -1], batch_size=BATCH_SIZE,
            edge_label_index=test_data.edge_label_index, edge_label=test_data.edge_label,
            shuffle=False,
        )

        model = GraphSAGE(
            in_channels=N_COMPONENTS,
            hidden_channels=N_COMPONENTS * 2,
            out_channels=N_COMPONENTS
        ).to(device)

        optimizer = torch.optim.Adam(
            params=model.parameters(), lr=LEARNING_RATE, weight_decay=5e-5
        )
        scheduler = ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=10
        )
        criterion = nn.BCEWithLogitsLoss().to(device)

        best_val_auc = 0
        patience, max_patience = 0, 50
        history = {'train_loss': [], 'train_auc': [], 'val_auc': []}
        model_save_path = os.path.join("outputs", "models", f'best_gnn_{locale}.pt')
        os.makedirs(os.path.dirname(model_save_path), exist_ok=True)

        timer = time.time()
        # epochs = 1000
        # for epoch in range(1, epochs + 1):
        print(f"\n===== Training GNN for {locale} =====")
        for epoch in range(1, 6):
            loss, train_auc = train(
                model, train_loader, optimizer, criterion, device
            )
            val_auc = test(model, val_loader, device)
            history['train_loss'].append(loss)
            history['train_auc'].append(train_auc)
            history['val_auc'].append(val_auc)
            scheduler.step(val_auc)
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                torch.save(model.state_dict(), model_save_path)
                patience = 0
            else:
                patience += 1
            if epoch % 10 == 0:
                print(f'Epoch: {epoch:03d}, Train loss: {loss:.4f}, Train AUC: {train_auc:.4f}, Best Val AUC: {best_val_auc:.4f}')
            if patience >= max_patience:
                print(f"Early stopping at epoch {epoch}")
                break
        print(f"Training completed in {time.time() - timer:.2f} seconds.")
        plot_and_save_history(history, locale)
        model.load_state_dict(torch.load(model_save_path, weights_only=True))
        test_auc = test(model, test_loader, device)
        print(f"AUC: Validation: {best_val_auc:.4f}, Test: {test_auc:.4f}")

        with torch.no_grad():
            final_embeddings_gpu = model.encode(pyg_data.x.to(device), pyg_data.edge_index.to(device))
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
        print(f"Saved enhanced embeddings for {locale} to {save_path}")


if __name__ == '__main__':
    main()
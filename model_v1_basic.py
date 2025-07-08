import torch
import torch.nn as nn


class QueryTower(nn.Module):
    def __init__(self, num_items, embedding_dim, hidden_dim, n_layers, precomputed_embeddings):
        super().__init__()
        self.item_embeddings = nn.Embedding.from_pretrained(precomputed_embeddings, freeze=True)
        self.gru = nn.GRU(embedding_dim, hidden_dim, n_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, embedding_dim)
        self.layer_norm = nn.LayerNorm(embedding_dim)

    def forward(self, session_item_indices):
        embedded_session = self.item_embeddings(session_item_indices)
        _, last_hidden_state = self.gru(embedded_session)
        session_vector = self.fc(last_hidden_state[-1])
        return self.layer_norm(session_vector)


class TwoTowerModel(nn.Module):
    def __init__(self, num_items, embedding_dim, hidden_dim, n_layers, precomputed_embeddings):
        super().__init__()
        self.query_tower = QueryTower(num_items, embedding_dim, hidden_dim, n_layers, precomputed_embeddings)

    def forward(self, session_indices, positive_item_indices, negative_item_indices):
        session_embedding = self.query_tower(session_indices)
        positive_item_embedding = self.query_tower.item_embeddings(positive_item_indices)
        negative_item_embedding = self.query_tower.item_embeddings(negative_item_indices)
        return session_embedding, positive_item_embedding, negative_item_embedding

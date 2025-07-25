# models.py

import math

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(1)].transpose(0, 1)
        return self.dropout(x)


class SessionEncoder(nn.Module):
    def __init__(self, embedding_dim, nhead, num_encoder_layers, dim_feedforward, dropout=0.1):
        super().__init__()
        self.pos_encoder = PositionalEncoding(embedding_dim, dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim, 
            nhead=nhead, 
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=num_encoder_layers
        )
        self.layer_norm = nn.LayerNorm(embedding_dim)

    def forward(self, embedded_session):
        seq_len = embedded_session.size(1)
        causal_mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(embedded_session.device)
        x = self.pos_encoder(embedded_session)
        transformer_output = self.transformer_encoder(x, mask=causal_mask)
        session_vector = transformer_output[:, -1, :]
        return self.layer_norm(session_vector)


class QueryTower(nn.Module):
    def __init__(self, embedding_dim, nhead, num_encoder_layers, dim_feedforward, precomputed_embeddings, dropout=0.1):
        super().__init__()
        # self.item_embeddings = nn.Embedding.from_pretrained(precomputed_embeddings, freeze=True)
        self.item_embeddings = nn.Embedding.from_pretrained(precomputed_embeddings)
        self.session_encoder = SessionEncoder(embedding_dim, nhead, num_encoder_layers, dim_feedforward, dropout)

    def forward(self, session_item_indices):
        embedded_session = self.item_embeddings(session_item_indices)
        session_vector = self.session_encoder(embedded_session)
        return session_vector


class TwoTowerModel(nn.Module):
    def __init__(self, embedding_dim, nhead, num_encoder_layers, dim_feedforward, precomputed_embeddings, dropout=0.1):
        super().__init__()
        self.query_tower = QueryTower(embedding_dim, nhead, num_encoder_layers, dim_feedforward, precomputed_embeddings, dropout)

    def forward(self, session_indices, positive_item_indices, negative_item_indices):
        session_embedding = self.query_tower(session_indices)
        positive_item_embedding = self.query_tower.item_embeddings(positive_item_indices).squeeze(1)
        negative_item_embedding = self.query_tower.item_embeddings(negative_item_indices).squeeze(1)
        return session_embedding, positive_item_embedding, negative_item_embedding

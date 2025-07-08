# lightning_model.py

import torch
import torch.nn as nn
import pytorch_lightning as pl

from model import TwoTowerModel


class LightningTwoTower(pl.LightningModule):
    def __init__(self, embedding_dim, nhead,
                 num_encoder_layers, dim_feedforward,
                 precomputed_embeddings, learning_rate,
                 triplet_margin, num_negatives, dropout=0.1
                 ):
        super().__init__()
        self.save_hyperparameters(
            'embedding_dim',
            'nhead',
            'num_encoder_layers',
            'dim_feedforward',
            'learning_rate',
            'triplet_margin',
            'num_negatives'
        )
        self.two_tower_model = TwoTowerModel(
            embedding_dim=embedding_dim,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            dim_feedforward=dim_feedforward,
            precomputed_embeddings=precomputed_embeddings,
            dropout=dropout
        )
        self.loss_fn = nn.TripletMarginLoss(margin=self.hparams.triplet_margin)

    def forward(self, session_indices, positive_item_indices, negative_item_indices):
        return self.two_tower_model(session_indices, positive_item_indices, negative_item_indices)

    def training_step(self, batch, batch_idx):
        session, pos_item, neg_items = batch
        session_emb, pos_emb, neg_emb_batch = self(session, pos_item, neg_items)
        loss = 0
        for neg_emb in torch.unbind(neg_emb_batch, dim=1):
            loss += self.loss_fn(session_emb, pos_emb, neg_emb)
        loss /= self.hparams.num_negatives
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.learning_rate)
        return optimizer

    @property
    def query_tower(self):
        return self.two_tower_model.query_tower

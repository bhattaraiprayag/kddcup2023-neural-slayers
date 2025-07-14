# lightning_model.py

import os
import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from models import TwoTowerModel
from dataset import SessionDataset


class LightningTwoTower(pl.LightningModule):
    def __init__(self, embedding_dim, nhead,
                 num_encoder_layers, dim_feedforward,
                 precomputed_embeddings, learning_rate,
                 triplet_margin, dropout=0.1,
                 train_sessions_df=None, id_to_idx=None,
                 neg_samples_map=None, max_session_length=None,
                 num_negatives=None, batch_size=None
                 ):
        super().__init__()
        self.save_hyperparameters(
            'embedding_dim', 'nhead', 'num_encoder_layers',
            'dim_feedforward', 'learning_rate', 'triplet_margin',
            'num_negatives', 'batch_size'
        )
        self.train_sessions_df = train_sessions_df
        self.id_to_idx = id_to_idx
        self.neg_samples_map = neg_samples_map
        self.max_session_length = max_session_length

        self.two_tower_model = TwoTowerModel(
            embedding_dim=embedding_dim,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            dim_feedforward=dim_feedforward,
            precomputed_embeddings=precomputed_embeddings,
            dropout=dropout
        )
        self.loss_fn = nn.TripletMarginLoss(margin=self.hparams.triplet_margin)

    def forward(self, session_indices, positive_item_indices,
                negative_item_indices):
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
        # return optimizer
        scheduler = CosineAnnealingWarmRestarts(
            optimizer,
            T_0=50,  # Number of epochs for the first restart
            T_mult=2,  # Factor to increase T_0 after each restart
            eta_min=1e-8  # Minimum learning rate
        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'train_loss'
            }
        }

    def train_dataloader(self):
        if self.train_sessions_df is None:
            return None
        train_dataset = SessionDataset(
            self.train_sessions_df, self.id_to_idx, self.neg_samples_map,
            self.max_session_length, self.hparams.num_negatives
        )
        return DataLoader(
            train_dataset, batch_size=self.hparams.batch_size, shuffle=True,
            num_workers=min(16, int(os.cpu_count() // 2)), pin_memory=True
        )

    @property
    def query_tower(self):
        return self.two_tower_model.query_tower

# dataset.py

import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset


class SessionDataset(Dataset):
    def __init__(
            self, sessions_df, product_to_idx,
            neg_samples_map, max_len, num_negs
            ):
        self.sessions = sessions_df['prev_items'].str.split(',').tolist()
        self.labels = sessions_df['next_item'].tolist()
        self.product_to_idx = product_to_idx
        self.neg_samples_map = neg_samples_map
        self.max_len = max_len
        self.num_negs = num_negs

    def __len__(self):
        return len(self.sessions)

    def __getitem__(self, idx):
        session = self.sessions[idx]
        label = self.labels[idx]
        session_indices = [self.product_to_idx.get(item, 0) for item in session]
        if len(session_indices) > self.max_len:
            session_indices = session_indices[-self.max_len:]
        else:
            session_indices = [0] * (self.max_len - len(session_indices)) + session_indices
        label_idx = self.product_to_idx.get(label, 0)
        possible_negs = self.neg_samples_map.get(label, [])
        if not possible_negs: # Fallback for items not in map
            possible_negs = list(self.neg_samples_map.keys())
        chosen_negs_ids = np.random.choice(possible_negs, self.num_negs, replace=True)
        negative_indices = [self.product_to_idx.get(nid, 0) for nid in chosen_negs_ids]

        return (
            torch.tensor(session_indices, dtype=torch.long),
            torch.tensor(label_idx, dtype=torch.long),
            torch.tensor(negative_indices, dtype=torch.long)
        )

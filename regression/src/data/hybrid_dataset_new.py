# -*- coding: utf-8 -*-
"""
Created on Mon Apr  7 11:40:35 2025

@author: gmfet
"""

import torch
from torch.utils.data import Dataset

class VGDDataset(Dataset):
    def __init__(self, all_data, data_transforms, pred_vars, seq_len=50):
        self.seq_len = seq_len
        self.all_data = all_data
        
        self.dyn_transform = data_transforms[0]
        self.static_transform = data_transforms[1]
        self.target_transform = data_transforms[2]
        
        self.dyn_vars = pred_vars[0]
        self.static_vars = pred_vars[1]
        self.target_var = pred_vars[2]
        
        self.grouped_data = self._group_data()
        self.sample_indices = self._create_sample_indices()

    def __len__(self):
        return len(self.sample_indices)

    def __getitem__(self, idx):
        
        (target_east, target_north), start_idx = self.sample_indices[idx]
        mp_data = self.grouped_data.get_group((target_east, target_north))
        mp_data = mp_data.sort_values(by='time') # Ensure it's sorted

        ################################ Dynamic feature ################################
        dyn_seq = mp_data.loc[start_idx : start_idx + self.seq_len - 1][self.dyn_vars].values
                
        ################################ Static feature ################################
        static_pred = mp_data.loc[mp_data.index[0]][self.static_vars].values
        
        ################################ Target feature ################################
        target_idx = start_idx + self.seq_len
        if target_idx in mp_data.index:
            target = mp_data.loc[target_idx][self.target_var].values
        else:
            raise IndexError("Target index out of bounds")
        
        dyn_seq = self.dyn_transform(dyn_seq) if self.dyn_transform else dyn_seq
        static_pred = self.static_transform(static_pred) if self.static_transform else static_pred
        target = self.target_transform(target) if self.target_transform else target
        
        dyn_tensor    = torch.tensor(dyn_seq, dtype=torch.float32)
        static_tensor = torch.tensor(static_pred, dtype=torch.float32)
        target_tensor = torch.tensor(target, dtype=torch.float32)
        
        return dyn_tensor, static_tensor, target_tensor, target_east, target_north
    
    def _group_data(self):
        """Groups the data by MP and sorts by time."""
        df = self.all_data
        return df.groupby(['easting', 'northing'])

    def _create_sample_indices(self):
        """Creates a list of tuples (mp_coords, start_time_index) for each possible sample."""
        indices = []
        for mp_coords, mp_data in self.grouped_data:
            mp_data = mp_data.sort_values(by='time')
            num_time_steps = len(mp_data)
            if num_time_steps >= self.seq_len + 1:
                for i in range(num_time_steps - self.seq_len):
                    indices.append((mp_coords, mp_data.index[i]))
        return indices




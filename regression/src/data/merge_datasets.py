# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 09:25:04 2024

@author: 39351
"""
import torch
from torch.utils.data import Dataset

def merge_datasets(datasets, batch_size=32, shuffle=True):
    """
    Merges multiple datasets into one and returns the merged dataset.

    Args:
        datasets (list of Dataset): List of datasets to merge.
        batch_size (int): Batch size for the DataLoader.
        shuffle (bool): Whether to shuffle the data.

    Returns:
        DataLoader: A DataLoader for the merged dataset.
    """
    merged_data = torch.cat([dataset[i][0].unsqueeze(0) for dataset in datasets for i in range(len(dataset))], dim=0)
    merged_targets = torch.cat([dataset[i][1].unsqueeze(0) for dataset in datasets for i in range(len(dataset))], dim=0)

    # Create a new merged dataset
    merged_dataset = MergedVGDDataset(merged_data, merged_targets)

    return merged_dataset

class MergedVGDDataset(Dataset):
    """
    A custom dataset class for merged data.
    """
    def __init__(self, data, targets):
        """
        Args:
            data (Tensor): The input data.
            targets (Tensor): The corresponding targets.
        """
        assert len(data) == len(targets), "Data and targets must have the same length"
        self.data = data
        self.targets = targets

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]

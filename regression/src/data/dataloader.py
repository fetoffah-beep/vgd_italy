# -*- coding: utf-8 -*-
"""
Created on Mon Jan 13 09:49:46 2025

@author: 39351
"""
import torch
from torch.utils.data import DataLoader
from line_profiler import profile

class VGDDataLoader:
    @profile
    def __init__(self,  dataset, batch_size, num_workers=0, shuffle=False):
        """
        A class to handle DataLoader creation for both static and dynamic features.
        
        
        Args:
            dataset (VGDDataset): The dataset instance (VGDDataset) containing both static and dynamic features.
            batch_size (int): Batch size for loading the data.
            num_workers (int): Number of workers for parallel data loading.
            shuffle (bool): Whether to shuffle the data (only for training).
        
        Source:
            https://pytorch.org/tutorials/beginner/basics/data_tutorial.html
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.pin_memory = torch.cuda.is_available()
        self.persistent_workers = self.num_workers > 0
        self.prefetch_factor = 2

        self.data_loader = self._create_dataloaders()
        
    @profile
    def _create_dataloaders(self):
        data_loader = DataLoader(self.dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=self.shuffle,
                                  pin_memory=self.pin_memory, persistent_workers =self.persistent_workers) #, prefetch_factor=prefetch_factor)
        
        return data_loader

    @profile
    @property
    def dataloader(self):
        return self.data_loader

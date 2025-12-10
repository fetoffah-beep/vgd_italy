# -*- coding: utf-8 -*-
"""
Created on Mon Jan 13 09:49:46 2025

@author: 39351
"""
import torch
from torch.utils.data import DataLoader
import time

class VGDDataLoader:
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

        self.data_loader = self._create_dataloaders()
        

    def _create_dataloaders(self):
        pin_memory = torch.cuda.is_available()
        persistent_workers = self.num_workers > 0
        # prefetch_factor = 1



        data_loader = DataLoader(self.dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=self.shuffle,
                                  pin_memory=pin_memory, persistent_workers =persistent_workers) #, prefetch_factor=prefetch_factor)
        
        return data_loader


    @property
    def dataloader(self):
        return self.data_loader

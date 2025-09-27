# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 11:33:13 2025

@author: 39351
"""
# https://arxiv.org/pdf/1610.09513

import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import math

from scipy.spatial import cKDTree
from torchvision.transforms import Compose
from .transforms.transforms import NormalizeTransform, ReshapeTransform, LogTransform, ReshapeTransformCNN




class VGDDataset(Dataset):
    def __init__(self, split, target_displacement, target_times, target_mean, target_std, dyn_predictors, pred_vars, pred_mean, pred_std, static_predictors, static_vars, static_mean, static_std, device='cuda', seq_len=30):
        
        """
        Args:
            transform (callable, optional): Optional transform to apply to the predictor.
            target_transform (callable, optional): Optional transform to apply to target.
            device (str): Device to use ('cuda' or 'cpu').
            seq_len (int): Length of each time series sequence.
        """
        self.split = split
        self.device = device
        self.seq_len = seq_len
        

        # Load displacement and predictor data as before        
        self.target_displacement = target_displacement
        self.target_times = target_times
        self.target_mean = target_mean
        self.target_std = target_std
        
        self.dyn_predictors = dyn_predictors
        self.pred_vars = pred_vars
        self.pred_mean = pred_mean
        self.pred_std = pred_std

        
        self.static_predictors = static_predictors
        self.static_vars = static_vars
        self.static_mean = static_mean
        self.static_std = static_std

        
        self.H = self.W = 5
        
        
        
        # # Split data into train, val, test
        # self.time_steps = self._get_time_steps()
        
        # self.sequences = self._generate_sequences()
        
        
        
        
        
        self.dyn_predictors["valid_time"] = pd.to_datetime(self.dyn_predictors["valid_time"])
        
        
        # self.target_displacement  =self._get_split(self.split, self.target_displacement)
        
        self.num_points, _ = self.target_displacement.shape
        
        
        
    def __len__(self):
        """Return the total number of sequences."""
        # When working with time series data in sequences, you are essentially sliding a 
        # window of size seq_len over the time axis, so you 
        # can extract multiple sequences from the available time steps.
        
        # Total time_steps = T
        # Total time sequences = T-seq_len+1
        # Multiply this by the number of points to get the length of the dataset
        
        
        return self.num_points * (len(self.target_times) - self.seq_len + 1)
    
    

    def __getitem__(self, idx):
        """Retrieve a time series segment for a point."""
        
        
        #################### Displacement of an MP at a point in time ####################
        
        # idx is to get a sample
        # a sample is to contain the seq_len of data for corresponding point
        
                
        spatial_idx = idx // (len(self.target_times) - self.seq_len + 1) 
        time_idx = idx % (len(self.target_times) - self.seq_len + 1)
        
        target_point = self.target_displacement.iloc[spatial_idx]
        target_east, target_north = target_point["easting"], target_point["northing"]
        grid_points = self._generate_grid(target_east, target_north)
        
        
        start_time = self.target_times[time_idx]
        end_time = self.target_times[time_idx + self.seq_len-1]
        
        target_seq = self.target_displacement.iloc[spatial_idx]
        
        target_seq = target_seq.loc[start_time:end_time].values.reshape(self.seq_len, 1)
        
        target_dates = self.target_times[(self.target_times >= start_time) & (self.target_times <= end_time)]
        
        target_dates = pd.to_datetime(target_dates, format="%Y%m%d")
        
        
        ############################## Dynamic feature ##############################
        
        dyn_seq = np.zeros((self.seq_len, len(self.pred_vars), len(grid_points)))
        
        for t in range(len(target_dates)):
            time_filtered = self.dyn_predictors[(self.dyn_predictors["valid_time"] >= target_dates[t]) & 
                                      (self.dyn_predictors["valid_time"] < target_dates[t]+pd.Timedelta(days=1))]
            
                        
            for f in range(len(self.pred_vars)):
                feature_filtered = time_filtered[["latitude", "longitude", self.pred_vars[f]]]
                
                for i, (lon, lat) in enumerate(grid_points):                    
                    nearest_pred_idx = (
                            (feature_filtered["latitude"] - lat) ** 2 +
                            (feature_filtered["longitude"] - lon) ** 2
                        ).argmin()
                    nearest_pred = feature_filtered.iloc[nearest_pred_idx]

                    daily_avg = nearest_pred[self.pred_vars[f]].mean()
        
                    dyn_seq[t, f, i] = daily_avg 
                    
              
        
        
        # ############################## Static feature ##############################
        
        static_seq = np.zeros((len(self.static_vars), len(grid_points)))
        
        for f in range(len(self.static_vars)):
            feature_filtered = self.static_predictors[["latitude", "longitude", self.pred_vars[f]]]
            
            for i, (lon, lat) in enumerate(grid_points):
                nearest_idx = (
                            (feature_filtered["latitude"] - lat) ** 2 +
                            (feature_filtered["longitude"] - lon) ** 2
                        ).argmin()
                
                nearest_pred = feature_filtered.iloc[nearest_idx]
                
                stat_avg = nearest_pred[self.pred_vars[f]]
                 
                static_seq[f, i] = stat_avg 
                   
  

        # Define transformations
        dyn_transform = Compose([
                NormalizeTransform(self.pred_vars, self.pred_mean, self.pred_std, feature_type='dynamic'),
                ReshapeTransformCNN(self.seq_len, len(self.pred_vars), self.H, self.W)
            ])

        static_transform = Compose([
                NormalizeTransform(self.static_vars, self.static_mean, self.static_std, feature_type='static'),
                ReshapeTransformCNN(self.seq_len, len(self.static_vars), self.H, self.W, is_static=True)
            ])
        
        target_transform = Compose([
            # targets - target_means) / target_stds
                # LogTransform()
            ])
        
        

        dyn_seq = dyn_transform(dyn_seq) if dyn_transform else dyn_seq
        static_seq = static_transform(static_seq) if static_transform else static_seq
        # target_seq = target_transform(target_seq) if target_transform else target_seq
        target_seq = (target_seq - self.target_mean) / self.target_std
        
        
        dyn_predictor_tensor = torch.tensor(dyn_seq, dtype=torch.float32)
        static_predictor_tensor = torch.tensor(static_seq, dtype=torch.float32)
        target_tensor = torch.tensor(target_seq, dtype=torch.float32)
        
        
        dyn_tensor = dyn_predictor_tensor.to(self.device)
        static_predictor_tensor = static_predictor_tensor.to(self.device)
        target_tensor = target_tensor.to(self.device)
        
        return dyn_tensor, static_predictor_tensor, target_tensor, target_east, target_north
    
    

    def _generate_grid(self, easting, northing):
        """
        Generate a grid of points within a square bounding box centered at the target MP.
    
        Parameters:
        - easting: X-coordinate (meters) of MP.
        - northing: Y-coordinate (meters).
    
        Returns:
        - grid_points (numpy array): Array of (x, y) coordinates.
        """
        half_size = 200
        step_size = 100 # Distance between grid points
    
        # Create coordinate ranges
        x_values = np.arange(easting - half_size, easting + half_size + step_size, step_size)
        y_values = np.arange(northing - half_size, northing + half_size + step_size, step_size)
    
        # Create meshgrid
        xx, yy = np.meshgrid(x_values, y_values)
    
        # Stack into (x, y) coordinate pairs
        grid_points = np.column_stack([xx.ravel(), yy.ravel()])
    
        return grid_points
    




    
    def _get_neighbors(self, target_x, target_y, max_distance):
        """Find neighboring target points within a specified max distance (in meters)."""
        
        max_distance = 200  
        
        mask = (
                (self.target_displacement["easting"] >= target_x - max_distance) &
                (self.target_displacement["easting"] <= target_x + max_distance) &
                (self.target_displacement["northing"] >= target_y - max_distance) &
                (self.target_displacement["northing"] <= target_y + max_distance)
            )
        
        
        neighbors = list(zip(self.target_displacement.loc[mask, "easting"], 
                         self.target_displacement.loc[mask, "northing"]))
        
        num_neighbors = len(neighbors)
        
        if num_neighbors < 9:
            while len(neighbors) < 9:
                neighbors.append((target_x, target_y))

        elif num_neighbors > 9:
            neighbors = neighbors[:9]          
        
        return neighbors


    
    
     
    def _get_split(self, split, target):
        """Split the data into train, validation, and test sets."""
        # Logic to split based on position
        
        train_ratio, val_ratio, test_ratio = 0.7, 0.2, 0.1
        self.validate_ratios(train_ratio, val_ratio, test_ratio)
        
        num_points = len(target)
        
        train_end_idx = int(train_ratio * num_points)
        val_end_idx = int((train_ratio + val_ratio) * num_points)
        
        if self.split == 'train':
            return target[:train_end_idx]
        elif self.split == 'val':
            return target[train_end_idx:val_end_idx]
        elif self.split == 'test':
            return target[val_end_idx:]
        else:
            print('Using all data as input for the model without splitting.')
            return target




    def _get_time_steps(self, split, target):
        """Split the data into train, validation, and test sets."""
        # Example logic to split based on time
        self.num_timestamps = len(self.target_times)
        train_ratio, val_ratio, test_ratio = 0.7, 0.2, 0.1
        self.validate_ratios(train_ratio, val_ratio, test_ratio)
        
        # Compute the index split points
        train_end_idx = int(train_ratio * self.num_timestamps)
        val_end_idx = int((train_ratio + val_ratio) * self.num_timestamps)
        
        if self.split == 'train':
            return np.arange(train_end_idx)
        elif self.split == 'val':
            return np.arange(train_end_idx, val_end_idx)
        elif self.split == 'test':
            return np.arange(val_end_idx, self.num_timestamps)
        else:
            print('Using all data as input for the model without splitting.')
            return np.arange(self.num_timestamps)



    
    def _generate_sequences(self):
        """Generate the starting index of the split at a point."""
        return [start_time_idx for start_time_idx in self.time_steps if start_time_idx + self.seq_len <= self.num_timestamps]





    @staticmethod
    def validate_ratios(train_ratio, val_ratio, test_ratio):
        """
        Validate that the provided ratios sum up to 1.
        """
        total_ratio = train_ratio + val_ratio + test_ratio
        if not np.isclose(total_ratio, 1.0):
            raise ValueError(f"Ratios must sum to 1.0, but got {total_ratio:.2f}")


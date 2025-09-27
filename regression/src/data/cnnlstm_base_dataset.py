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
    def __init__(self, data_paths, aoi_path, split, target_displacement, target_times, predictors, pred_vars, pred_mean, pred_std, device='cuda', seq_len=30, max_distance=150):
        
        """
        Args:
            data_paths (list of str): Paths to the NetCDF files.
            aoi_path (list of str): Paths to the file containing AOI definition.
            transform (callable, optional): Optional transform to apply to the predictor.
            target_transform (callable, optional): Optional transform to apply to target.
            device (str): Device to use ('cuda' or 'cpu').
            seq_len (int): Length of each time series sequence.
            max_distance_m (int): Maximum distance (in m) to consider for the grid size.
            grid_size (int): Number of nearest neighbors to consider within the spatial grid.
        """
        
        self.data_paths = data_paths
        self.aoi_path = aoi_path
        self.split = split
        self.pred_mean = pred_mean, 
        self.pred_std = pred_std, 
        self.device = device
        self.seq_len = seq_len
        self.max_distance = max_distance
        

        # Load displacement and predictor data as before        
        self.target_displacement = target_displacement
        self.target_times = target_times
        self.predictors = predictors
        self.pred_vars = pred_vars
        
        self.num_points, _ = self.target_displacement.shape
        self.H = self.W = 3
        
        
        
        # Split data into train, val, test
        self.time_steps = self._get_time_steps()
        
        self.sequences = self._generate_sequences()
        
    
    def __len__(self):
        """Return the total number of sequences."""
        return len(self.sequences) * self.num_points
    
    

    def __getitem__(self, idx):
        """Retrieve a time series segment for a point."""
        
        spatial_idx = idx // len(self.sequences)  
        sequence_idx = idx % len(self.sequences)
        
        start_time_idx = self.sequences[sequence_idx]
        end_time_idx = start_time_idx + self.seq_len
        
        
        target_point = self.target_displacement.iloc[spatial_idx]
        target_east, target_north = target_point["easting"], target_point["northing"]
        
        
        target_seq = self.target_displacement.iloc[spatial_idx, start_time_idx:end_time_idx].values 
        target_dates = pd.to_datetime(self.target_times[start_time_idx:end_time_idx], format="%Y%m%d").date

        neighbors = self._get_neighbors(target_east, target_north, self.max_distance)
        
        

        
        self.predictors["valid_time"] = pd.to_datetime(self.predictors["valid_time"])
        
        num_features = len(self.pred_vars)+1
        
        if neighbors is not None:
            num_neighbors = len(neighbors)
            self.W = self.H = int(np.sqrt(len(neighbors)))        
        
        
        # Get the predictor values for each of the neighboring target points
        predictor_seq = []
        for target_date in target_dates:

            daily_preds = self.predictors[
                (self.predictors["valid_time"].dt.date == target_date)
            ]
            


            
            print(f"daily_preds shape: {daily_preds.shape}")
            
            if daily_preds.empty:
                continue
                # raise ValueError(f"daily_preds is empty for target_date {target_date}")
            
            
            time_step_neighbours = []

            for neighbor in neighbors:
                nearest_pred_idx = (
                            (daily_preds["longitude"] - neighbor[0]) ** 2 +
                            (daily_preds["latitude"] - neighbor[1]) ** 2
                        ).argmin()
                nearest_pred = daily_preds.iloc[nearest_pred_idx]

                neighbour_predictors = {}
            
                for var in self.pred_vars:
                # nearest_pred_idx = (
                #                     (daily_preds["longitude"] - target_east) ** 2 +
                #                     (daily_preds["latitude"] - target_north) ** 2
                #                 ).argmin()
                # nearest_pred = daily_preds.iloc[nearest_pred_idx]
            
                # if not nearest_pred.empty:
                #     predictors[f"{var}_target"] = nearest_pred[var].mean()
                # else:
                #     predictors[f"{var}_target"] = np.nan

                    neighbour_predictors[f"{var}"] = nearest_pred[var].mean()
                

                

                    
                    # if not nearest_pred.empty:
                    #     predictors[f"{var}_{neighbor}"] = nearest_pred[var].mean()
                    # else:
                    #     predictors[f"{var}_{neighbor}"] = np.nan
                        
            
                    # Add time feature to the predictors
                    time_feature = int(target_date.strftime("%Y%m%d"))
                    neighbour_predictors[f"time_numeric"] = time_feature

                time_step_neighbours.append(list(neighbour_predictors.values()))
    
            
            time_step_neighbours = np.array(time_step_neighbours)

            
            
            
            predictor_seq.append(list(time_step_neighbours))

            
        # time_feature_index = list(predictors.keys()).index("time_numeric")
            
        predictor_seq = np.nan_to_num(predictor_seq, nan=0.0)

        predictor_seq = np.array(predictor_seq)

        predictor_seq = predictor_seq.reshape(len(target_dates), len(neighbors), len(self.pred_vars) + 1)

        
        print(f"predictor_seq shape: {predictor_seq.shape if isinstance(predictor_seq, np.ndarray) else type(predictor_seq)}")

        # Define transformations
        

        transform = Compose([
                # NormalizeTransform(num_features, self.pred_mean, self.pred_std),
                ReshapeTransformCNN(num_features, self.H, self.W, seq_len=self.seq_len)
            ])
        
        target_transform = Compose([
                LogTransform()
            ])
        
        

        predictor_seq = transform(predictor_seq) if transform else predictor_seq
        target = target_transform(target_seq) if target_transform else target_seq
        

        predictor_tensor = torch.tensor(predictor_seq, dtype=torch.float32)
        target_tensor = torch.tensor(target, dtype=torch.float32)

        target_tensor = target_tensor.to(self.device)
        predictor_tensor = predictor_tensor.to(self.device)
    
        return predictor_tensor, target_tensor
    
    
    # def _get_neighbors(self, target_x, target_y, max_distance):
    #     """Find neighboring target points within a specified max distance (in m)."""
    #     neighbors = []
        
    #     for i, row in self.target_displacement.iterrows():
    #         neighbor_x, neighbor_y = row["easting"], row["northing"]
    #         distance = np.sqrt((neighbor_x - target_x)**2 + (neighbor_y - target_y)**2)

    #         if distance <= max_distance:
    #             neighbors.append((neighbor_x, neighbor_y))

    #     return neighbors
    
    
    def _get_neighbors(self, target_x, target_y, max_distance):
        """Find neighboring target points within a specified max distance (in meters)."""
        
        max_distance = max_distance  
        
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


    
    
 
    def _get_time_steps(self):
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


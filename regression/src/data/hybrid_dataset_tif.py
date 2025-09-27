# -*- coding: utf-8 -*-
"""
Created on Wed Apr  9 09:36:45 2025

@author: gmfet
"""

import torch
from torch.utils.data import Dataset
import numpy as np
import os
import pandas as pd
from tqdm import tqdm

from statsmodels.tsa.seasonal import seasonal_decompose

class VGDDataset(Dataset):
    def __init__(self, metadata_file, data_dir, data_transforms, split, seq_len=20):
        self.metadata = pd.read_csv(metadata_file)
        self.data_dir = data_dir
        self.seq_len = seq_len
        self.dyn_transform = data_transforms[0]
        self.static_transform = data_transforms[1]
        self.target_transform = data_transforms[2]
        
        # if split=='train':
        #     self.metadata = self.metadata.iloc[:14000]

        # print(f"building {split} samples ")
        # if split=='val':
        #     self.metadata = self.metadata.iloc[:3000]
        #     # self.metadata = self.metadata.sample(n=500, random_state=42)
            
        # if split=='test':
        #     self.metadata = self.metadata.iloc[:3000]
        #     # self.metadata = self.metadata.sample(n=500, random_state=42)
                  
            
            #     self.metadata = self.metadata.sample(n=1200, random_state=42)
            

            

            
        #     with xr.open_mfdataset('../data/static/*.nc', engine='netcdf4') as static_ds:
        #         print(static_ds)
        #         self.static = static_ds
            
                
            with xr.open_mfdataset('../data/dynamic/*.nc', engine='netcdf4', combine='by_coords', drop_variables=['spatial_ref', 'ssm_noise']) as dynamic_ds:
                self.dynamic = dynamic_ds
                print(dynamic_ds)
           

        #     # if split=='train':
        #     #     self.metadata = self.metadata.iloc[:186210]

        #     print(f"building {split} samples ")

        #     self.target = np.load(os.path.join(self.data_dir,'targets.npy'), mmap_mode='r')
          
        #     self.samples = []  

        #     for i, entry in tqdm(self.metadata.iterrows(), total=len(self.metadata)):
        #         east_pos = entry["easting"]
        #         north_pos = entry["northing"]
        #         T = self.target.shape[1]

        #         for t in range(T - seq_len):  # For seq2one
        #             self.samples.append((i, t, east_pos, north_pos))

        # def __len__(self):
        #     return len(self.samples)
        
        
        
        
        
            
            
            
        #     space = 7000
        # elif split=='train':
        #     # self.metadata = self.metadata.iloc[:200]
        #     space = 2000
        # if split=='test':
        #     # self.metadata = self.metadata.iloc[:100]
        #     space = 1000
        # self.metadata = self.metadata.iloc[:space]
        
        self.static = np.load(os.path.join(self.data_dir, 'static.npy'), mmap_mode='r')
        self.dynamic = np.load(os.path.join(self.data_dir, 'dynamic.npy'), mmap_mode='r')
        self.target = np.load(os.path.join(self.data_dir,'targets.npy'), mmap_mode='r')
      
        self.samples = []  

        for i, entry in tqdm(self.metadata.iterrows(), total=len(self.metadata)):
            east_pos = entry["easting"]
            north_pos = entry["northing"]
            T = self.target.shape[1]

            for t in range(T - seq_len):  # For seq2one
                self.samples.append((i, t, east_pos, north_pos))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        idx, t, easting, northing = self.samples[idx]
        
        # Slice dynamic and target
        dynamic_seq = self.dynamic[idx, t : t + self.seq_len]  # [seq_len, C, H, W]
        target_seq = self.target[idx, t + self.seq_len]  # Predict next step
        static = self.static[idx]

        dynamic_seq = self.dyn_transform(dynamic_seq) if self.dyn_transform else dynamic_seq
        static = self.static_transform(static) if self.static_transform else static
        target_seq = self.target_transform(target_seq) if self.target_transform else target_seq

        dynamic_seq = torch.tensor(dynamic_seq, dtype=torch.float)
        static = torch.tensor(static, dtype=torch.float)
        target_seq = torch.tensor(target_seq, dtype=torch.float)

        return dynamic_seq, static, target_seq, easting, northing

    
    
    @staticmethod
    def compute_stats(data_path):
        
        static = np.load(os.path.join(data_path, 'static.npy'), mmap_mode='r')  # [C, H, W]
        dynamic = np.load(os.path.join(data_path, 'dynamic.npy'), mmap_mode='r')  # [T, C, H, W]
        target = np.load(os.path.join(data_path,'targets.npy'), mmap_mode='r')  # [T]
        print(static.shape)
        static = static[:14000]
        dynamic= dynamic[:3000]
        target= target[:3000]
          
         # Compute means
        static_mean = np.nanmean(static, axis=(0, 2, 3))         # shape: [C_static]
        dynamic_mean = np.nanmean(dynamic.astype(np.float32), axis=(0, 1, 3, 4))    # shape: [C_dynamic]
        target_mean = np.nanmean(target, axis=(0, 1))            # scalar -> [1]
        
        # Compute standard deviations
        static_std = np.nanstd(static, axis=(0, 2, 3))           # shape: [C_static]
        dynamic_std = np.nanstd(dynamic.astype(np.float32), axis=(0, 1, 3, 4))      # shape: [C_dynamic]
        target_std = np.nanstd(target, axis=(0, 1))              # scalar -> [1]

    
        # Aggregate by mean across all samples
        return {
            'mean': {
                'static': static_mean.tolist(),
                'dynamic': dynamic_mean.tolist(),
                'target': target_mean.tolist()
            },
            'std': {
                'static': static_std.tolist(),
                'dynamic': dynamic_std.tolist(),
                'target': target_std.tolist()
            }
        }
    
    
    
        
        
        
    def _decompose_(self, data):
        '''
        Source:
            https://machinelearningmastery.com/decompose-time-series-data-trend-seasonality/
            https://www.statsmodels.org/dev/generated/statsmodels.tsa.seasonal.seasonal_decompose.html
            https://www.geeksforgeeks.org/seasonality-detection-in-time-series-data/
            https://otexts.com/fpp3/
            
            residual:
                https://www.nature.com/articles/s41598-021-96674-0
                https://openaccess.thecvf.com/content/CVPR2024W/EarthVision/papers/Ebel_Implicit_Assimilation_of_Sparse_In_Situ_Data_for_Dense__CVPRW_2024_paper.pdf
            
            Lag:
                https://www.geeksforgeeks.org/what-is-lag-in-time-series-forecasting/


        Parameters
        ----------
        data : TYPE
            DESCRIPTION.

        Returns
        -------
        trend, seasonal, residual

        '''

        if data.ndim > 1:
            num_time = data.shape[0]
            num_vars = data.shape[1]
            height = data.shape[2]
            width = data.shape[3]
            trend = np.empty_like(data)
            seasonal = np.empty_like(data)
            residual = np.empty_like(data)

            for var_idx in range(num_vars):
                for h_idx in range(height):
                    for w_idx in range(width):
                        decomposition_additive = seasonal_decompose(data[:, var_idx, h_idx, w_idx], model='additive', period=61,
                                                                            extrapolate_trend='freq')
                        trend[:, var_idx, h_idx, w_idx] = decomposition_additive.trend
                        seasonal[:, var_idx, h_idx, w_idx] = decomposition_additive.seasonal
                        residual[:, var_idx, h_idx, w_idx] = decomposition_additive.resid
        else:  # 1-dimensional target data
            decomposition_additive = seasonal_decompose(data, model='additive', period=61, extrapolate_trend='freq')
            trend = decomposition_additive.trend
            seasonal = decomposition_additive.seasonal
            residual = decomposition_additive.resid
        return trend, seasonal, residual
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 09:25:04 2024

@author: 39351
"""

import numpy as np


class MinMaxTransform:
    """
    Min-Max scale the data to [0, 1].
    """
    def __init__(self, features, mins, maxs, feature_type):
        """
        Args:
            features (list): List of feature names.
            mins (list or array): Minimum values for each feature.
            maxs (list or array): Maximum values for each feature.
            feature_type (str): 'dynamic', 'static', or 'target'.
        """
        self.features = features
        self.mins = mins
        self.maxs = maxs
        self.feature_type = feature_type

    def __call__(self, sample):
        """
        Apply min-max scaling.

        Scaling formula:
            scaled = (x - min) / (max - min)

        Args:
            sample (numpy array): The data sample to be scaled.

        Returns:
            numpy array: The scaled sample.
        """
        scaled_features = sample.copy()

        if self.feature_type == 'dynamic':
            for f in range(len(self.features)):
                band = sample[:, f, :, :]
                band = np.where(np.isnan(band), self.mins[f], band)
                denom = (self.maxs[f] - self.mins[f]) or 1.0
                scaled_features[:, f, :, :] = (band - self.mins[f]) / denom

        elif self.feature_type == 'static':
            for f in range(len(self.features)):
                band = sample[f, :, :]
                band = np.where(np.isnan(band), self.mins[f], band)
                denom = (self.maxs[f] - self.mins[f]) or 1.0
                scaled_features[f, :, :] = (band - self.mins[f]) / denom

        elif self.feature_type == 'target':
            scaled_features = np.where(np.isnan(sample), self.mins, sample)
            denom = (self.maxs - self.mins) or 1.0
            scaled_features = (scaled_features - self.mins) / denom

        return scaled_features
    




class NormalizeTransform:
    """
    Normalize the data to zero mean and unit variance.
    """
    def __init__(self, features, means, stds, feature_type):
        
        self.features = features
        self.means = means
        self.stds = stds
        self.feature_type = feature_type
        
 
    
    def __call__(self, sample):
        """
        
        Standardization (Z-score Normalization)

            Standardization transforms the data to have a mean of 0 and a standard deviation of 1. 
            It's helpful when you assume your data follows a normal distribution. This is being used
            for the data (only predictor) due to this assumption.
            
            Option:
                If the dataset is assumed to have outliers, the robust normalisation technique could be used
                    def robust_normalize(data):
                        median = np.median(data)
                        iqr = np.percentile(data, 75) - np.percentile(data, 25)
                        return (data - median) / iqr

        Args:
            sample (numpy array): The data sample to be normalized.
        
        Returns:
            numpy array: The normalized sample.
        """

        
        normalized_features = sample.copy()
        
        if self.feature_type == 'dynamic':
            for f in range(len(self.features)):
                band = sample[:, f, :, :]
                band = np.where(np.isnan(band), self.means[f], band)
                normalized_features[:, f, :, :] = (band - self.means[f]) / self.stds[f]
        
        elif self.feature_type == 'static':
            for f in range(len(self.features)):
                band = sample[f, :, :]
                band = np.where(np.isnan(band), self.means[f], band)
                normalized_features[f, :, :] = (band - self.means[f]) / self.stds[f]
        

        
        elif self.feature_type == 'target':
            normalized_features = np.where(np.isnan(sample), self.means, sample)
            normalized_features = (normalized_features - self.means) / self.stds
        
        return normalized_features
    def inverse(self, sample):
        """
        Denormalize the target back to the original scale.
        """
        if self.feature_type == 'target':
            return sample * self.stds + self.means
        return sample

    
                
        




class ReshapeTransform:
    """     Reshape the input data to a specified shape, considering batch size and features. """
    
    def __init__(self, input_size, seq_len):
        """
            seq_len:
                the number of time steps you want the model to look at at once
                Setting seq_len to -1 is to indicate automatic inference of that dimension.
                when reshaping data, if you set one of the dimensions to -1, PyTorch will 
                automatically calculate the size for that dimension to ensure the total number 
                of elements in the tensor remains the same.
                
                source:
                    https://pytorch.org/docs/stable/generated/torch.reshape.html
                    https://machinelearningmastery.com/reshape-input-data-long-short-term-memory-networks-keras/

            input_size:
                the number of features in the data         
        """
        self.seq_len = seq_len 
        self.input_size = input_size
        
    
    def __call__(self, sample):
        """
        Apply the transformation (reshape) to the input data.
        
        Args:
            sample (numpy array): The data sample to be reshaped.
        
        Returns:
            numpy array: The reshaped sample.
        """
        
        if isinstance(sample, (int, float, np.float32, np.float64)):  
            sample = np.array([sample], dtype=np.float32)  # Convert scalar to array
        
        if not isinstance(sample, np.ndarray):
            raise TypeError("Input sample must be a numpy array or convertible to an array.")
        
        if len(sample.shape) == 1:  # 1D input
            return sample.reshape(-1, 1)  # Keep it flexible
        
        elif len(sample.shape) == 2:  
            return sample.reshape(self.seq_len, self.input_size)
        
        elif len(sample.shape) == 3:  
            return sample.reshape(sample.shape[0], self.seq_len, self.input_size)


class LogTransform:
    """ Apply log transformation to the target displacement. 
    
    Procedure:
        1 Apply log transformation to the target data.
        2 Apply Z-score normalization to the log-transformed target values to ensure 
          that the target values are on a scale compatible with the model's training procedure.

    
    """
    def __init__(self, epsilon=1e-6):
        self.epsilon = epsilon

    def __call__(self, sample):
        if isinstance(sample, (int, float, np.float32, np.float64)):  
            sample = np.array([sample], dtype=np.float32)  # Convert scalar to array

        if not isinstance(sample, np.ndarray):
            raise TypeError("Input sample must be a numpy array or convertible to an array.")

        sample = np.clip(sample + self.epsilon, a_min=1e-10, a_max=None)  
        sample = np.log(sample)

        return sample 

    
class ReshapeTransformCNN:
    """     Reshape the input data to a specified shape, considering batch size and features. """
    
    def __init__(self, seq_len, features, H, W, is_static=False):
        """
            seq_len:
                the number of time steps you want the model to look at at once
                Setting seq_len to -1 is to indicate automatic inference of that dimension.
                when reshaping data, if you set one of the dimensions to -1, PyTorch will 
                automatically calculate the size for that dimension to ensure the total number 
                of elements in the tensor remains the same.
                
                source:
                    https://pytorch.org/docs/stable/generated/torch.reshape.html
                    https://machinelearningmastery.com/reshape-input-data-long-short-term-memory-networks-keras/

            input_size:
                the number of features in the data         
        """
        self.seq_len = seq_len 
        self.features = features # or the number of features. treat each feature as a layer channel
        self.H = H
        self.W = W
        self.is_static = is_static
        
    
    def __call__(self, sample):
        """
        Apply the transformation (reshape) to the input data.
        
        Args:
            sample (numpy array): The data sample to be reshaped.
        
        Returns:
            numpy array: The reshaped sample.
        """
        
        if isinstance(sample, (int, float, np.float32, np.float64)):  
            sample = np.array([sample], dtype=np.float32)  # Convert scalar to array
        
        if not isinstance(sample, np.ndarray):
            raise TypeError("Input sample must be a numpy array or convertible to an array.")
            

        if self.is_static:
            return sample.reshape(self.features, self.H, self.W)
        elif not self.is_static:
            return sample.reshape(self.seq_len, self.features, self.H, self.W)

        else:
            raise ValueError("Specify if the data is static of spatio-temporal.")



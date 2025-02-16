# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 09:25:04 2024

@author: 39351
"""

import numpy as np

class NormalizeTransform:
    """
    Normalize the data to zero mean and unit variance.
    """
    def __init__(self):
        pass  
    
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
        if isinstance(sample, (int, float, np.float32, np.float64)):  
            sample = np.array([sample], dtype=np.float32)

        if not isinstance(sample, np.ndarray):
            raise TypeError("Input sample must be a numpy array.")
        
        mean = np.mean(sample)
        std = np.std(sample)
        
        if std == 0:
            return sample - mean  # Center the data around the mean
        
        return (sample - mean) / std


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
            
        print(f"ReshapeTransform - Input shape: {sample.shape}, Expected shape: {self.seq_len}, {self.input_size}")
        
        # For input shape (seq_len,) force to (seq_len, 1) 
        if len(sample.shape) == 1:  
            seq_len = sample.shape[0]  
            return sample.reshape(seq_len, 1)  
        
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
        self.epsilon = epsilon  # Small constant to avoid log(0)

    def __call__(self, sample):
      if isinstance(sample, (int, float, np.float32, np.float64)):  
            sample = np.array([sample], dtype=np.float32)  # Convert scalar to array
        
      if not isinstance(sample, np.ndarray):
          raise TypeError("Input sample must be a numpy array or convertible to an array.")

      sample = np.clip(sample + self.epsilon, a_min=1e-10, a_max=None)  # Clip to avoid log(0) or negative values
    
            
      return np.log(sample)



# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 13:33:07 2025

@author: 39351
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class VGDModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, seq_len, H, W, num_layers=3, dropout=0.2):
        
        
        """
        CNN-LSTM model for Vertical Ground Displacement (VGD) prediction.
        
        source:
            https://colah.github.io/posts/2015-08-Understanding-LSTMs/
            https://machinelearningmastery.com/lstms-with-python/
            https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html
            
            https://machinelearningmastery.com/faq/single-faq/what-is-the-difference-between-samples-timesteps-and-features-for-lstm-input/
            
            CNN-LSTM:
                https://arxiv.org/pdf/1506.04214
                https://arxiv.org/pdf/1411.4389
                https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/43455.pdf

                https://github.com/leizhang-geo/CNN-LSTM_for_DSM/blob/main/models.py
                https://github.com/andersonsam/cnn_lstm_era/blob/master/non_contributing_areas.ipynb
                https://tc.copernicus.org/articles/16/1447/2022/
                https://machinelearningmastery.com/cnn-long-short-term-memory-networks/
        
        Args:
            input_size (int): The number of neighboring points multiplied by the number of features at each neighbor point of the target MP.
            hidden_size (int): The number of hidden units in each LSTM layer.
            output_size (int): The number of output features.
            seq_len (int): Length of the input time series.
            H, W (int): Height and width of input grid.
            num_layers (int): The number of LSTM layers.
            dropout (float): Dropout rate for regularization between LSTM layers.
            
        """
        super().__init__()
        
        
        # Define the CNN layer
        self.conv1 = nn.Conv2d(in_channels=input_size, out_channels=16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten_size = 32 * (H // 2) * (W // 2) 
        


        # Define the LSTM layer
        self.lstm = nn.LSTM(input_size=self.flatten_size, 
                            hidden_size=hidden_size, 
                            num_layers=num_layers, 
                            batch_first=True,  
                            dropout=dropout,
                            bidirectional = False # Change to true for bidirectional LSTM
                            )
        

        # Fully connected layer to map LSTM output to the target
        self.fc = nn.Linear(hidden_size, output_size)
        
        
    def forward(self, x):
        """
        Forward pass for the CNN-LSTM model.
        
        Args:
            x (Tensor): Input tensor of shape (batch_size, seq_len, input_channels, H, W).
        
        Returns:
            Tensor: Output tensor of shape (batch_size, seq_len, output_size).
            
            
        """
        print(f"Input shape: {x.shape}")
        
        
        batch_size, seq_len, channels, H, W = x.shape
        
        # Process each time step through CNN
        cnn_features = []
        for t in range(seq_len):
            xt = x[:, t, :, :, :]  # Extract spatial features at time t
            xt = F.relu(self.conv1(xt))
            xt = F.relu(self.conv2(xt))
            xt = self.pool(xt)
            xt = torch.flatten(xt, start_dim=1) 
            cnn_features.append(xt)

        # Stack extracted CNN features into (batch_size, seq_len, flattened_features)
        cnn_features = torch.stack(cnn_features, dim=1)

        # Pass CNN features through LSTM
        lstm_out, _ = self.lstm(cnn_features) # Shape: (batch_size, seq_len, hidden_size)
        
        
        
        # Pass through the fully connected layer
        out = self.fc(lstm_out) 
        
        return out



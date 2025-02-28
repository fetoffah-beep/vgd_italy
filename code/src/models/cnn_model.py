# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 13:18:17 2025

@author: 39351
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class VGDCNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_dim):
        
        
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
        
        self.seq_model = nn.Sequential(
            nn.Conv2d(in_channels=input_size, out_channels=16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            F.relu(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            F.relu(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Conv2d(32, 64, kernel_size=5, padding=1),
            nn.BatchNorm2d(64),
            F.relu(),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        
        # Fully connected layers to map output to the target
        self.fc1 = nn.Linear(64 * 4 * 4, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_dim) 
        
        
        
    def forward(self, static_input):
        """
        Forward pass for the CNN-ConvLSTM model.
        
        Args:
            static_input (Tensor): Input tensor of shape (batch_size, features, H, W).
        
            dynamic_input (Tensor): Input tensor of shape (batch_size, seq_len, input_channels, H, W).
        
        Returns:
            Tensor: Output tensor of shape (batch_size, seq_len, target_feature).
            
            
        """
        
        
        #################I CNN for static features #################
        
        x_cnn = static_input[:, :, :, :]
        x_cnn = self.seq_model(x_cnn)
        x_cnn = torch.flatten(x_cnn, 1)  # flatten all dimensions except batch
        x_cnn = F.relu(self.fc1(x_cnn))
        static_out = self.fc2(x_cnn) 
        
        return static_out

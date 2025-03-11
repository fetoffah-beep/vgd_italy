# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 12:06:02 2025

@author: 39351

Source:
    ConvLSTM:
        https://medium.com/neuronio/an-introduction-to-convlstm-55c9025563a7
        https://github.com/ndrplz/ConvLSTM_pytorch/blob/master/convlstm.py
        https://arxiv.org/pdf/1506.04214v2
    CNN:
        https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
        
    Batch normalisation:
        https://arxiv.org/pdf/1502.03167
        

"""
import torch
import torch.nn as nn
from .convlstm import ConvLSTM
import torch.nn.functional as F


class VGDModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=3):
        """
        CNN-ConvLSTM model for Vertical Ground Displacement (VGD) prediction.
        
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
            seq_len (int): Number of time-steps.
            output_size (int): The number of output features.
            H, W (int): Height and width of input grid.
            num_layers (int): The number of LSTM layers.
        """
        super(VGDModel, self).__init__()
        
        self.input_size = input_size
        self.output_size = 1
        self.hidden_dim = hidden_size
        self.kernel_size = (3,3)
        self.num_layers=3
        
        
        
        self.seq_model = nn.Sequential(
            nn.Conv2d(in_channels=input_size, out_channels=16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.AdaptiveAvgPool2d((4, 4)),
            # nn.MaxPool2d(kernel_size=2, stride=1),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.AdaptiveAvgPool2d((4, 4)),
            # nn.MaxPool2d(kernel_size=2, stride=1),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.AdaptiveAvgPool2d((4, 4)),
            # nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        
        

        self.dynamic_model = ConvLSTM(self.input_size, 
                                      self.hidden_dim, 
                                      self.kernel_size, 
                                      self.num_layers,
                                      batch_first=True
                            )
        # Fully connected layers to map output to the target        
        self.fc_cnn = nn.Sequential(
            nn.Linear(64*4*4, self.hidden_dim),
            # nn.ReLU(),
            # nn.Dropout(0.1),
            # nn.Linear(self.hidden_dim, self.output_size)
        )
        
        self.hybrid_fc = nn.Sequential(
            nn.Linear(self.hidden_dim*2, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_dim, self.output_size)
        )
        
         

    def forward(self, dynamic_input, static_input):
        """
            Forward pass for the CNN-ConvLSTM model.
            
            Args:
                static_input (Tensor): Input tensor of shape (batch_size, features, H, W).
            
                dynamic_input (Tensor): Input tensor of shape (batch_size, seq_len, features, H, W).
            
            Returns:
                Tensor: Output tensor of shape (batch_size, seq_len, target_feature).
            
            
        """
        
        timef  = dynamic_input.shape[1]
        
        # ################# ConvLSTM for Dynamic features #################
        dynamic_out, _ = self.dynamic_model(dynamic_input)
        # print(dynamic_out)
        # dynamic_out = torch.stack(dynamic_out, dim=1)
        dynamic_out = torch.mean(dynamic_out[-1], dim=(3, 4))
        dynamic_out = torch.flatten(dynamic_out, start_dim=2)  # flatten all dimensions except batch and time_step
        
        
        
        ################# CNN for static features #################
        x_cnn = static_input[:, :, :, :]
        x_cnn = self.seq_model(x_cnn)
        x_cnn = torch.flatten(x_cnn, 1)  # flatten all dimensions except batch and time_step
        x_cnn = self.fc_cnn(x_cnn)
    
        x_cnn = x_cnn.unsqueeze(1)
        static_out = x_cnn.repeat(1, timef, 1)
        
        # Non-empty tensors provided for concatenation must have the same shape, except in the cat dimension.
        # Concatenate the tensors along the feature dimension        
        dynstat_output = torch.cat([dynamic_out, static_out], dim=2)
        
        output = self.hybrid_fc(dynstat_output)
        return output

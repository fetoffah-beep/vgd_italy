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
    def __init__(self, dyn_feat, static_feat, hidden_size, output_size):
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
        
        self.output_size = 1
        self.hidden_dim = [32] #[128,64,32] #[128, 64, 64]
        self.kernel_size = (3,3)
        self.num_layers= 1 #3
        
        self.num_static_features = len(static_feat)
        self.num_dyn_features = len(dyn_feat)
        
        
        
        
        self.seq_model = nn.Sequential(
            nn.Conv2d(in_channels=self.num_static_features, out_channels=16, kernel_size=3, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.MaxPool2d(kernel_size=2, stride=1),
            
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.MaxPool2d(kernel_size=2, stride=1)
        )
        

        self.dynamic_model = ConvLSTM(self.num_dyn_features, 
                                      self.hidden_dim, 
                                      self.kernel_size, 
                                      self.num_layers,
                                      batch_first=True
                            )
        
        
        
        self.fusion_layer = ConvLSTM(
            input_dim=64, 
            hidden_dim=[32],
            kernel_size=(3, 3),
            num_layers=1,
            batch_first=True,
        )
        
        self.final_layer = nn.Conv2d(in_channels=32,
                                      out_channels=self.output_size,
                                      kernel_size=3,
                                      padding=1)
        
        
        

    def forward(self, dynamic_input, static_input):
        """
            Forward pass for the CNN-ConvLSTM model.
            
            Args:
                static_input (Tensor): Input tensor of shape (batch_size, features, H, W).
            
                dynamic_input (Tensor): Input tensor of shape (batch_size, seq_len, features, H, W).
            
            Returns:
                Tensor: Output tensor of shape (batch_size, seq_len, target_feature).
            
            
        """
        batch_size, time_frame, _, h, w = dynamic_input.size()
        
        
        # ################# ConvLSTM for Dynamic features #################
        dynamic_out_layers, _ = self.dynamic_model(dynamic_input)
        dynamic_out = dynamic_out_layers[-1]

        
        ################# CNN for static features #################
        x_cnn = static_input[:, :, :, :]
        x_cnn = self.seq_model(x_cnn)
        
        static_out = x_cnn.unsqueeze(1).repeat(1, time_frame, 1, 1, 1)
        

        ################# HYBRID MODEL #################           
        # Non-empty tensors provided for concatenation must have the same shape, except in the cat dimension.
        # Ensure the spatial dimensions match before concatenation

        fused_features = torch.cat([dynamic_out, static_out], dim=2)
        
          
        # Further fusion with a convolutional layer
        fused_output, _ = self.fusion_layer(fused_features)
        fused_output = fused_output[-1]
        
        next_timestep_features = fused_output[:, -1]
        
        
        
        # fused_output = fused_output.view(batch_size, time_frame, -1, fused_features.size(3), fused_features.size(4))

        # Predict displacement
        displacement_predictions = self.final_layer(next_timestep_features)
        output = F.adaptive_avg_pool2d(displacement_predictions, (1, 1)).squeeze(-1).squeeze(-1)  # [B, 1]
        
        # output = displacement_predictions.view(batch_size, time_frame, self.output_size, fused_output.size(3), fused_output.size(4))
        # output = output[:, -1, :, output.size(-2)//2, output.size(-1)//2]

        return output


      
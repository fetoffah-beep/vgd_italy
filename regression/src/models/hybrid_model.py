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
from line_profiler import profile

class VGDModel(nn.Module):
    @profile
    def __init__(self, dyn_feat, static_feat, cat_info, hidden_size, output_size):
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
        
        self.output_size = output_size
        self.hidden_dim = hidden_size
        self.kernel_size = (3,3)
        self.num_layers= len(hidden_size)
        
        cat_info = cat_info
        
        # If we’re in a hurry, one rule of thumb is to use the fourth root of the total number of unique 
        # categorical elements while another is that the embedding dimension should be approximately 1.6 
        # times the square root of the number of unique elements in the category, and no less than 600.
        # If we are doing hyperparameter tuning, it might be worth searching within this range.
        self.embeddings = nn.ModuleDict({
            var_name: nn.Embedding(num_embeddings=len(classes), embedding_dim=max(2, int(1.6 * (len(classes)**0.5))))
            for var_name, classes in cat_info.items()
        })

        # static_feat and dyn_feat contains the number of continuous and categorical variables.
        # To rightly set the number of features for the CNN and ConvLSTM, we have to reduce by the num of categorical features and 
        # add the total embedding dimensions for all features in static or dynamic group
        # i.e. self.num_static_features = static_feat - num_categorical_static_features + sum_of_embeddings for all_static_features
        static_cat = [name for name in cat_info.keys() if name != 'month']
        dynamic_cat = ['month']

        sum_emb_static = sum([self.embeddings[name].embedding_dim for name in static_cat])
        self.num_static_features = static_feat - len(static_cat) + sum_emb_static

        sum_emb_dyn = sum([self.embeddings[name].embedding_dim for name in dynamic_cat])
        self.num_dyn_features = dyn_feat - len(dynamic_cat) + sum_emb_dyn

        self.static_model = nn.Sequential(
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
                                      dropout=0.2,
                                      batch_first=True
                            )
        
        # self.fc_static= nn.Sequential(
        #                         nn.LazyLinear(256), #nn.Linear(64*5*5*25, 256),
        #                         nn.ReLU()
        #                     )
        
        # self.fc_dynamic = nn.Sequential(
        #                         nn.LazyLinear(256), # nn.Linear(256*5*5*25, 256),
        #                         nn.ReLU()
        #                     )
        
        self.fc_static = nn.Sequential(
                nn.Linear(32 * 5 * 5, 256),
                nn.ReLU()
            )
        
        self.fc_dynamic = nn.Sequential(
                nn.Linear(self.hidden_dim[-1] * 5 * 5 , 256),
                nn.ReLU()
            )

        
        self.fc_fused = nn.Sequential(
                                nn.Linear(256*2, 128),
                                nn.ReLU(),
                                # nn.Dropout(0.1),
                                nn.Linear(128, 1)
                            )
        

    @profile  
    def forward(self, continous_dynamic_input, continous_static_input, categorical_dynamic_input, categorical_static_input):
        """
            Forward pass for the CNN-ConvLSTM model.
            
            Args:
                static_input (Tensor): Input tensor of shape (batch_size, features, H, W).
            
                dynamic_input (Tensor): Input tensor of shape (batch_size, seq_len, features, H, W).
            
            Returns:
                Tensor: Output tensor of shape (batch_size, seq_len, target_feature).
            
            
        """
        batch_size, time_frame, _, _, _ = continous_dynamic_input.size()
        
        ################# CNN for static features #################
        if continous_static_input.numel() != 0:
             static_embedded = [continous_static_input]
        else:
             static_embedded = []
        
        
        cat_static_idx = 0
        for i, (varname, embedding) in enumerate(self.embeddings.items()):
            if varname != 'month':
                indices = categorical_static_input[:, cat_static_idx].long()
                embedded = embedding(indices).permute(0, 4, 1, 2, 3)
                static_embedded.append(embedded)
                cat_static_idx += 1
        input_static = torch.cat(static_embedded, dim=1).squeeze(2)
        cnn_output = self.static_model(input_static)
        # static2dyn = cnn_output.unsqueeze(1).repeat(1, time_frame, 1, 1, 1)


        # ################# ConvLSTM for Dynamic features #################
        dynamic_embedded = [continous_dynamic_input]
        month_indices = categorical_dynamic_input[:, 0].long()
        dyn_embedded = self.embeddings['month'](month_indices).permute(0, 4, 1, 2, 3)
        dynamic_embedded.append(dyn_embedded)
        # raw_cat_channel = categorical_dynamic_input.permute(0, 2, 1, 3, 4) 
        # dynamic_embedded.append(raw_cat_channel)
        input_dynamic = torch.cat(dynamic_embedded, dim=1).permute(0, 2, 1, 3, 4)
        dynamic_out_layers, _ = self.dynamic_model(input_dynamic)
        convlstm_output = dynamic_out_layers[-1]

        
        # ################# FULLY CONNECETED LAYERS ################# 
        # dynamic_out = convlstm_output.view(convlstm_output.shape[0], -1)  
        # static_out = static2dyn.view(static2dyn.shape[0], -1)
        static_flat = cnn_output.reshape(batch_size, -1)
        dynamic_last = convlstm_output[:, -1, :, :, :]
        dynamic_flat = dynamic_last.reshape(batch_size, -1)

        dynamic_out = self.fc_dynamic(dynamic_flat)
        static_out = self.fc_static(static_flat)


        # ################# HYBRID MODEL #################           
        # # Non-empty tensors provided for concatenation must have the same shape, except in the cat dimension.
        # # Ensure the spatial dimensions match before concatenation
        fused_features = torch.cat([dynamic_out, static_out], dim=1)
        # Further fusion with a convolutional layer
        output = self.fc_fused(fused_features) 

        return output
    
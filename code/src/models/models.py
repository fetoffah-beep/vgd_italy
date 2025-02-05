from sklearn.ensemble import RandomForestClassifier
from src.features.feature_engineering import create_features
from src.models.checkpoint import save_checkpoint, load_checkpoint
from src.data.dataloader import get_dataloader
import torch
import torch.optim as optim
import torch.nn as nn


class VGDModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=2, dropout=0.2):
        """
        Initialize the LSTM-based model for predicting VGD.
        
        Args:
            input_size (int): The number of features in the input data.
            hidden_size (int): The number of hidden units in each LSTM layer.
            output_size (int): The number of output features (typically 1 for regression tasks).
            num_layers (int): The number of LSTM layers.
            dropout (float): Dropout rate for regularization between LSTM layers.
        """
        super(VGDModel, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Define the LSTM layer
        self.lstm = nn.LSTM(input_size=input_size, 
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
        Forward pass for the LSTM model.
        
        Args:
            x (Tensor): Input tensor of shape (batch_size, seq_len, input_size).
        
        Returns:
            Tensor: Output tensor of shape (batch_size, output_size).
        """
        # Pass through the LSTM layer
        lstm_out, _ = self.lstm(x)  
        
        # Use the last hidden state for prediction
        last_hidden_state = lstm_out[:, -1, :] 
        
        # Pass through the fully connected layer
        out = self.fc(last_hidden_state) 
        
        return out



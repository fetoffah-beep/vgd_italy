
import torch.nn as nn


class VGDModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=3, dropout=0.2):
        """
        Initialize the LSTM-based model for predicting VGD.
        
        source:
            https://colah.github.io/posts/2015-08-Understanding-LSTMs/
            https://machinelearningmastery.com/lstms-with-python/
            https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html
            
            https://machinelearningmastery.com/faq/single-faq/what-is-the-difference-between-samples-timesteps-and-features-for-lstm-input/
        
        Args:
            input_size (int): The number of features in the input data.
            hidden_size (int): The number of hidden units in each LSTM layer.
            output_size (int): The number of output features (typically 1 for regression tasks).
            num_layers (int): The number of LSTM layers.
            dropout (float): Dropout rate for regularization between LSTM layers.
        """
        super().__init__()
        
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
        print(f"Input shape: {x.shape}")
        # Pass through the LSTM layer
        lstm_out, _ = self.lstm(x)  
        
        # # Use the all hidden state for prediction (sequence-to-sequence)
        # last_hidden_state = lstm_out[:, :, :] 
         
        
        # Pass through the fully connected layer
        out = self.fc(last_hidden_state) 
        
        return out



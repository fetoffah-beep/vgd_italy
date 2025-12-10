import torch
import torch.nn as nn
import torch.nn.functional as F

class LinearModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(LinearModel, self).__init__()

        self.lin_model = nn.LazyLinear(1)

    def forward(self, dynamic_input, static_input):
        
        batch_size, time_frame, _, h, w = dynamic_input.size()
        
        
        # ################# Input features #################       
        static_out = static_input.unsqueeze(1).repeat(1, time_frame, 1, 1, 1)
        
        dynamic_out = dynamic_input.view(dynamic_input.shape[0], -1)  
        static_out = static_out.view(static_out.shape[0], -1)

        # ################# SIMPLE LINEAR REGRESSION MODEL #################           
        fused_features = torch.cat([dynamic_out, static_out], dim=1)
        output = self.lin_model(fused_features)
    
        return output
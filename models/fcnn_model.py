#%%

# Import modules:
import torch
import torch.nn as nn
import torch.nn.functional as F
from complexPyTorch.complexLayers import ComplexLinear
from complexPyTorch.complexFunctions import complex_relu

#%%

# Define the model class:
class FCNNModel(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(FCNNModel, self).__init__()

        # Defining the architecture:
        # Input layer to first hidden layer:
        self.hidden_layers = nn.ModuleList([ComplexLinear(input_size, hidden_sizes[0])])
        # Intermediary, between hidden layers:
        self.hidden_layers.extend([ComplexLinear(hidden_sizes[i], hidden_sizes[i+1]) for i in range(len(hidden_sizes) - 1)])
        # Final hidden layer to output layer:
        self.output_layer = ComplexLinear(hidden_sizes[-1], output_size)

    # Defining the forward pass:
    def forward(self, x):
        # Forward pass through hidden layers with ReLU activation:
        for layer in self.hidden_layers:
            x = complex_relu(layer(x))

        # Forward pass through output layer:
        x = self.output_layer(x)
        return x
    
#%%
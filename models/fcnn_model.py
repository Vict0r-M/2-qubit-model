import torch
import torch.nn as nn
import torch.nn.functional as F

class FCNNModel(nn.Module):
    def __init__(self):
        super(FCNNModel, self).__init__()
        
        # Define the architecture:
        self.fc1 = nn.Linear(4, 128) # Input layer to first hidden layer
        self.fc2 = nn.Linear(128, 64) # First hidden layer to second hidden layer
        self.fc3 = nn.Linear(64, 32) # Second hidden layer to third hidden layer
        self.fc4 = nn.Linear(32, 3) # Third hidden layer to output layer

    def forward(self, x):
        # Forward pass through the network:
        x = F.relu(self.fc1(x)) # Activation function for first hidden layer
        x = F.relu(self.fc2(x)) # Activation function for second hidden layer
        x = F.relu(self.fc3(x)) # Activation function for third hidden layer
        x = self.fc4(x) # No activation function for output layer
        return x
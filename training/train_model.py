#%%

# Import modules:
import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from torch.utils.data import DataLoader
from data.data_generator import StatesDataset
from models.fcnn_model import FCNNModel
from utils.hparams import input_size, hidden_sizes, output_size, batch_size, learning_rate, num_epochs

#%%

# Load and wrap dataset in DataLoader:
dataset = StatesDataset()
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Instantiate the model and quantum circuit:
model = FCNNModel(input_size, hidden_sizes, output_size)

# Optimizer:
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Custom MSE function for complex numbers:
def complex_mse_loss(output, target):
    real_diff = output.real - target.real
    imag_diff = output.imag - target.imag
    loss = torch.mean(real_diff ** 2 + imag_diff ** 2)
    return loss

# Initialize list to store average loss per epoch
epoch_losses = []

# Training loop:
for epoch in range(num_epochs):
    model.train()
    batch_losses = []  # List to store losses for each batch
    for batch_idx, (state_batch, target_theta_batch) in enumerate(train_loader):

        # Ensure target and output are compatible with complex numbers
        state_batch = state_batch.to(torch.cfloat)  # Convert inputs to complex if not already
        target_theta_batch = target_theta_batch.to(torch.cfloat)  # Ensure targets are complex

        # Forward pass:
        predicted_theta_batch = model(state_batch)
        loss = complex_mse_loss(predicted_theta_batch, target_theta_batch)

        # Backward pass and optimization:
        optimizer.zero_grad()  # Clear existing gradients
        loss.backward()  # Compute gradient of loss w.r.t. model parameters
        optimizer.step()  # Update model parameters

        batch_losses.append(loss.item())  # Store loss for this batch

    # Calculate average loss for the epoch:
    avg_epoch_loss = sum(batch_losses) / len(batch_losses)
    epoch_losses.append(avg_epoch_loss)  # Store average loss for this epoch

    # Print average loss for the epoch:
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_epoch_loss:.4f}")

#%%

torch.save(model.state_dict(), 'models/model_state_dict.pth')
np.save("evaluation/epoch_losses.npy", np.array(epoch_losses))

#%%
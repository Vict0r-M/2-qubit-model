import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from models.fcnn_model import FCNNModel
from models.two_qubit_circuit import create_quantum_circuit

# Load the dataset:
dataset_path = 'data/wavefunctions_dataset.pt'
wavefunctions_dataset = torch.load(dataset_path)

# Wrap the dataset in a TensorDataset and DataLoader for batch processing:
dataset = TensorDataset(wavefunctions_dataset, wavefunctions_dataset)  # Dummy labels since we only have inputs
train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

# Instantiate the model and quantum circuit:
model = FCNNModel()
quantum_circuit = create_quantum_circuit()

# Loss function and optimizer:
loss_function = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop:
num_epochs = 100
for epoch in range(num_epochs):
    total_loss = 0
    for wavefunction_batch, _ in train_loader:  # Using dummy labels
        optimizer.zero_grad()
        
        batch_loss = 0
        for wavefunction in wavefunction_batch:
            # Process each wavefunction through the model to get predicted theta:
            predicted_theta = model(wavefunction.unsqueeze(0))  # Add batch dimension
            # Process each predicted_theta through the quantum circuit:
            output_wavefunction = quantum_circuit(predicted_theta.squeeze())  # Remove batch dimension if needed
            
            # Calculate loss for the current wavefunction:
            wavefunction_loss = loss_function(output_wavefunction, wavefunction.unsqueeze(0))  # Add batch dimension if needed
            batch_loss += wavefunction_loss
        
        batch_loss /= len(wavefunction_batch)  # Average loss over the batch
        total_loss += batch_loss.item()

        # Backpropagation:
        batch_loss.backward()
        optimizer.step()
    
    avg_epoch_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1}, Average Loss: {avg_epoch_loss}")
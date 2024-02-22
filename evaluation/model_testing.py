import torch
import numpy as np
from torch.utils.data import DataLoader
from data.data_generator import StatesDataset  # Adjust import path as necessary
from models.fcnn_model import FCNNModel  # Adjust import path as necessary
from utils.hparams import input_size, hidden_sizes, output_size  # Adjust import path as necessary

def load_model(model_path, input_size, hidden_sizes, output_size):
    model = FCNNModel(input_size, hidden_sizes, output_size)
    model.load_state_dict(torch.load(model_path))
    model.eval()  # Set the model to evaluation mode
    return model

def compute_accuracy(model, data_loader, tolerance_degrees=0.0001):
    tolerance_radians = np.radians(tolerance_degrees)  # Convert degrees to radians
    correct_predictions = 0
    total_predictions = 0

    with torch.no_grad():
        for state_batch, target_theta_batch in data_loader:
            predicted_theta_batch = model(state_batch).real  # Get real part of prediction
            # Ensure broadcasting works by adding extra dimensions where necessary
            pred_expanded = predicted_theta_batch.unsqueeze(2)  # Expand dims for broadcasting
            target_expanded = target_theta_batch.unsqueeze(1)  # Expand dims for broadcasting
            
            # Calculate absolute error in radians for all possible pairs
            absolute_errors = torch.abs(pred_expanded - target_expanded)
            
            # Adjust for cyclical nature of angles
            absolute_errors = torch.min(absolute_errors, 2 * np.pi - absolute_errors)

            # Find minimum error for each predicted angle across all targets
            min_errors, _ = torch.min(absolute_errors, dim=2)
            
            # A prediction is correct if all its angles have a matching target angle within tolerance
            correct_preds_per_sample = min_errors <= tolerance_radians
            correct_predictions += torch.all(correct_preds_per_sample, dim=1).sum().item()
            total_predictions += state_batch.size(0)

    accuracy = (correct_predictions / total_predictions) * 100
    return accuracy

def main():
    dataset_path = 'data/testing_amplitude_theta_dataset.npz'
    model_path = 'models/model2_state_dict.pth'

    dataset = StatesDataset(npz_file=dataset_path)
    data_loader = DataLoader(dataset, batch_size=32, shuffle=False)

    model = load_model(model_path, input_size, hidden_sizes, output_size)
    accuracy = compute_accuracy(model, data_loader, tolerance_degrees=1)
    print(f"Model Accuracy: {accuracy:.2f}%")

if __name__ == "__main__":
    main()

#%%

# Import libraries:
import numpy as np
import torch
from torch.utils.data import Dataset
from models.two_qubit_circuit import compute_state

#%%

def generate_data(samples=10000, angle_range=(0, 2*np.pi)):
    """
    Generates and saves a dataset of quantum states (in complex form) and 
    corresponding theta values.
    """
    thetas = np.random.uniform(low=angle_range[0], high=angle_range[1], size=(samples, 3))
    states = np.zeros((samples, 4), dtype=np.complex64)  # Allocate array for complex quantum states
    
    for i in range(samples):
        states[i] = compute_state(thetas[i])
    
    # Save data to file:
    np.savez('data/testing_amplitude_theta_dataset.npz', states=states, thetas=thetas)
    print(f"Generated and saved dataset with {samples} samples.")

#%%

class StatesDataset(Dataset):
    """Quantum states dataset, utilizing complex numbers."""
    def __init__(self, npz_file='data/testing_amplitude_theta_dataset.npz'):
        data = np.load(npz_file)
        self.states = data['states']
        self.thetas = data['thetas']
    
    def __len__(self):
        return len(self.thetas)
    
    def __getitem__(self, idx):
        state = torch.tensor(self.states[idx], dtype=torch.cfloat)  # Use complex tensors
        theta = torch.tensor(self.thetas[idx], dtype=torch.float)
        return state, theta

if __name__ == "__main__":
    # Generate the dataset if this script is run directly:
    generate_data()

#%%
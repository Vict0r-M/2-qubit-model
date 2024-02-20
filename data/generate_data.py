import torch

def generate_random_wavefunctions_torch(num_samples: int):
    """
    Generates random wavefunctions using PyTorch, represented by normalized tensors of 4 numbers.
    
    Parameters:
    - num_samples: int, the number of wavefunction vectors to generate.
    
    Returns:
    - A PyTorch tensor of shape (num_samples, 4), each row is a normalized wavefunction vector.
    """
    # Generate random tensors of shape (num_samples, 4)
    random_tensors = torch.rand(num_samples, 4)
    
    # Normalize each tensor to ensure they represent valid quantum states
    norms = torch.norm(random_tensors, p=2, dim=1, keepdim=True)
    normalized_wavefunctions = random_tensors / norms
    
    return normalized_wavefunctions

'''
# Example function call to create a dataset:
num_samples = 10000
wavefunctions_dataset = generate_random_wavefunctions_torch(num_samples)

# Code to save the dataset:
torch.save(wavefunctions_dataset, 'data/wavefunctions_dataset.pt')
'''
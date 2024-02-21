import numpy as np
import matplotlib.pyplot as plt

# Load the epoch losses:
epoch_losses = np.load("evaluation/epoch_losses.npy")

# Plotting:
plt.figure(figsize=(10, 6))
plt.plot(epoch_losses, label='Epoch Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss Over Epochs')
plt.legend()

plt.savefig("evaluation/epoch_loss.png")

# Display the plot:
plt.show()
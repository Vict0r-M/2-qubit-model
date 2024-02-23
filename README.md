We chose to generate a different dataset, complete with the desired states and their corresponding parameters. This way, we were able to exploit the batch processing capabilities of PyTorch. For the sake of experimentation, we chose to use linear layers this time, along with the Adam optimizer and ReLU activation function, all adapted to handling the complex amplitudes of the input state. We've modified the MSE loss function such that it takes into consideration the cyclical nature of angular measurements in the case of the circuit parameters. During training and testing, we obtained a minimum loss of 0.03 and a pseudo-accuracy of 18.5% (computed by considering a tolerated error of the angular parameters of one degree).

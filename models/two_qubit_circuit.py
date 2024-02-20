import pennylane as qml
from pennylane import numpy as np

def create_quantum_circuit(theta):
    """
    Creates a Pennylane quantum circuit that implements the minimal universal 2-qubit gate.
    """
    # Define a device with 2 qubits:
    dev = qml.device('default.qubit', wires=2)

    @qml.qnode(dev, interface='torch')
    def circuit(theta):
        # Implement the circuit:
        qml.CNOT(wires=[1, 0])
        qml.RX(theta[0], wires=1)
        qml.CNOT(wires=[1, 0])
        qml.RZ(theta[1], wires=1)
        qml.RX(theta[2], wires=1)
        qml.CNOT(wires=[1, 0])
        
        # Return the statevector:
        return qml.state()
    
    return circuit
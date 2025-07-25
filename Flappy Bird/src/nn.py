import numpy as np

class NeuralNetwork:
    """
    Simple feedforward neural network for Flappy Bird AI.

    Attributes:
        W1 (np.ndarray): Weights for input to hidden layer.
        b1 (np.ndarray): Biases for hidden layer.
        W2 (np.ndarray): Weights for hidden to output layer.
        b2 (np.ndarray): Biases for output layer.
    """
    def __init__(self, in_sz=3, hid_sz=6, out_sz=1):
        """
        Initialize the neural network with random weights and zero biases.

        Args:
            in_sz (int): Number of input neurons.
            hid_sz (int): Number of hidden neurons.
            out_sz (int): Number of output neurons.
        """
        self.W1 = np.random.randn(hid_sz, in_sz)
        self.b1 = np.zeros((hid_sz, 1))
        self.W2 = np.random.randn(out_sz, hid_sz)
        self.b2 = np.zeros((out_sz, 1))

    def forward(self, x):
        """
        Perform a forward pass through the network.

        Args:
            x (np.ndarray): Input array of shape (in_sz,).

        Returns:
            np.ndarray: Output array after sigmoid activation.
        """
        x = x.reshape(-1, 1)
        a1 = np.tanh(self.W1 @ x + self.b1)
        z2 = self.W2 @ a1 + self.b2
        return 1 / (1 + np.exp(-z2))

    def get_weights(self):
        """
        Get all weights and biases as a flat array.

        Returns:
            np.ndarray: Flattened weights and biases.
        """
        return np.concatenate([
            self.W1.flatten(),
            self.b1.flatten(),
            self.W2.flatten(),
            self.b2.flatten(),
        ])

    def set_weights(self, flat):
        """
        Set all weights and biases from a flat array.

        Args:
            flat (np.ndarray): Flattened weights and biases.
        """
        i = 0
        for mat in (self.W1, self.b1, self.W2, self.b2):
            size = mat.size
            mat[:] = flat[i:i+size].reshape(mat.shape)
            i += size

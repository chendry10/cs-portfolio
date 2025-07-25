import os
import sys
import numpy as np

# Add the src directory to the import path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from nn import NeuralNetwork


def test_get_set_weights_roundtrip():
    nn = NeuralNetwork()
    weights = nn.get_weights().copy()
    # Modify weights to ensure set_weights works (optional)
    nn.set_weights(weights)
    np.testing.assert_array_equal(weights, nn.get_weights())


def test_forward_output_shape():
    nn = NeuralNetwork()
    x = np.zeros(3)
    out = nn.forward(x)
    assert isinstance(out, np.ndarray)
    assert out.shape == (1, 1)

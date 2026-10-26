import numpy as np

from mlp.network import Network
from mlp.layers import Dense
from mlp.activations import ReLU, LeakyReLU, Sigmoid, Tanh, Softmax

ACTIVATIONS = {
    "sigmoid": Sigmoid,
    "tanh": Tanh,
    "relu": ReLU,
    "leaky_relu": LeakyReLU,
    "softmax": Softmax,
}

def build_mlp(n_in, hidden, n_out, activation="sigmoid", output_activation=None, bias=True, seed=0):
    rng = np.random.RandomState(seed)
    
    sizes = [n_in] + list(hidden) + [n_out]
    layers = []
    for i in range(len(sizes) - 1):
        layers.append(Dense(sizes[i], sizes[i + 1], rng, activation, bias))
        if i < len(sizes) - 2:
            layers.append(ACTIVATIONS[activation]())
    
    if output_activation is not None:
        layers.append(ACTIVATIONS[output_activation]())
    
    return Network(layers)

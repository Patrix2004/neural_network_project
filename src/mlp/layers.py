from abc import ABC, abstractmethod
import numpy as np

class Layer(ABC):
    
    @abstractmethod
    def forward(self, input):
        ...
    
    @abstractmethod
    def backward(self, gradient):
        ...
    
    def update(self, lr):
        pass

INIT_SCALES = {
    # He: variance 2 / fan_in, for rectifiers
    "relu": lambda i, o: np.sqrt(2 / i),
    "leaky_relu": lambda i, o: np.sqrt(2 / i),
    # Xavier (Glorot): variance 2 / (fan_in + fan_out), for symmetric / saturating functions
    "sigmoid": lambda i, o: np.sqrt(2 / (i + o)),
    "tanh": lambda i, o: np.sqrt(2 / (i + o)),
    "softmax": lambda i, o: np.sqrt(2 / (i + o)),
    "linear": lambda i, o: np.sqrt(2 / (i + o)),
}

class Dense(Layer):
    
    def __init__(self, in_features, out_features, rng, activation="linear", bias=True):
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.activation = activation
        self.bias = bias
        
        scale = INIT_SCALES[activation](in_features, out_features)
        self.weights = rng.randn(in_features, out_features) * scale
        self.biases = np.zeros(out_features)
    
    def forward(self, input):
        self.input = input
        return input @ self.weights + self.biases
    
    def backward(self, dZ):
        self.dweights = self.input.T @ dZ
        self.dbiases = np.sum(dZ, axis=0) if self.bias else np.zeros_like(self.biases)
        return dZ @ self.weights.T
    
    def update(self, lr):
        self.weights -= lr * self.dweights
        self.biases -= lr * self.dbiases
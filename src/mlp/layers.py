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

class Dense(Layer):
    
    def __init__(self, in_features, out_features, rng, bias = True):
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        
        # He initialization
        # self.weights = np.random.randn(in_features, out_features) * np.sqrt(2 / in_features)
        #Xavier
        self.weights = rng.randn(in_features, out_features) * np.sqrt(2/ (in_features + out_features))
        self.biases = np.zeros(out_features)
        self.bias = bias
    
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
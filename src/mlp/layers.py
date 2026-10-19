from abc import ABC, abstractmethod
import numpy as np

class Layer(ABC):
    
    @abstractmethod
    def forward(self, input):
        ...
    
    @abstractmethod
    def backward(self, gradient):
        ...
    
    @abstractmethod
    def update(self, lr):
        ...

class Dense(Layer):
    
    def __init__(self, in_features, out_features):
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        
        # He initialization
        # self.weights = np.random.randn(in_features, out_features) * np.sqrt(2 / in_features)
        #Xavier
        self.weights = np.random.randn(in_features, out_features) * np.sqrt(2/ (in_features + out_features))
        self.biases = np.zeros(out_features)
    
    def forward(self, input):
        self.input = input
        return input @ self.weights + self.biases
    
    def backward(self, dZ):
        self.dweights = self.input.T @ dZ
        self.dbiases = np.sum(dZ, axis=0)
        return dZ @ self.weights.T
    
    def update(self, lr):
        self.weights -= lr * self.dweights
        self.biases -= lr * self.dbiases
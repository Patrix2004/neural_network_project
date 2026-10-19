import numpy as np
from src.mlp.layers import Layer

class ReLU(Layer):
    
    def forward(self, input):
        self.mask = input > 0
        return np.maximum(0, input)
    
    def backward(self, dA):
        return dA * self.mask
    
    def update(self, lr):
        pass

class Sigmoid(Layer):
    
    def forward(self, input):
        self.value = 1 / (1 + np.exp(-1 * input))
        return self.value
    
    def backward(self, dA):
        return dA * (self.value * (1 - self.value))
    
    def update(self, lr):
        pass
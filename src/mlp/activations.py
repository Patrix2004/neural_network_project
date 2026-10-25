import numpy as np
from src.mlp.layers import Layer

class ReLU(Layer):
    
    def forward(self, input):
        self.mask = input > 0
        return np.maximum(0, input)
    
    def backward(self, dA):
        return dA * self.mask

class LeakyReLU(Layer):
    
    def __init__(self, a=0.01):
        super().__init__()
        self.a = a
    
    def forward(self, input):
        self.mask = input > 0
        return np.where(self.mask, input, input * self.a)
    
    def backward(self, dA):
        return dA * np.where(self.mask, 1, self.a)

class Sigmoid(Layer):
    
    def forward(self, input):
        self.value = 1 / (1 + np.exp(-1 * input))
        return self.value
    
    def backward(self, dA):
        return dA * (self.value * (1 - self.value))
    

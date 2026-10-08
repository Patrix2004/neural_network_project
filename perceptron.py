from abc import ABC, abstractmethod
import numpy as np
import pandas as pd

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

class Loss(ABC):
    
    @abstractmethod
    def forward(self, target, preficted):
        ...
    
    @abstractmethod
    def backward(self, target, predicted):
        ...

class MSE(Loss):
    
    @staticmethod
    def forward(target, predicted):
        return np.sum(np.square(predicted - target)) / (target.shape[0] * target.shape[1])
    
    @staticmethod
    def backward(target, predicted):
        return 2 * (predicted - target) / (target.shape[0] * target.shape[1])


class Dense(Layer):
    
    def __init__(self, in_features, out_features):
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        
        # He initialization
        self.weights = np.random.randn(in_features, out_features) * np.sqrt(2 / in_features)
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

class ReLU(Layer):
    
    def forward(self, input):
        self.mask = input > 0
        return np.maximum(0, input)
    
    def backward(self, dA):
        return dA * self.mask
    
    def update(self, lr):
        pass

class Network():
    
    def __init__(self, layers):
        self.layers = layers
    
    def forward(self, X):
        for layer in self.layers:
            X = layer.forward(X)
        return X

    def backward(self, gradient):
        for layer in reversed(self.layers):
            gradient = layer.backward(gradient)

    def update(self, lr):
        for layer in self.layers:
            layer.update(lr)

per = Network(
    [
        Dense(2, 10),
        ReLU(),
        Dense(10, 2),
        ReLU(),
    ]
)

df = pd.read_csv("/home/SOS/Downloads/projekt1/classification/data.simple.train.100.csv")

n = 20
epochs = 5
list_df = [df[i:i + n] for i in range(0, df.shape[0], n)]

for epoch in epochs:
    for batch in list_df:
        target = df[-1]
        
        pred = per.forward(df[0:2].to_numpy())

        loss = MSE.forward(target, pred)

        gradient = MSE.backward(target, pred)

        per.backward(gradient)

        per.update(0.01)

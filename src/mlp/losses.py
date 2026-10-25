from abc import ABC, abstractmethod
import numpy as np

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

class MAE(Loss):
    
    @staticmethod
    def forward(target, predicted):
        return np.sum(np.abs(predicted - target)) / (target.shape[0] * target.shape[1])
    
    @staticmethod
    def backward(target, predicted):
        return np.sign(predicted - target) / (target.shape[0] * target.shape[1])

class CrossEntropy(Loss):
    eps = 1e-12
    
    @staticmethod
    def forward(target, predicted):
        return -np.sum(target * np.log(predicted + CrossEntropy.eps)) / target.shape[0]
    
    @staticmethod
    def backward(target, predicted):
        return -target / (predicted + CrossEntropy.eps) / target.shape[0]
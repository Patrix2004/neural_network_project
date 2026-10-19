import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.mlp.network import Network
from src.mlp.layers import Dense
from src.mlp.losses import MSE
from src.mlp.activations import *
from src.utils.preprocessing import *



    



per = Network(
    [
        Dense(1, 128),
        Sigmoid(),
        Dense(128, 128),
        Sigmoid(),
        Dense(128, 1),
    ]
)

df = pd.read_csv("/home/SOS/Downloads/projekt1/regression/data.cube.train.10000.csv")

n = 20
epochs = 500
list_df = [df[i:i + n] for i in range(0, df.shape[0], n)]

for epoch in range(epochs):
    for batch in list_df:
        target = batch.iloc[:, -1:].to_numpy()
        
        pred = per.forward(batch.iloc[:, 0].to_numpy().reshape((n, 1)))

        loss = MSE.forward(target, pred)

        gradient = MSE.backward(target, pred)

        per.backward(gradient)

        per.update(0.001)

df = pd.read_csv("/home/SOS/Downloads/projekt1/regression/data.cube.test.1000.csv")

target = df.iloc[:, -1:].to_numpy()

pred = per.forward(np.array(df.iloc[:, 0]).reshape((-1,1)))

loss = MSE.forward(target, pred)

print(df.iloc[:, [0, -1]].describe())

print(pred - target)

x = df.iloc[:, 0]

plt.scatter(x, target, label = "Target")
plt.scatter(x, pred, label = "Prediction")

plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.show()

print(per.layers[0].weights)
print(per.layers[0].biases)

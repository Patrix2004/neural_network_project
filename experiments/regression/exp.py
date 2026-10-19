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
        Dense(3, 32),
        Sigmoid(),
        Dense(32, 16),
        Sigmoid(),
        Dense(16, 1),
    ]
)

df = pd.read_csv("/home/SOS/Downloads/projekt1/regression/data.cube.train.10000.csv")

for i in (2, 3):
    df.insert(i - 1, f"x{i}", df["x"] ** i)
    
X = df.iloc[:, :-1].to_numpy(dtype=np.float64)
y = df.iloc[:, -1:].to_numpy(dtype=np.float64)

X_mean = np.mean(X, axis=0)
X_std = np.std(X, axis=0)

y_mean = np.mean(y, axis=0)
y_std = np.std(y, axis=0)

print((X_mean, X_std, y_mean, y_std))

X = (X - X_mean) / X_std
y = (y - y_mean) / y_std

lower = X.iloc[:, 0].min()
upper = X.iloc[:, 0].max()

    
n = 50
max_epochs = 10_000
lr = 0.01

patience = 100
min_delta = 1e-6

loss_history = []
best_loss = float("inf")
epochs_without_improvement = 0

# list_df = [df[i:i + n] for i in range(0, df.shape[0], n)]

for epoch in range(max_epochs):
    epoch_loss = 0.0
    # print(f"Epoch {epoch}")
    
    indices = np.random.permutation(len(X))
    X_shuffled = X[indices]
    y_shuffled = y[indices]
    
    for start in range(0, len(X), n):
        X_batch = X_shuffled[start:start + n]
        y_batch = y_shuffled[start:start + n]
        
        pred = per.forward(X_batch)
        loss = MSE.forward(y_batch, pred)

        gradient = MSE.backward(y_batch, pred)
        per.backward(gradient)
        
        per.update(lr)
        
        epoch_loss += loss * len(X_batch)
    
    epoch_loss = epoch_loss / len(X)
    
    loss_history.append(epoch_loss)
    
    if epoch_loss < best_loss - min_delta:
        best_loss = epoch_loss
        epochs_without_improvement = 0
        # TODO save model
    else:
        epochs_without_improvement += 1
    
    if epoch % 10 == 0:
        print(f"Epoch {epoch}: loss = {epoch_loss:.6f}")
    
    if epochs_without_improvement >= patience:
        print(f"Early stopping at epoch {epoch}")
        break

plt.plot(loss_history)
plt.show()


df = pd.read_csv("/home/SOS/Downloads/projekt1/regression/data.cube.test.1000.csv")

for i in (2, 3):
    df.insert(i - 1, f"x{i}", df["x"] ** i)

X_test = df.iloc[:, :-1].to_numpy(dtype=np.float64)
y_test = df.iloc[:, -1:].to_numpy(dtype=np.float64)

X_test = (X_test - X_mean) / X_std
y_test = (y_test - y_mean) / y_std

X_test.iloc[:, 0] = X_test.iloc[:, 0].clip(lower=lower, upper=upper)

pred = per.forward(X_test)

loss = MSE.forward(y_test, pred)

plt.scatter(X_test[:, 0], y_test, label = "Target")
plt.scatter(X_test[:, 0], pred, label = "Prediction")

# plt.xlim(-3, 3)
# plt.ylim(-250, 250)
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.show()

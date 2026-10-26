import numpy as np

from mlp.losses import MSE, CrossEntropy

def r2_score(target, predicted):
    return 1 - np.sum(np.square(target - predicted)) / np.sum(np.square(target - np.mean(target, axis=0)))

def accuracy(target, predicted):
    return np.mean(np.argmax(target, axis=1) == np.argmax(predicted, axis=1))

def evaluate(net, ds, split):
    # metrics in original units (regression) / on probabilities (classification)
    X = getattr(ds, f"X_{split}")
    Y = getattr(ds, f"Y_{split}")
    y = getattr(ds, f"y_{split}")
    pred = net.forward(X)
    
    if ds.task == "regression":
        pred = ds.y_scaler.inverse(pred)
        return {
            "mse": float(np.mean(np.square(pred - y))),
            "mae": float(np.mean(np.abs(pred - y))),
            "r2": float(r2_score(y, pred)),
        }
    
    return {
        "acc": float(np.mean(np.argmax(pred, axis=1) == y)),
        "ce": float(CrossEntropy.forward(Y, pred)),
        "brier": float(MSE.forward(Y, pred)),
    }

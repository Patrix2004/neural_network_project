from pathlib import Path

import numpy as np
import pandas as pd

class Standardizer:
    
    def fit(self, A):
        self.mean = np.mean(A, axis=0)
        self.std = np.std(A, axis=0)
        self.std[self.std == 0] = 1
        return self
    
    def transform(self, A):
        return (A - self.mean) / self.std
    
    def inverse(self, A):
        return A * self.std + self.mean

class Dataset:
    # X_*, Y_* - network space (standardized inputs, standardized targets / one-hot)
    # y_*      - original targets (regression values / class indices)
    
    def __init__(self, task, train, test, val_fraction=0.15, split_seed=0, name=""):
        self.task = task
        self.name = name
        X_train, y_train = train
        X_test, y_test = test
        
        if task == "classification":
            self.classes = np.unique(np.concatenate([y_train, y_test]))
            y_train = np.searchsorted(self.classes, y_train)
            y_test = np.searchsorted(self.classes, y_test)
        else:
            y_train, y_test = y_train.reshape(-1, 1), y_test.reshape(-1, 1)
        
        # validation split (used for early stopping), test set stays untouched
        indices = np.random.RandomState(split_seed).permutation(len(X_train))
        n_val = int(len(X_train) * val_fraction)
        val, tr = indices[:n_val], indices[n_val:]
        
        splits = {
            "train": (X_train[tr], y_train[tr]),
            "val": (X_train[val], y_train[val]),
            "test": (X_test, y_test),
        }
        
        self.x_scaler = Standardizer().fit(splits["train"][0])
        if task == "regression":
            self.y_scaler = Standardizer().fit(splits["train"][1])
        
        for split, (X, y) in splits.items():
            setattr(self, f"X_{split}", self.x_scaler.transform(X))
            setattr(self, f"y_{split}", y)
            setattr(self, f"Y_{split}", self.encode(y))
        
        self.n_in = self.X_train.shape[1]
        self.n_out = 1 if task == "regression" else len(self.classes)
    
    def encode(self, y):
        if self.task == "regression":
            return self.y_scaler.transform(y)
        return np.eye(len(self.classes))[y]
    
    @property
    def has_val(self):
        return len(self.X_val) > 0

def load_csv(path):
    df = pd.read_csv(path)
    return df.iloc[:, :-1].to_numpy(dtype=np.float64), df.iloc[:, -1].to_numpy()

PROBLEMS = {
    "regression": ["cube", "activation"],
    "classification": ["three_gauss", "simple"],
}
SIZES = [100, 500, 1000, 10000]
TEST_SIZES = {100: 100, 500: 500, 1000: 1000, 10000: 10000}  # <- dopasuj do swoich plików

def load_problem(data_dir, task, name, size, **kwargs):
    base = Path(data_dir) / task
    train = load_csv(base / f"data.{name}.train.{size}.csv")
    test = load_csv(base / f"data.{name}.test.{TEST_SIZES[size]}.csv")
    return Dataset(task, train, test, name=f"{name}.{size}", **kwargs)
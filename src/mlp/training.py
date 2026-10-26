import numpy as np

from mlp.layers import Dense
from mlp.metrics import r2_score, accuracy

class History:
    
    def __init__(self, layer_shapes):
        self.layer_shapes = layer_shapes
        self.train_loss = []
        self.val_loss = []
        self.train_metric = []
        self.val_metric = []
        self.grad_norms = []
        self.snapshot_epochs = []
        self.snapshots = []
        self.best_epoch = 0
        self.stopped_epoch = 0

def dense_layers(net):
    return [layer for layer in net.layers if isinstance(layer, Dense)]

def get_params(net):
    return [(layer.weights.copy(), layer.biases.copy()) for layer in dense_layers(net)]

def train(net, loss, X, Y, lr, batch_size, max_epochs, seed=0, X_val=None, Y_val=None,
          task="regression", patience=100, min_delta=1e-6, n_snapshots=200, verbose=0):
    # SGD with mini-batches; early stopping on the validation loss (or train loss
    # when there is no validation set); the best weights are restored at the end
    rng = np.random.RandomState(seed)
    metric = r2_score if task == "regression" else accuracy
    layers = dense_layers(net)
    has_val = X_val is not None and len(X_val) > 0
    history = History([layer.weights.shape for layer in layers])
    
    snapshot_every = max(1, max_epochs // n_snapshots)
    history.snapshot_epochs.append(0)
    history.snapshots.append(get_params(net))
    
    best_loss = float("inf")
    best_params = get_params(net)
    epochs_without_improvement = 0
    
    with np.errstate(over="ignore"):
        for epoch in range(max_epochs):
            epoch_loss = 0.0
            norms = np.zeros(len(layers))
            n_batches = 0
            
            indices = rng.permutation(len(X))
            for start in range(0, len(X), batch_size):
                idx = indices[start:start + batch_size]
                X_batch, Y_batch = X[idx], Y[idx]
                
                pred = net.forward(X_batch)
                epoch_loss += loss.forward(Y_batch, pred) * len(idx)
                
                net.backward(loss.backward(Y_batch, pred))
                norms += [np.linalg.norm(layer.dweights) for layer in layers]
                n_batches += 1
                
                net.update(lr)
            
            history.train_loss.append(epoch_loss / len(X))
            history.grad_norms.append(norms / n_batches)
            history.train_metric.append(metric(Y, net.forward(X)))
            
            if has_val:
                pred_val = net.forward(X_val)
                history.val_loss.append(loss.forward(Y_val, pred_val))
                history.val_metric.append(metric(Y_val, pred_val))
            
            if (epoch + 1) % snapshot_every == 0:
                history.snapshot_epochs.append(epoch + 1)
                history.snapshots.append(get_params(net))
            
            monitored = history.val_loss[-1] if has_val else history.train_loss[-1]
            if monitored < best_loss - min_delta:
                best_loss = monitored
                best_params = get_params(net)
                history.best_epoch = epoch
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
            
            if verbose and epoch % verbose == 0:
                print(f"Epoch {epoch}: loss = {history.train_loss[-1]:.6f}, monitored = {monitored:.6f}")
            
            if epochs_without_improvement >= patience:
                break
    
    history.stopped_epoch = epoch
    if history.snapshot_epochs[-1] != epoch + 1:
        history.snapshot_epochs.append(epoch + 1)
        history.snapshots.append(get_params(net))
    
    for layer, (weights, biases) in zip(dense_layers(net), best_params):
        layer.weights = weights.copy()
        layer.biases = biases.copy()
    
    return history

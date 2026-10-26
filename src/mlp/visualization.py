from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from mlp.training import dense_layers

PALETTE = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3", "#937860", "#DA8BC3", "#8C8C8C"]

def set_style():
    plt.rcParams.update({
        "figure.dpi": 110,
        "savefig.dpi": 150,
        "savefig.bbox": "tight",
        "axes.grid": True,
        "grid.alpha": 0.25,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.prop_cycle": plt.cycler(color=PALETTE),
        "legend.frameon": False,
    })

def save_or_show(fig, path=None, show=False):
    if path is not None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path)
    if show:
        plt.show()
    plt.close(fig)

def predict(net, X):
    with np.errstate(over="ignore"):
        return net.forward(X)

# regression R -> R

def plot_regression_fit(net, ds, title=None, ax=None, show_residuals=True, legend=True):
    from mlp.metrics import evaluate
    
    x_all = np.concatenate([ds.x_scaler.inverse(ds.X_train), ds.x_scaler.inverse(ds.X_test)])
    lo, hi = x_all.min(), x_all.max()
    pad = 0.05 * (hi - lo)
    grid = np.linspace(lo - pad, hi + pad, 500).reshape(-1, 1)
    curve = ds.y_scaler.inverse(predict(net, ds.x_scaler.transform(grid)))
    
    x_train = ds.x_scaler.inverse(ds.X_train)
    x_test = ds.x_scaler.inverse(ds.X_test)
    pred_test = ds.y_scaler.inverse(predict(net, ds.X_test))
    scores = evaluate(net, ds, "test")
    
    if ax is None:
        if show_residuals:
            fig, (ax, ax_res) = plt.subplots(1, 2, figsize=(12, 4.2), gridspec_kw={"width_ratios": [2, 1]})
        else:
            fig, ax = plt.subplots(figsize=(7, 4.2))
    else:
        fig, ax_res = ax.figure, None
    
    ax.scatter(x_train, ds.y_train, s=8, alpha=0.35, color="#8C8C8C", label="zbiór uczący")
    ax.scatter(x_test, ds.y_test, s=12, alpha=0.7, color=PALETTE[1], label="zbiór testowy")
    ax.plot(grid, curve, color=PALETTE[0], lw=2.2, label="sieć")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(title or f"Regresja: MSE={scores['mse']:.3g}, MAE={scores['mae']:.3g}, R²={scores['r2']:.4f}")
    if legend:
        ax.legend(loc="best")
    
    if show_residuals and ax_res is not None:
        ax_res.scatter(x_test, (ds.y_test - pred_test), s=10, alpha=0.6, color=PALETTE[3])
        ax_res.axhline(0, color="black", lw=1)
        ax_res.set_xlabel("x")
        ax_res.set_ylabel("y - ŷ")
        ax_res.set_title("Reszty na zbiorze testowym")
    
    return fig

# classification R^2 -> classes

def plot_decision_regions(net, ds, ax, split="test", title=None, resolution=300):
    X = ds.x_scaler.inverse(getattr(ds, f"X_{split}"))
    y = getattr(ds, f"y_{split}")
    X_all = np.concatenate([ds.x_scaler.inverse(ds.X_train), ds.x_scaler.inverse(ds.X_test)])
    lo, hi = X_all.min(axis=0), X_all.max(axis=0)
    pad = 0.08 * (hi - lo)
    xx, yy = np.meshgrid(np.linspace(lo[0] - pad[0], hi[0] + pad[0], resolution),
                         np.linspace(lo[1] - pad[1], hi[1] + pad[1], resolution))
    grid = np.c_[xx.ravel(), yy.ravel()]
    probs = predict(net, ds.x_scaler.transform(grid))
    regions = np.argmax(probs, axis=1).reshape(xx.shape)
    
    colors = PALETTE[:len(ds.classes)]
    cmap = ListedColormap(colors)
    ax.contourf(xx, yy, regions, levels=np.arange(-0.5, len(ds.classes), 1), cmap=cmap, alpha=0.22)
    ax.contour(xx, yy, regions, levels=np.arange(0.5, len(ds.classes) - 1, 1), colors="black", linewidths=0.8)
    
    pred = np.argmax(predict(net, getattr(ds, f"X_{split}")), axis=1)
    for k, cls in enumerate(ds.classes):
        mask = y == k
        ax.scatter(X[mask, 0], X[mask, 1], s=14, color=colors[k], edgecolor="white", linewidth=0.3, label=f"klasa {cls}")
    wrong = pred != y
    ax.scatter(X[wrong, 0], X[wrong, 1], s=45, facecolors="none", edgecolors="black", linewidth=1, label="błędne")
    
    acc = 1 - wrong.mean()
    ax.set_title(title or f"{'Zbiór testowy' if split == 'test' else 'Zbiór uczący'}: dokładność={acc:.3f}")
    ax.set_xlabel("x₁")
    ax.set_ylabel("x₂")
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.grid(False)

def plot_classification(net, ds, title=None):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    plot_decision_regions(net, ds, axes[0], "train")
    plot_decision_regions(net, ds, axes[1], "test")
    axes[1].legend(loc="upper right", fontsize=8, framealpha=0.8, frameon=True)
    if title:
        fig.suptitle(title, fontweight="bold")
    return fig

# training diagnostics

def plot_training(history, task, loss_name="strata", title=None):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    epochs = np.arange(1, len(history.train_loss) + 1)
    
    ax = axes[0]
    ax.plot(epochs, history.train_loss, label="uczący", lw=1.8)
    if history.val_loss:
        ax.plot(epochs, history.val_loss, label="walidacyjny", lw=1.8)
    ax.axvline(history.best_epoch + 1, color="black", ls="--", lw=1, label=f"najlepsza epoka ({history.best_epoch + 1})")
    ax.set_yscale("log")
    ax.set_xlabel("epoka")
    ax.set_ylabel(loss_name)
    ax.set_title("Błąd w kolejnych epokach")
    ax.legend()
    
    ax = axes[1]
    metric_name = "R²" if task == "regression" else "dokładność"
    ax.plot(epochs, history.train_metric, label="uczący", lw=1.8)
    if history.val_metric:
        ax.plot(epochs, history.val_metric, label="walidacyjny", lw=1.8)
    ax.set_xlabel("epoka")
    ax.set_ylabel(metric_name)
    ax.set_title(f"{metric_name} w kolejnych epokach")
    ax.legend()
    
    ax = axes[2]
    grads = np.array(history.grad_norms)
    for i in range(grads.shape[1]):
        ax.plot(epochs, grads[:, i], label=f"warstwa {i + 1} {history.layer_shapes[i][0]}→{history.layer_shapes[i][1]}", lw=1.5)
    ax.set_yscale("log")
    ax.set_xlabel("epoka")
    ax.set_ylabel("‖∂L/∂W‖ (średnia z batchy)")
    ax.set_title("Błąd propagowany wstecz (norma gradientu)")
    ax.legend(fontsize=8)
    
    if title:
        fig.suptitle(title, fontweight="bold", y=1.03)
    fig.tight_layout()
    return fig

def plot_weights(history, max_lines=60, seed=0, title=None):
    # trajectories of single weights and biases; one panel per Dense layer
    n_layers = len(history.layer_shapes)
    ncols = min(n_layers, 3)
    nrows = int(np.ceil(n_layers / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.2 * ncols, 3.6 * nrows), squeeze=False)
    epochs = np.array(history.snapshot_epochs)
    rng = np.random.RandomState(seed)
    
    for i in range(n_layers):
        ax = axes[i // ncols, i % ncols]
        W = np.array([snap[i][0].ravel() for snap in history.snapshots])
        B = np.array([snap[i][1] for snap in history.snapshots])
        
        cols = np.arange(W.shape[1])
        if len(cols) > max_lines:
            cols = rng.choice(cols, max_lines, replace=False)
        ax.plot(epochs, W[:, cols], color=PALETTE[0], alpha=0.35, lw=0.9)
        if np.any(B != 0):
            ax.plot(epochs, B[:, :max_lines], color=PALETTE[1], alpha=0.5, lw=0.9, ls="--")
        ax.plot(epochs, np.mean(np.abs(W), axis=1), color="black", lw=2)
        ax.set_title(f"Warstwa {i + 1}: {history.layer_shapes[i][0]}→{history.layer_shapes[i][1]}")
        ax.set_xlabel("epoka")
        ax.set_ylabel("wartość wagi")
    
    handles = [plt.Line2D([], [], color=PALETTE[0], alpha=0.6, label="wagi"),
               plt.Line2D([], [], color=PALETTE[1], ls="--", label="biasy"),
               plt.Line2D([], [], color="black", lw=2, label="średnie |w|")]
    axes[0, 0].legend(handles=handles, loc="best", fontsize=8)
    for j in range(n_layers, nrows * ncols):
        axes[j // ncols, j % ncols].axis("off")
    
    fig.suptitle(title or "Ewolucja wag", fontweight="bold")
    fig.tight_layout()
    return fig

def plot_weight_heatmaps(history, title=None):
    # weight matrices before and after training
    n_layers = len(history.layer_shapes)
    fig, axes = plt.subplots(2, n_layers, figsize=(4.2 * n_layers, 5.2), squeeze=False)
    
    for i in range(n_layers):
        first, last = history.snapshots[0][i][0], history.snapshots[-1][i][0]
        vmax = max(np.abs(first).max(), np.abs(last).max())
        for row, (W, name) in enumerate([(first, "przed uczeniem"), (last, "po uczeniu")]):
            ax = axes[row, i]
            im = ax.imshow(W, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
            ax.set_title(f"Warstwa {i + 1} ({name})", fontsize=9)
            ax.set_xlabel("neuron wyjściowy")
            ax.set_ylabel("wejście")
            ax.grid(False)
        fig.colorbar(im, ax=axes[:, i], shrink=0.8, pad=0.02)
    
    fig.suptitle(title or "Macierze wag", fontweight="bold")
    return fig

def plot_run(net, ds, history, loss_name, out_dir, prefix="", show=False):
    # all visualizations required for a single run
    out_dir = Path(out_dir)
    fit = plot_regression_fit(net, ds) if ds.task == "regression" else plot_classification(net, ds)
    save_or_show(fit, out_dir / f"{prefix}fit.png", show)
    save_or_show(plot_training(history, ds.task, loss_name), out_dir / f"{prefix}training.png", show)
    save_or_show(plot_weights(history), out_dir / f"{prefix}weights.png", show)
    save_or_show(plot_weight_heatmaps(history), out_dir / f"{prefix}weights_heatmap.png", show)

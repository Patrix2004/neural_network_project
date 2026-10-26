import numpy as np
import matplotlib.pyplot as plt

from mlp.visualization import PALETTE, plot_regression_fit, plot_decision_regions

def group_runs(runs):
    groups = {}
    for run in runs:
        groups.setdefault(run.cfg.label, []).append(run)
    return groups

def plot_curves(runs, key, ylabel, logy=False, title=None):
    # mean +- std over seeds; runs stopped early are padded with NaN
    groups = group_runs(runs)
    fig, ax = plt.subplots(figsize=(8, 4.8))
    
    for i, (label, group) in enumerate(groups.items()):
        n = max(len(getattr(run.history, key)) for run in group)
        curves = np.full((len(group), n), np.nan)
        for j, run in enumerate(group):
            values = getattr(run.history, key)
            curves[j, :len(values)] = values
        mean, std = np.nanmean(curves, axis=0), np.nanstd(curves, axis=0)
        x = np.arange(1, n + 1)
        color = PALETTE[i % len(PALETTE)]
        ax.plot(x, mean, label=label, color=color, lw=2)
        ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.15)
    
    if logy:
        ax.set_yscale("log")
    ax.set_xlabel("epoka")
    ax.set_ylabel(ylabel)
    ax.set_title(title or ylabel)
    ax.legend(fontsize=8)
    return fig

def plot_metric_comparison(df, metrics, order, title=None):
    # metrics: list of (column, label, log scale); large marker = mean, whiskers = std, small dots = single seeds
    fig, axes = plt.subplots(1, len(metrics), figsize=(5.2 * len(metrics) + 1, 4.4), squeeze=False)
    
    for ax, (col, label, logy) in zip(axes[0], metrics):
        for i, name in enumerate(order):
            values = df.loc[df["label"] == name, col].to_numpy()
            color = PALETTE[i % len(PALETTE)]
            ax.scatter(np.full(len(values), i) + np.linspace(-0.12, 0.12, len(values)), values,
                       s=14, color=color, alpha=0.5, zorder=2)
            ax.errorbar(i, values.mean(), yerr=values.std(), fmt="D", color=color, ms=8, capsize=5,
                        mec="black", zorder=3)
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels(order, rotation=35, ha="right", fontsize=8)
        ax.set_ylabel(label)
        ax.set_title(label)
        if logy:
            ax.set_yscale("log")
        ax.grid(axis="x", visible=False)
    
    if title:
        fig.suptitle(title, fontweight="bold")
    fig.tight_layout()
    return fig

def plot_depth_width(df, metrics, title=None):
    # x = number of hidden layers, one point per architecture (mean +- std over seeds)
    fig, axes = plt.subplots(1, len(metrics), figsize=(5.8 * len(metrics), 4.6), squeeze=False)
    summary = df.groupby(["label", "depth", "arch"], sort=False)
    per_depth = df.groupby("depth")["label"].nunique()
    seen = {}
    
    for ax, (col, label, logy) in zip(axes[0], metrics):
        seen.clear()
        for (name, depth, arch), group in summary:
            values = group[col].to_numpy()
            k = seen.get(depth, 0)
            seen[depth] = k + 1
            x = depth + (k - (per_depth[depth] - 1) / 2) * 0.16
            ax.errorbar(x, values.mean(), yerr=values.std(), fmt="o", color=PALETTE[depth % len(PALETTE)],
                        capsize=3, ms=7)
            ax.annotate(arch, (x, values.mean()), textcoords="offset points", xytext=(6, 4), fontsize=7)
        best = df.groupby("depth")[col].mean()
        ax.plot(best.index, best.values, color="black", lw=1, ls=":", label="średnia dla głębokości")
        ax.set_xticks(sorted(df["depth"].unique()))
        ax.set_xlabel("liczba warstw ukrytych")
        ax.set_ylabel(label)
        ax.set_title(label)
        if logy:
            ax.set_yscale("log")
        ax.legend(fontsize=8)
    
    if title:
        fig.suptitle(title, fontweight="bold")
    fig.tight_layout()
    return fig

def plot_gallery(runs, ds, ncols=3, title=None):
    # fit / decision regions of the first seed of every configuration
    firsts = [group[0] for group in group_runs(runs).values()]
    nrows = int(np.ceil(len(firsts) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.8 * nrows), squeeze=False)
    
    for k, run in enumerate(firsts):
        ax = axes[k // ncols, k % ncols]
        scores = run.test
        if ds.task == "regression":
            sub = f"{run.cfg.label}\nMSE={scores['mse']:.3g}, R²={scores['r2']:.3f}"
            plot_regression_fit(run.net, ds, title=sub, ax=ax, show_residuals=False, legend=(k == 0))
        else:
            sub = f"{run.cfg.label}\nacc={scores['acc']:.3f}, CE={scores['ce']:.3f}"
            plot_decision_regions(run.net, ds, ax, "test", title=sub, resolution=200)
    
    for k in range(len(firsts), nrows * ncols):
        axes[k // ncols, k % ncols].axis("off")
    
    if title:
        fig.suptitle(title, fontweight="bold")
    fig.tight_layout()
    return fig

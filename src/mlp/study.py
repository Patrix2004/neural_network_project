import argparse
import time
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd

from mlp.losses import MSE, MAE, CrossEntropy
from mlp.builder import build_mlp
from mlp.training import train
from mlp.metrics import evaluate
from mlp import data, visualization as viz, compare

LOSSES = {"mse": MSE, "mae": MAE, "ce": CrossEntropy}

REGRESSION_METRICS = [("mse", "MSE (test)", True), ("mae", "MAE (test)", True), ("r2", "R² (test)", False)]
CLASSIFICATION_METRICS = [("acc", "accuracy (test)", False), ("ce", "cross-entropy (test)", True), ("brier", "MSE na one-hot (test)", True)]

@dataclass
class Config:
    label: str
    hidden: list
    activation: str = "tanh"
    loss: str = "mse"
    bias: bool = True
    lr: float = 0.05
    batch_size: int = 50
    max_epochs: int = 500
    patience: int = 50

@dataclass
class Run:
    cfg: Config
    seed: int
    net: object
    history: object
    train: dict
    test: dict
    seconds: float

def run_config(cfg, ds, seed):
    output = None if ds.task == "regression" else "softmax"
    net = build_mlp(ds.n_in, cfg.hidden, ds.n_out, cfg.activation, output, cfg.bias, seed)
    
    start = time.time()
    history = train(net, LOSSES[cfg.loss], ds.X_train, ds.Y_train, cfg.lr, cfg.batch_size, cfg.max_epochs,
                    seed=seed, X_val=ds.X_val, Y_val=ds.Y_val, task=ds.task, patience=cfg.patience)
    seconds = time.time() - start
    
    return Run(cfg, seed, net, history, evaluate(net, ds, "train"), evaluate(net, ds, "test"), seconds)

def arch_name(hidden):
    return "-".join(str(h) for h in hidden) if hidden else "brak"

def runs_to_frame(runs):
    rows = []
    for run in runs:
        rows.append({
            "label": run.cfg.label, "seed": run.seed, "arch": arch_name(run.cfg.hidden),
            "depth": len(run.cfg.hidden), "activation": run.cfg.activation, "loss": run.cfg.loss,
            "bias": run.cfg.bias, "lr": run.cfg.lr, "epochs": run.history.stopped_epoch + 1,
            "best_epoch": run.history.best_epoch + 1, "seconds": run.seconds,
            **{f"train_{k}": v for k, v in run.train.items()}, **run.test,
        })
    return pd.DataFrame(rows)

def summarize(df, metrics):
    cols = [m[0] for m in metrics] + ["epochs"]
    summary = df.groupby("label", sort=False)[cols].agg(["mean", "std"])
    summary.columns = [f"{a}_{b}" for a, b in summary.columns]
    return summary

def run_study(name, configs, ds, args, out_dir, metrics, loss_curves=True, depth_plot=False):
    out_dir = Path(out_dir) / name
    print(f"\n=== {ds.name} / {name}: {len(configs)} configs x {len(args.seeds)} seeds ===")

    runs = []
    for cfg in configs:
        for seed in args.seeds:
            runs.append(run_config(cfg, ds, seed))
        scores = runs_to_frame(runs[-len(args.seeds):])[[m[0] for m in metrics]].mean()
        print(f"  {cfg.label:<18}" + "  ".join(f"{k}={v:.4g}" for k, v in scores.items()))
    
    df = runs_to_frame(runs)
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "runs.csv", index=False)
    summary = summarize(df, metrics)
    summary.to_csv(out_dir / "summary.csv")
    
    order = list(dict.fromkeys(df["label"]))
    metric_name = "R²" if ds.task == "regression" else "accuracy"
    title = f"{ds.name}: {name}"
    
    viz.save_or_show(compare.plot_metric_comparison(df, metrics, order, title), out_dir / "metrics.png", args.show)
    viz.save_or_show(compare.plot_curves(runs, "val_metric", f"{metric_name} (walidacja)", title=f"{title}: {metric_name} na zbiorze walidacyjnym"),
                     out_dir / "curves_metric.png", args.show)
    if loss_curves:
        viz.save_or_show(compare.plot_curves(runs, "val_loss", "błąd walidacyjny", logy=True, title=f"{title}: błąd walidacyjny"),
                         out_dir / "curves_loss.png", args.show)
    if depth_plot:
        viz.save_or_show(compare.plot_depth_width(df, metrics[:2], title), out_dir / "depth_width.png", args.show)
    viz.save_or_show(compare.plot_gallery(runs, ds, title=title), out_dir / "gallery.png", args.show)
    return runs, df

# studies from the assignment

def study_activations(ds, base, args, out_dir, metrics):
    configs = [replace(base, label=a, activation=a) for a in ("sigmoid", "tanh", "relu", "leaky_relu")]
    return run_study("activation", configs, ds, args, out_dir, metrics)

ARCHITECTURES = [
    [],
    [8], [32], [128],
    [32, 32], [64, 16],
    [32, 32, 32], [64, 32, 16],
    [32] * 4, [64, 32, 16, 8],
]

def study_architectures(ds, base, args, out_dir, metrics):
    configs = [replace(base, label=f"{len(h)} | {arch_name(h)}", hidden=h) for h in ARCHITECTURES]
    return run_study("architecture", configs, ds, args, out_dir, metrics, depth_plot=True)

def study_losses(ds, base, args, out_dir, metrics):
    names = ("mse", "mae") if ds.task == "regression" else ("ce", "mse")
    configs = [replace(base, label=n, loss=n) for n in names]
    # different losses are not comparable by their value, so only metrics and R2/accuracy curves
    return run_study("loss", configs, ds, args, out_dir, metrics, loss_curves=False)

def study_bias(ds, base, args, out_dir, metrics):
    configs = [replace(base, label="z biasami", bias=True), replace(base, label="bez biasów", bias=False)]
    return run_study("bias", configs, ds, args, out_dir, metrics)

STUDIES = {"activation": study_activations, "architecture": study_architectures,
           "loss": study_losses, "bias": study_bias}

def baseline(ds, base, args, out_dir):
    # one run with all required visualizations: fit, error over epochs, weights evolution
    run = run_config(base, ds, seed=args.seeds[0])
    print(f"\n=== {ds.name} / baseline ({base.activation}, {arch_name(base.hidden)}): test {run.test} ===")
    viz.plot_run(run.net, ds, run.history, base.loss.upper(), Path(out_dir) / "baseline", show=args.show)

def get_datasets(args, task):
    for name in args.problems or data.PROBLEMS[task]:
        for size in args.sizes:
            yield data.load_problem(args.data_dir, task, name, size)

def main(task):
    parser = argparse.ArgumentParser(description=f"Studies for {task}")
    parser.add_argument("--data-dir", default="/home/SOS/Downloads/projekt1")
    parser.add_argument("--problems", nargs="*", help="nazwy problemów, domyślnie oba z data.PROBLEMS")
    parser.add_argument("--sizes", nargs="*", type=int, default=[1000], choices=data.SIZES, help="rozmiary zbioru uczącego, np. --sizes 100 1000 10000")
    parser.add_argument("--out", default="results")
    parser.add_argument("--studies", nargs="*", choices=["baseline", *STUDIES], default=["baseline", *STUDIES])
    parser.add_argument("--seeds", type=int, nargs="*", help="seeds of the runs, e.g. --seeds 1 2 3; if not given, random ones are generated")
    parser.add_argument("--n-seeds", type=int, default=5, help="how many random seeds to generate when --seeds is not given")
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--activation", default="tanh", help="activation used in studies other than 'activation'")
    parser.add_argument("--hidden", type=int, nargs="*", default=[32, 16], help="baseline hidden layers")
    parser.add_argument("--quick", action="store_true", help="2 seeds, 100 epochs")
    parser.add_argument("--show", action="store_true", help="show figures (they are always saved)")
    args = parser.parse_args()
    
    if args.quick:
        args.seeds, args.epochs, args.patience = 2, 100, 20
    if not args.seeds:
        args.seeds = np.random.RandomState().randint(0, 2**31 - 1, args.n_seeds).tolist()
    print(f"Seeds: {args.seeds}  (reproduce with --seeds {' '.join(map(str, args.seeds))})")
    if not args.show:
        import matplotlib
        matplotlib.use("Agg")
    viz.set_style()
    
    metrics = REGRESSION_METRICS if task == "regression" else CLASSIFICATION_METRICS
    base = Config("baseline", args.hidden, args.activation, "mse" if task == "regression" else "ce",
                  lr=args.lr, batch_size=args.batch_size, max_epochs=args.epochs, patience=args.patience)
    
    for ds in get_datasets(args, task):
        out_dir = Path(args.out) / task / ds.name
        for name in args.studies:
            if name == "baseline":
                baseline(ds, base, args, out_dir)
            else:
                STUDIES[name](ds, base, args, out_dir, metrics)

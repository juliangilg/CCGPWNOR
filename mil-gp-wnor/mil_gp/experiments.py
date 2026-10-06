"""Experiment loop over inducing points / seeds and result summaries."""

import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .model import MILGP
from .train_eval import MILTrainer

# Settings used in the original notebooks for each dataset.
CONFIGS = {
    "camelyon16": dict(
        model=dict(input_dim=1024, num_dim=32, mode="correlated", num_latents=2, num_samples=50,
                   minis=-1, maxis=1, diag_sampling=True, elbo_scale=True),
        train=dict(inducing_values=(50,), seeds=(0,), epochs=10, lr=1e-3, ngd_lr=0.01,
                   instance_key=None),
        batch_size=1, load_at_init=False,
    ),
    "panda": dict(
        model=dict(input_dim=1024, num_dim=32, mode="correlated", num_latents=2, num_samples=50,
                   minis=-1, maxis=1, diag_sampling=True, elbo_scale=True),
        train=dict(inducing_values=(50,), seeds=(0,), epochs=10, lr=1e-3, ngd_lr=0.01,
                   instance_key=None),
        batch_size=32, load_at_init=True,
    ),
    "rsna": dict(
        model=dict(input_dim=2048, num_dim=32, mode="correlated", num_latents=2, num_samples=50,
                   minis=-1, maxis=1, diag_sampling=True, elbo_scale=True),
        train=dict(inducing_values=(50,), seeds=(0,), epochs=10, lr=1e-3, ngd_lr=0.01,
                   instance_key="y_inst"),
        batch_size=32, load_at_init=True,
    ),
}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def run_experiments(train_loader, val_loader, test_loader, save_dir, model_config,
                    inducing_values=(50,), seeds=(0,), epochs=100, lr=1e-3, ngd_lr=0.01,
                    instance_key=None, device=None):
    """Train/evaluate one model per (M, seed). Results are appended to CSVs in `save_dir`."""
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    mode = model_config.get("mode", "independent")
    results, histories = [], []

    for M in inducing_values:
        for seed in seeds:
            print(f"\n{'=' * 60}\nModel: {mode} | M={M} | seed={seed}\n{'=' * 60}")
            set_seed(seed)
            config = {**model_config, "M": int(M)}
            trainer = MILTrainer(MILGP(**config), device, len(train_loader.dataset), lr, ngd_lr)

            history, best_info, thresholds = trainer.fit(
                train_loader, val_loader, epochs,
                checkpoint_path=save_dir / f"best_{mode}_M{M}_seed{seed}.pth",
                model_config=config, seed=seed, instance_key=instance_key,
            )
            histories.append(history.assign(M=M, seed=seed, mode=mode))

            test_rows = trainer.evaluate(test_loader, thresholds, instance_key=instance_key)
            results += [{"M": M, "seed": seed, "mode": mode, **best_info, **r} for r in test_rows]
            print("\nTest results:\n" + pd.DataFrame(test_rows).round(4).to_string(index=False))

            # Save after every run so partial results are never lost.
            pd.DataFrame(results).to_csv(save_dir / "test_metrics.csv", index=False)
            pd.concat(histories).to_csv(save_dir / "training_history.csv", index=False)

    results_df = pd.DataFrame(results)
    return results_df, pd.concat(histories), summarize_results(results_df, save_dir)


def summarize_results(results_df, save_dir):
    """Mean ± std across seeds per (mode, M), saved as CSV and LaTeX."""
    save_dir = Path(save_dir)
    metrics = ["F1", "AUC", "NLPD", "Brier", "ECE", "AURC_entropy", "AURC_variance"]
    tables, latex = {}, []

    for level, df in results_df.groupby("level"):
        grouped = df.groupby(["mode", "M"])[metrics]
        mean, std = grouped.mean(), grouped.std()
        fmt = lambda m, s: "--" if pd.isna(m) else (f"{m:.4f}" if pd.isna(s) else f"{m:.4f} ± {s:.4f}")
        table = pd.DataFrame({c: [fmt(m, s) for m, s in zip(mean[c], std[c])] for c in metrics},
                             index=[f"{mo}, M={M}" for mo, M in mean.index])
        tables[level] = table

        print(f"\n{level}: mean ± std across seeds\n{table.to_string()}")
        table.to_csv(save_dir / f"summary_{level.lower()}.csv")
        grouped.agg(["mean", "std", "count"]).to_csv(save_dir / f"summary_numeric_{level.lower()}.csv")
        latex.append(f"% Level: {level}\n" + table.map(lambda s: s.replace("±", r"$\pm$"))
                     .rename(columns={"AURC_entropy": r"AURC$_H$", "AURC_variance": r"AURC$_V$"})
                     .to_latex(escape=False))

    (save_dir / "metrics_tables.tex").write_text("\n".join(latex), encoding="utf-8")
    return tables

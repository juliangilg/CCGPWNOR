"""Training loop (NGD + Adam on the ELBO) and evaluation metrics."""

from pathlib import Path

import gpytorch
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, roc_curve


# ----------------------------------------------------------------------------- metrics
def select_threshold(y_true, y_scores):
    """G-mean optimal threshold on the ROC curve (0.5 if only one class is present)."""
    y, p = np.asarray(y_true).ravel(), np.asarray(y_scores, dtype=float).ravel()
    if np.unique(y).size != 2:
        return 0.5
    fpr, tpr, thr = roc_curve(y, p)
    ok = np.isfinite(thr)
    return float(thr[ok][np.argmax(np.sqrt(tpr[ok] * (1 - fpr[ok])))])


def _aurc(errors, uncertainty):
    """Area under the risk-coverage curve (ties averaged)."""
    order = np.argsort(uncertainty)
    e = errors[order]
    _, starts, counts = np.unique(uncertainty[order], return_index=True, return_counts=True)
    e = np.repeat(np.add.reduceat(e, starts) / counts, counts)
    return (np.cumsum(e) / np.arange(1, len(e) + 1)).mean()


def _ece(y, p, n_bins=10):
    """Expected calibration error on the positive-class probability."""
    bins = np.minimum((p * n_bins).astype(int), n_bins - 1)
    return sum((bins == b).mean() * abs(p[bins == b].mean() - y[bins == b].mean())
               for b in range(n_bins) if (bins == b).any())


def binary_metrics(y_true, y_scores, threshold, y_vars=None):
    y, p = np.asarray(y_true).ravel(), np.asarray(y_scores, dtype=float).ravel()
    pred = (p >= threshold).astype(int)
    ps = np.clip(p, 1e-12, 1 - 1e-12)
    nll = -(y * np.log(ps) + (1 - y) * np.log1p(-ps))
    entropy = -(ps * np.log(ps) + (1 - ps) * np.log1p(-ps))
    errors = (pred != y).astype(float)
    return {
        "Accuracy": accuracy_score(y, pred),
        "F1": f1_score(y, pred, zero_division=0),
        "AUC": roc_auc_score(y, p) if np.unique(y).size == 2 else np.nan,
        "NLPD": nll.mean(),
        "Brier": np.mean((p - y) ** 2),
        "ECE": _ece(y, p),
        "AURC_entropy": _aurc(errors, entropy),
        "AURC_variance": _aurc(errors, np.asarray(y_vars).ravel()) if y_vars is not None else np.nan,
    }


# ----------------------------------------------------------------------------- trainer
class MILTrainer:
    def __init__(self, model, device, num_train_bags, lr=1e-3, ngd_lr=0.01):
        self.model = model.to(device)
        self.device = device

        # Natural variational parameters -> NGD; everything else -> Adam.
        var_params = list(self.model.gp_layer.variational_parameters())
        var_ids = {id(p) for p in var_params}
        self.ngd = gpytorch.optim.NGD(var_params, num_data=num_train_bags, lr=ngd_lr)
        self.adam = torch.optim.Adam(
            [p for p in self.model.parameters() if p.requires_grad and id(p) not in var_ids], lr=lr
        )
        self.mll = gpytorch.mlls.VariationalELBO(
            self.model.likelihood, self.model.gp_layer, num_data=num_train_bags
        )

    def _forward(self, batch):
        X = batch["X"].to(self.device).float()
        Y = batch["Y"].to(self.device).float().reshape(-1)
        mask = batch["mask"].to(self.device).bool()

        # The likelihood needs the bag structure of the flattened instances.
        lik = self.model.likelihood
        lik.B, lik.N = mask.shape
        lik.mask = mask
        return self.model(X.reshape(-1, X.shape[-1])), Y, mask

    def train_epoch(self, loader):
        self.model.train()
        total, n = 0.0, 0
        for batch in loader:
            self.ngd.zero_grad()
            self.adam.zero_grad()
            output, Y, _ = self._forward(batch)
            loss = -self.mll(output, Y)
            loss.backward()
            self.ngd.step()
            self.adam.step()
            total, n = total + loss.item() * Y.numel(), n + Y.numel()
        return total / n

    @torch.no_grad()
    def predict(self, loader, instance_key=None):
        """Bag (and optionally instance) labels, predictive means and variances."""
        self.model.eval()
        out = {k: [] for k in ["bag_y", "bag_p", "bag_var", "inst_y", "inst_p", "inst_var"]}
        for batch in loader:
            output, Y, mask = self._forward(batch)
            p, v, p_inst, v_inst, _, _ = self.model.likelihood(output)
            out["bag_y"].append(Y.cpu().numpy())
            out["bag_p"].append(p.cpu().numpy().ravel())
            out["bag_var"].append(v.cpu().numpy().ravel())
            if instance_key is not None:  # padded instances are excluded
                y_inst = batch[instance_key].to(self.device).reshape(mask.shape)
                out["inst_y"].append(y_inst[mask].cpu().numpy())
                out["inst_p"].append(p_inst[mask].cpu().numpy())
                out["inst_var"].append(v_inst[mask].cpu().numpy())
        return {k: np.concatenate(v) if v else None for k, v in out.items()}

    def fit(self, train_loader, val_loader, epochs, checkpoint_path, model_config, seed,
            instance_key=None):
        """Train, keep the best validation-AUC checkpoint and pick thresholds on validation."""
        checkpoint_path = Path(checkpoint_path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        best_auc, history = -np.inf, []

        for epoch in range(1, epochs + 1):
            train_loss = self.train_epoch(train_loader)
            pred = self.predict(val_loader)
            y, p = pred["bag_y"], pred["bag_p"]
            ps = np.clip(p.astype(float), 1e-12, 1 - 1e-12)
            row = {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_auc": roc_auc_score(y, p),
                "val_nlpd": -np.mean(y * np.log(ps) + (1 - y) * np.log1p(-ps)),
                "val_brier": np.mean((p - y) ** 2),
            }
            history.append(row)

            improved = row["val_auc"] > best_auc
            if improved:
                best_auc = row["val_auc"]
                torch.save({"model_state_dict": self.model.state_dict(), "model_config": model_config,
                            "seed": int(seed), "epoch": epoch, "val_auc": float(best_auc)},
                           checkpoint_path)
            print(f"Epoch {epoch:03d}/{epochs} | -ELBO {train_loss:.4f} | "
                  f"Val AUC {row['val_auc']:.4f} | NLPD {row['val_nlpd']:.4f} | "
                  f"Brier {row['val_brier']:.4f}" + (" | saved" if improved else ""), flush=True)

        # Reload best model and select thresholds on validation only.
        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model_state_dict"])
        pred = self.predict(val_loader, instance_key=instance_key)
        thresholds = {"Bag": select_threshold(pred["bag_y"], pred["bag_p"])}
        if instance_key is not None:
            thresholds["Instance"] = select_threshold(pred["inst_y"], pred["inst_p"])
        ckpt["thresholds"] = thresholds
        torch.save(ckpt, checkpoint_path)

        print(f"Best epoch {ckpt['epoch']} | Val AUC {ckpt['val_auc']:.4f} | thresholds {thresholds}")
        best_info = {"best_epoch": ckpt["epoch"], "best_val_auc": ckpt["val_auc"],
                     "checkpoint": str(checkpoint_path)}
        return pd.DataFrame(history), best_info, thresholds

    def evaluate(self, loader, thresholds, instance_key=None):
        """Test metrics at bag (and instance) level with fixed thresholds."""
        pred = self.predict(loader, instance_key=instance_key)
        levels = [("Bag", "bag")] + ([("Instance", "inst")] if instance_key else [])
        return [
            {"level": lvl, "threshold": thresholds[lvl],
             **binary_metrics(pred[f"{k}_y"], pred[f"{k}_p"], thresholds[lvl],
                              y_vars=np.maximum(pred[f"{k}_var"], 0))}
            for lvl, k in levels
        ]

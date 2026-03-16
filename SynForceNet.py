from __future__ import annotations

import argparse
import json
import os
import random
from dataclasses import asdict, dataclass
from math import gamma, pi
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import auc, precision_recall_curve, roc_auc_score, roc_curve
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

matplotlib.rcParams["font.family"] = "Times New Roman"


@dataclass
class Config:
    vin: str = "201"
    data_root: str = "./data"
    output_root: str = "./outputs"
    train_csv: str | None = None
    val_csv: str | None = None
    exp_name: str = "synforcenet_balanced_eval_splitA"
    seed: int = 42
    win_len: int = 60
    stride_tr: int = 5
    batch: int = 64
    epochs: int = 50
    lr: float = 1e-4
    latent: int = 6
    d_zg: int = 2
    w_diff: float = 0.01
    w_vol: float = 0.01
    lambda_stdp: float = 0.0125
    a_plus: float = 1.0
    tau_plus: float = 5.0
    max_lag: int = 6
    sig_force: float = 2.0
    force_soft: float = 0.05
    balanced_strategy: str = "random"
    nearby_window: int = 180
    print_epoch_eval: bool = False
    center_momentum: float = 0.99
    diffusion_k: int = 20
    diffusion_d: float = 1.0
    diffusion_sigma: float = 1.0
    volume_beta: float = 0.01
    volume_sigma: float = 1.0
    num_workers: int = 0

    @property
    def d_zl(self) -> int:
        return self.latent - self.d_zg

    def resolved_train_csv(self) -> Path:
        if self.train_csv is not None:
            return Path(self.train_csv)
        return Path(self.data_root) / self.vin / "train_seq_features.csv"

    def resolved_val_csv(self) -> Path:
        if self.val_csv is not None:
            return Path(self.val_csv)
        return Path(self.data_root) / self.vin / "val_seq_features.csv"

    def resolved_save_dir(self) -> Path:
        return Path(self.output_root) / self.vin / self.exp_name


def parse_args() -> Config:
    parser = argparse.ArgumentParser(description="SynForceNet training and evaluation")
    parser.add_argument("--vin", type=str, default="201")
    parser.add_argument("--data-root", type=str, default="./data")
    parser.add_argument("--output-root", type=str, default="./outputs")
    parser.add_argument("--train-csv", type=str, default=None)
    parser.add_argument("--val-csv", type=str, default=None)
    parser.add_argument("--exp-name", type=str, default="synforcenet_balanced_eval_splitA")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--win-len", type=int, default=60)
    parser.add_argument("--stride-tr", type=int, default=5)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--latent", type=int, default=6)
    parser.add_argument("--d-zg", type=int, default=2)
    parser.add_argument("--w-diff", type=float, default=0.01)
    parser.add_argument("--w-vol", type=float, default=0.01)
    parser.add_argument("--lambda-stdp", type=float, default=0.0125)
    parser.add_argument("--a-plus", type=float, default=1.0)
    parser.add_argument("--tau-plus", type=float, default=5.0)
    parser.add_argument("--max-lag", type=int, default=6)
    parser.add_argument("--sig-force", type=float, default=2.0)
    parser.add_argument("--force-soft", type=float, default=0.05)
    parser.add_argument("--balanced-strategy", type=str, choices=["random", "nearby"], default="random")
    parser.add_argument("--nearby-window", type=int, default=180)
    parser.add_argument("--print-epoch-eval", action="store_true")
    parser.add_argument("--center-momentum", type=float, default=0.99)
    parser.add_argument("--diffusion-k", type=int, default=20)
    parser.add_argument("--diffusion-d", type=float, default=1.0)
    parser.add_argument("--diffusion-sigma", type=float, default=1.0)
    parser.add_argument("--volume-beta", type=float, default=0.01)
    parser.add_argument("--volume-sigma", type=float, default=1.0)
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args()
    return Config(**vars(args))


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_plot(fig: plt.Figure, filename: str, save_dir: Path) -> None:
    save_path = save_dir / filename
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Saved plot: {save_path}")


def safe_torch_load(path: Path, map_location: torch.device):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def build_balanced_eval(df: pd.DataFrame, strategy: str, nearby_window: int, seed: int) -> pd.DataFrame:
    df = df.reset_index(drop=True)
    idx_a = np.where(df["label"].values == 1)[0]
    idx_n = np.where(df["label"].values == 0)[0]
    if len(idx_a) == 0:
        raise RuntimeError("No anomalies found in eval source.")
    if len(idx_n) == 0:
        raise RuntimeError("No normal samples found in eval source.")

    rng = np.random.default_rng(seed)

    if strategy == "random":
        replace = len(idx_n) < len(idx_a)
        chosen_n = rng.choice(idx_n, size=len(idx_a), replace=replace)
    elif strategy == "nearby":
        chosen = []
        n_set = set(idx_n.tolist())
        for i in idx_a:
            left = max(0, i - nearby_window)
            right = min(len(df) - 1, i + nearby_window)
            cand = [j for j in range(left, right + 1) if j in n_set]
            if len(cand) == 0:
                continue
            chosen.append(rng.choice(cand))

        chosen = np.array(chosen, dtype=int)
        if len(chosen) < len(idx_a):
            need = len(idx_a) - len(chosen)
            remain = np.setdiff1d(idx_n, chosen, assume_unique=False)
            pool = remain if len(remain) > 0 else idx_n
            replace = len(pool) < need
            extra = rng.choice(pool, size=need, replace=replace)
            chosen = np.concatenate([chosen, extra])

        chosen_n = chosen[: len(idx_a)]
    else:
        raise ValueError("strategy must be 'random' or 'nearby'")

    idx = np.concatenate([idx_a, chosen_n])
    rng.shuffle(idx)
    return df.iloc[idx].reset_index(drop=True)


class WindowDataset(Dataset):
    def __init__(self, df: pd.DataFrame, pc_cols: List[str], win_len: int, stride: int, only_normal: bool = False):
        self.x = df[pc_cols].values.astype(np.float32)
        self.y = df["label"].values.astype(np.int64)
        self.win_len = win_len
        self.starts: List[int] = []
        for s in range(0, len(df) - win_len + 1, stride):
            if only_normal and self.y[s : s + win_len].sum() != 0:
                continue
            self.starts.append(s)

    def __len__(self) -> int:
        return len(self.starts)

    def __getitem__(self, idx: int) -> torch.Tensor:
        s = self.starts[idx]
        return torch.from_numpy(self.x[s : s + self.win_len])


class SurrogateSpike(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, thresh: float) -> torch.Tensor:
        ctx.save_for_backward(x)
        return (x >= thresh).float()

    @staticmethod
    def backward(ctx, g: torch.Tensor):
        (x,) = ctx.saved_tensors
        return g * (1.0 / (1.0 + torch.abs(x)) ** 2), None


sur_spike = SurrogateSpike.apply


class SpikingLinear(nn.Module):
    def __init__(self, inp: int, out: int, thresh: float = 0.5):
        super().__init__()
        self.fc = nn.Linear(inp, out)
        self.th = thresh

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mem = self.fc(x)
        act = torch.sigmoid(mem)
        spk = sur_spike(act, self.th)
        return act, spk


class SNNEncoderSplit(nn.Module):
    def __init__(self, in_dim: int, d_zg: int, d_zl: int):
        super().__init__()
        self.l1 = SpikingLinear(in_dim, 512, 0.5)
        self.l2 = SpikingLinear(512, 256, 0.5)
        self.zg = nn.Linear(256, d_zg)
        self.zl = nn.Linear(256, d_zl)
        self.act = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x, _ = self.l1(x)
        x, _ = self.l2(x)
        z_global = self.act(self.zg(x))
        z_local = self.act(self.zl(x))
        return z_global, z_local


class Decoder(nn.Module):
    def __init__(self, latent_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, out_dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


recon_fn = nn.MSELoss()


def euclid(z: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    return torch.norm(z - c.unsqueeze(0), dim=1)


def diffusion_reg(z: torch.Tensor, k: int, diffusion_d: float, sigma: float) -> torch.Tensor:
    dist = torch.cdist(z, z)
    k = min(k, dist.shape[1])
    knn_d, knn_idx = dist.topk(k=k, largest=False)
    A = torch.exp(-(knn_d ** 2) / (2 * sigma ** 2))
    rho = A.sum(1)
    lap = (A * (rho[knn_idx] - rho.unsqueeze(1))).sum(1)
    return diffusion_d * (lap ** 2).mean()


def vol_comp(z: torch.Tensor, c: torch.Tensor, latent_dim: int, beta: float, sigma: float) -> torch.Tensor:
    d = torch.norm(z - c.unsqueeze(0), dim=1)
    rho = torch.exp(-(d ** 2) / (2 * sigma ** 2))
    R = d.max().clamp_min(1e-6)
    V = (pi ** (latent_dim / 2) / gamma(latent_dim / 2 + 1)) * (R ** latent_dim)
    vol = (d ** 3 * rho).sum() / rho.sum().clamp_min(1e-6)
    return beta * vol / V


def stdp_force_loss_seq(
    zl_seq: torch.Tensor,
    max_lag: int,
    a_plus: float,
    tau_plus: float,
    sig_force: float,
    force_soft: float,
) -> torch.Tensor:
    B, L, _ = zl_seq.shape
    if L <= max_lag + 1:
        return zl_seq.new_tensor(0.0)

    z_now = zl_seq[:, max_lag : L - 1, :]
    v = zl_seq[:, max_lag + 1 : L, :] - z_now
    F = torch.zeros_like(v)
    eps2 = force_soft ** 2

    for d in range(1, max_lag + 1):
        z_past = zl_seq[:, max_lag - d : L - 1 - d, :]
        diff = z_now - z_past
        dist2 = (diff * diff).sum(-1).clamp_min(eps2)
        force = diff / (dist2.unsqueeze(-1) + eps2)
        phi = torch.exp(-dist2 / (2 * sig_force ** 2))
        w = a_plus * torch.exp(-torch.tensor(float(d), device=zl_seq.device) / tau_plus)
        F = F + (w * phi).unsqueeze(-1) * force

    v_norm = torch.norm(v, dim=-1, keepdim=True).clamp_min(1e-9)
    F_norm = torch.norm(F, dim=-1, keepdim=True).clamp_min(1e-9)
    v_dir = v / v_norm
    F_dir = F / F_norm
    cos = (v_dir * F_dir).sum(-1).clamp(-1.0, 1.0)
    return (1.0 - cos).mean()


@torch.no_grad()
def select_threshold(y_true: np.ndarray, scores: np.ndarray) -> Tuple[float, float, float, float]:
    precision_vals, recall_vals, thresholds = precision_recall_curve(y_true, scores, pos_label=1)
    f1_max = -1.0
    cutoff = 0.0
    precision_at_cutoff = 0.0
    recall_at_cutoff = 0.0
    for i, t in enumerate(thresholds):
        precision = precision_vals[i + 1]
        recall = recall_vals[i + 1]
        f1 = 2 * precision * recall / (precision + recall + 1e-9)
        if f1 > f1_max:
            f1_max = float(f1)
            cutoff = float(t)
            precision_at_cutoff = float(precision)
            recall_at_cutoff = float(recall)
    return cutoff, precision_at_cutoff, recall_at_cutoff, f1_max


@torch.no_grad()
def score_points(
    enc: SNNEncoderSplit,
    center_c_g: torch.Tensor,
    df: pd.DataFrame,
    pc_cols: List[str],
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    X = df[pc_cols].values.astype(np.float32)
    y = df["label"].values.astype(np.int64)
    scores = []
    zg_all = []
    bs = 4096
    for i in range(0, len(X), bs):
        xb = torch.from_numpy(X[i : i + bs]).to(device)
        zg, _ = enc(xb)
        scores.append(euclid(zg, center_c_g).detach().cpu().numpy())
        zg_all.append(zg.detach().cpu().numpy())
    return np.concatenate(scores), y, np.concatenate(zg_all)


def to_2d(arr: np.ndarray) -> np.ndarray:
    if arr.shape[1] >= 2:
        return arr[:, :2]
    return np.concatenate([arr, np.zeros((len(arr), 1), dtype=arr.dtype)], axis=1)


def save_json(config: Config, save_dir: Path) -> None:
    with open(save_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(asdict(config), f, indent=2, ensure_ascii=False)


def train_and_eval(
    cfg: Config,
    device: torch.device,
    train_loader: DataLoader,
    bal_df: pd.DataFrame,
    pc_cols: List[str],
    in_dim: int,
    save_dir: Path,
) -> Dict[str, float]:
    enc = SNNEncoderSplit(in_dim, cfg.d_zg, cfg.d_zl).to(device)
    dec = Decoder(cfg.latent, in_dim).to(device)
    center_c_g = torch.randn(cfg.d_zg, device=device) * 0.1

    log_sig_r = nn.Parameter(torch.zeros(1, device=device))
    log_sig_s = nn.Parameter(torch.zeros(1, device=device))
    log_sig_e = nn.Parameter(torch.zeros(1, device=device))

    opt = optim.Adam(
        list(enc.parameters()) + list(dec.parameters()) + [log_sig_r, log_sig_s, log_sig_e],
        lr=cfg.lr,
    )

    epoch_metric_log = []
    prev_center_for_shift = center_c_g.detach().clone()
    running_f1_max = -1.0
    auc_best = -1.0

    @torch.no_grad()
    def ema_center_epoch() -> torch.Tensor:
        zg_all = []
        for xb in train_loader:
            xb = xb.to(device)
            zg, _ = enc(xb.reshape(-1, in_dim))
            zg_all.append(zg)
        return torch.cat(zg_all).mean(0)

    for epoch in range(cfg.epochs):
        enc.train()
        dec.train()

        epoch_recon = []
        epoch_svdd = []
        epoch_enc = []
        epoch_stdp = []
        epoch_total = []

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{cfg.epochs}", unit="batch")
        for xb in pbar:
            xb = xb.to(device)
            bsz = xb.size(0)
            x_flat = xb.reshape(-1, in_dim)

            zg, zl = enc(x_flat)
            z = torch.cat([zg, zl], dim=1)
            x_hat = dec(z)

            recon = recon_fn(x_hat, x_flat)
            svdd = euclid(zg, center_c_g).mean()

            zg_hat, zl_hat = enc(x_hat.detach())
            z_hat = torch.cat([zg_hat, zl_hat], dim=1)
            enc_cnst = recon_fn(z_hat, z.detach())

            zl_seq = zl.view(bsz, cfg.win_len, cfg.d_zl)
            stdpL = stdp_force_loss_seq(
                zl_seq=zl_seq,
                max_lag=cfg.max_lag,
                a_plus=cfg.a_plus,
                tau_plus=cfg.tau_plus,
                sig_force=cfg.sig_force,
                force_soft=cfg.force_soft,
            )

            diff = diffusion_reg(z, cfg.diffusion_k, cfg.diffusion_d, cfg.diffusion_sigma)
            ext_center = torch.cat([center_c_g, torch.zeros(cfg.d_zl, device=device)])
            vol = vol_comp(z, ext_center, cfg.latent, cfg.volume_beta, cfg.volume_sigma)

            loss = (
                (torch.exp(-2 * log_sig_r) * recon + log_sig_r)
                + (torch.exp(-2 * log_sig_s) * svdd + log_sig_s)
                + (torch.exp(-2 * log_sig_e) * enc_cnst + log_sig_e)
                + cfg.w_diff * diff
                + cfg.w_vol * vol
                + cfg.lambda_stdp * stdpL
            )

            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(enc.parameters(), 1.0)
            opt.step()

            epoch_recon.append(float(recon.item()))
            epoch_svdd.append(float(svdd.item()))
            epoch_enc.append(float(enc_cnst.item()))
            epoch_stdp.append(float(stdpL.item()))
            epoch_total.append(float(loss.item()))

            pbar.set_postfix(
                {
                    "Recon": f"{recon.item():.4f}",
                    "SVDD": f"{svdd.item():.6f}",
                    "Enc": f"{enc_cnst.item():.2e}",
                    "STDP": f"{stdpL.item():.3f}",
                }
            )

        with torch.no_grad():
            new_center = ema_center_epoch()
            center_shift = torch.norm(new_center - prev_center_for_shift).item()
            center_c_g.data = cfg.center_momentum * center_c_g.data + (1.0 - cfg.center_momentum) * new_center
            prev_center_for_shift = center_c_g.detach().clone()

        enc.eval()
        with torch.no_grad():
            scores, y_true, _ = score_points(enc, center_c_g, bal_df, pc_cols, device)
            auc_val = roc_auc_score(y_true, scores)
            threshold_now, precision_now, recall_now, f1_now = select_threshold(y_true, scores)

        pred = (scores > threshold_now).astype(int)
        pred_rate = float(pred.mean())
        TP = int(((pred == 1) & (y_true == 1)).sum())
        FP = int(((pred == 1) & (y_true == 0)).sum())
        FN = int(((pred == 0) & (y_true == 1)).sum())

        normal_scores = scores[y_true == 0]
        anomaly_scores = scores[y_true == 1]
        mean_normal = float(normal_scores.mean())
        mean_anomaly = float(anomaly_scores.mean())
        score_gap = mean_anomaly - mean_normal

        f1_best = max(f1_best, f1_now)
        auc_best = max(auc_best, auc_val)

        epoch_metric_log.append(
            {
                "epoch": epoch + 1,
                "AUC": float(auc_val),
                "F1": float(f1_now),
                "P": float(precision_now),
                "R": float(recall_now),
                "threshold": float(threshold_now),
                "pred_rate": pred_rate,
                "TP": TP,
                "FP": FP,
                "FN": FN,
                "score_min": float(scores.min()),
                "score_max": float(scores.max()),
                "mean_normal_score": mean_normal,
                "mean_anomaly_score": mean_anomaly,
                "score_gap": float(score_gap),
                "center_shift": float(center_shift),
                "recon_epoch_avg": float(np.mean(epoch_recon)),
                "svdd_epoch_avg": float(np.mean(epoch_svdd)),
                "enc_epoch_avg": float(np.mean(epoch_enc)),
                "stdp_epoch_avg": float(np.mean(epoch_stdp)),
                "total_epoch_avg": float(np.mean(epoch_total)),
                "f1_best": float(f1_best),
                "auc_best": float(auc_best),
            }
        )

        if cfg.print_epoch_eval:
            print(
                f"Epoch={epoch + 1}: AUC={auc_val:.3f}, F1={f1_now:.3f}, ",
                f"P={precision_now:.3f}, R={recall_now:.3f}, th={threshold_now:.6f}, gap={score_gap:.6f}"
            )

        torch.save(
            {
                "encoder": enc.state_dict(),
                "decoder": dec.state_dict(),
                "center_c_g": center_c_g.detach().cpu().clone(),
                "epoch": epoch + 1,
                "AUC": float(auc_val),
                "F1": float(f1_now),
                "P": float(precision_now),
                "R": float(recall_now),
                "threshold": float(threshold_now),
            },
            save_dir / f"model_epoch_{epoch + 1}.pth",
        )

    epoch_metrics_df = pd.DataFrame(epoch_metric_log)
    epoch_metrics_df.to_csv(save_dir / "epoch.csv", index=False)

    best_idx = int(epoch_metrics_df["F1"].idxmax())
    best_epoch = int(epoch_metrics_df.loc[best_idx, "epoch"])
    f1_value = float(epoch_metrics_df.loc[best_idx, "F1"])
    score_cutoff = float(epoch_metrics_df.loc[best_idx, "threshold"])

    checkpoint = safe_torch_load(save_dir / f"model_epoch_{best_epoch}.pth", map_location=device)
    enc.load_state_dict(checkpoint["encoder"])
    dec.load_state_dict(checkpoint["decoder"])
    center_c_g = checkpoint["center_c_g"].to(device)

    enc.eval()
    with torch.no_grad():
        final_scores, final_labels, final_zg = score_points(enc, center_c_g, bal_df, pc_cols, device)

    final_predictions = (final_scores > score_cutoff).astype(int)
    TP = int(((final_predictions == 1) & (final_labels == 1)).sum())
    FP = int(((final_predictions == 1) & (final_labels == 0)).sum())
    FN = int(((final_predictions == 0) & (final_labels == 1)).sum())
    PPV = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    TPR = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    final_auc = float(roc_auc_score(final_labels, final_scores))

    print(f"\n=== Best epoch={best_epoch}, F1={best_f1:.4f}, threshold={best_threshold:.6f} ===")
    print(f"TPR: {TPR:.4f}, PPV: {PPV:.4f}, AUC: {final_auc:.4f}")

    experimental_df = pd.DataFrame(
        {
            "epoch": [best_epoch],
            "TPR": [TPR],
            "PPV": [PPV],
            "F1": [f1_value],
            "AUC": [final_auc],
            "threshold": [score_cutoff],
            "score_min": [float(final_scores.min())],
            "score_max": [float(final_scores.max())],
            "lambda_stdp": [float(cfg.lambda_stdp)],
        }
    )
    experimental_df.to_csv(save_dir / "summary.csv", index=False)

    latent_vectors = to_2d(final_zg)
    center_np = center_c_g.detach().cpu().numpy()
    center_xy = center_np[:2] if len(center_np) >= 2 else np.array([center_np[0], 0.0], dtype=np.float32)

    fig = plt.figure(figsize=(10, 6))
    plt.scatter(latent_vectors[final_labels == 0, 0], latent_vectors[final_labels == 0, 1], label="Normal", alpha=0.5)
    plt.scatter(latent_vectors[final_labels == 1, 0], latent_vectors[final_labels == 1, 1], label="Anomaly", alpha=0.5, marker="x")
    plt.scatter(center_xy[0], center_xy[1], c="red", s=100, label="Center")
    plt.legend()
    plt.title("Latent Space Distribution")
    save_plot(fig, "latent.png", save_dir)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(epoch_metrics_df["epoch"], epoch_metrics_df["recon_epoch_avg"], marker="o", label="recon_loss")
    ax.plot(epoch_metrics_df["epoch"], epoch_metrics_df["svdd_epoch_avg"], marker="o", label="svdd_loss")
    ax.plot(epoch_metrics_df["epoch"], epoch_metrics_df["enc_epoch_avg"], marker="o", label="enc_loss")
    ax.plot(epoch_metrics_df["epoch"], epoch_metrics_df["stdp_epoch_avg"], marker="o", label="stdp_loss")
    ax.plot(epoch_metrics_df["epoch"], epoch_metrics_df["total_epoch_avg"], marker="o", label="total_loss")
    ax.set_title(f"Loss Trend over Epochs (final epoch = {cfg.epochs})")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.legend()
    save_plot(fig, "loss.png", save_dir)
    plt.close(fig)

    fig = plt.figure(figsize=(10, 6))
    plt.plot(epoch_metrics_df["epoch"], epoch_metrics_df["F1"], marker="o", linewidth=2, label="f1")
    plt.plot(epoch_metrics_df["epoch"], epoch_metrics_df["f1_best"], marker="s", linewidth=2, label="f1_best")
    plt.title("F1 Trend")
    plt.xlabel("Epoch")
    plt.ylabel("F1 Score")
    plt.grid(True, alpha=0.3)
    plt.legend()
    save_plot(fig, "f1.png", save_dir)
    plt.close(fig)

    fig = plt.figure(figsize=(16, 6))
    sc = plt.scatter(
        x=np.arange(len(final_labels)),
        y=final_scores,
        c=final_labels,
        cmap="coolwarm",
        vmin=0,
        vmax=1,
        alpha=0.6,
        edgecolors="w",
        linewidths=0.5,
    )
    plt.colorbar(sc, label="True Label (0=Normal, 1=Anomaly)")
    plt.axhline(y=score_cutoff, color="r", linestyle="--", label="Threshold")
    plt.xlabel("Sample Index")
    plt.ylabel("Anomaly Score")
    plt.title(f"Score Distribution (epoch={best_epoch})")
    plt.legend()
    save_plot(fig, "scores.png", save_dir)
    plt.close(fig)

    fpr, tpr, _ = roc_curve(final_labels, final_scores, pos_label=1)
    roc_auc_val = auc(fpr, tpr)
    fig = plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, color="magenta", marker="o", linewidth=2, label=f"ROC curve (area = {roc_auc_val:.2f})")
    plt.plot([0, 1], [0, 1], color="orange", linestyle="--", label="y = x")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xticks(np.arange(0, 1.1, 0.2))
    plt.yticks(np.arange(0, 1.1, 0.2))
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic (ROC)")
    plt.legend(loc="lower right")
    save_plot(fig, "roc.png", save_dir)
    plt.close(fig)

    fig = plt.figure(figsize=(10, 6))
    plt.plot(epoch_metrics_df["epoch"], epoch_metrics_df["AUC"], marker="o", linewidth=2, label="auc")
    plt.plot(epoch_metrics_df["epoch"], epoch_metrics_df["auc_best"], marker="s", linewidth=2, label="auc_best")
    plt.title("AUC Trend")
    plt.xlabel("Epoch")
    plt.ylabel("AUC")
    plt.grid(True, alpha=0.3)
    plt.legend()
    save_plot(fig, "auc.png", save_dir)
    plt.close(fig)

    fig = plt.figure(figsize=(10, 6))
    plt.plot(epoch_metrics_df["epoch"], epoch_metrics_df["mean_normal_score"], marker="o", linewidth=2, label="mean_normal_score")
    plt.plot(epoch_metrics_df["epoch"], epoch_metrics_df["mean_anomaly_score"], marker="o", linewidth=2, label="mean_anomaly_score")
    plt.plot(epoch_metrics_df["epoch"], epoch_metrics_df["score_gap"], marker="s", linewidth=2, label="score_gap")
    plt.title("Score Separation Trend")
    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.grid(True, alpha=0.3)
    plt.legend()
    save_plot(fig, "gap.png", save_dir)
    plt.close(fig)

    fig = plt.figure(figsize=(10, 6))
    plt.plot(epoch_metrics_df["epoch"], epoch_metrics_df["center_shift"], marker="o", linewidth=2)
    plt.title("Center Shift Trend")
    plt.xlabel("Epoch")
    plt.ylabel("Center Shift")
    plt.grid(True, alpha=0.3)
    save_plot(fig, "center.png", save_dir)
    plt.close(fig)

    result = {
        "tag": "STDP_local_only",
        "lambda_stdp": float(cfg.lambda_stdp),
        "best_epoch": best_epoch,
        "AUC": final_auc,
        "F1": f1_value,
        "P": PPV,
        "R": TPR,
        "threshold": score_cutoff,
        "score_min": float(final_scores.min()),
        "score_max": float(final_scores.max()),
    }
    pd.DataFrame([result]).to_csv(save_dir / "result.csv", index=False)
    return result


def main() -> None:
    cfg = parse_args()
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    save_dir = cfg.resolved_save_dir()
    save_dir.mkdir(parents=True, exist_ok=True)
    save_json(cfg, save_dir)

    train_csv = cfg.resolved_train_csv()
    val_csv = cfg.resolved_val_csv()
    if not train_csv.exists():
        raise FileNotFoundError(f"Train CSV not found: {train_csv}")
    if not val_csv.exists():
        raise FileNotFoundError(f"Validation CSV not found: {val_csv}")

    train_df = pd.read_csv(train_csv).sort_values("TIME").reset_index(drop=True)
    val_df = pd.read_csv(val_csv).sort_values("TIME").reset_index(drop=True)
    if "label" not in val_df.columns:
        raise RuntimeError("Validation CSV must contain a 'label' column.")

    pc_cols = [c for c in train_df.columns if c.startswith("PC")]
    if len(pc_cols) == 0:
        raise RuntimeError("No PCA feature columns found. Expected columns starting with 'PC'.")

    in_dim = len(pc_cols)
    print("[Feat] IN_DIM =", in_dim)

    bal_df = build_balanced_eval(
        val_df,
        strategy=cfg.balanced_strategy,
        nearby_window=cfg.nearby_window,
        seed=cfg.seed,
    )
    bal_path = save_dir / "eval.csv"
    bal_df.to_csv(bal_path, index=False)
    print("[Balanced eval saved]", bal_path, "shape=", bal_df.shape, "pos=", int(bal_df["label"].sum()))

    train_set = WindowDataset(train_df, pc_cols, cfg.win_len, cfg.stride_tr, only_normal=True)
    train_loader = DataLoader(
        train_set,
        batch_size=cfg.batch,
        shuffle=True,
        drop_last=True,
        num_workers=cfg.num_workers,
    )
    print("[Train windows]", len(train_set))

    result = train_and_eval(cfg, device, train_loader, bal_df, pc_cols, in_dim, save_dir)
    print("\n=== Balanced point-level result (global SVDD, local STDP) ===")
    print(
        f"[{result['tag']}] lambda={result['lambda_stdp']}, epoch={result['best_epoch']}: "
        f"AUC={result['AUC']:.3f}, F1={result['F1']:.3f}, P={result['P']:.3f}, "
        f"R={result['R']:.3f}, th={result['threshold']:.6f}"
    )
    print("[Saved]", save_dir / "result.csv")


if __name__ == "__main__":
    main()

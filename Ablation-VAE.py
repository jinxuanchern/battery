import argparse
import json
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path

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

matplotlib.rcParams['font.family'] = 'Times New Roman'


@dataclass
class Config:
    vin: str = '201'
    data_root: str = './data'
    output_root: str = './outputs'
    train_name: str = 'train_features(2).csv'
    test_name: str = 'test_features(2).csv'
    exp_name: str = 'VAE'
    seed: int = 42
    batch_size: int = 512
    latent_dim: int = 6
    epochs: int = 50
    lr: float = 1e-4
    beta: float = 1e-3
    feature_cols: tuple = ('PC1', 'PC2', 'PC3', 'PC4', 'PC5', 'PC6')
    label_col: str = 'label'

    @property
    def train_csv(self) -> Path:
        return Path(self.data_root) / self.vin / self.train_name

    @property
    def test_csv(self) -> Path:
        return Path(self.data_root) / self.vin / self.test_name

    @property
    def save_dir(self) -> Path:
        return Path(self.output_root) / self.vin / self.exp_name


class ArrayDataset(Dataset):
    def __init__(self, data: np.ndarray):
        self.data = torch.tensor(data, dtype=torch.float32)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return self.data[idx]


class Encoder(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 256)
        self.mu = nn.Linear(256, latent_dim)
        self.log_var = nn.Linear(256, latent_dim)
        self.act = nn.LeakyReLU(0.2)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)

    def forward(self, x):
        x = self.act(self.fc1(x))
        x = self.bn1(x)
        x = self.act(self.fc2(x))
        x = self.bn2(x)
        x = self.act(self.fc3(x))
        mu = self.mu(x)
        log_var = torch.clamp(self.log_var(x), min=-5.0, max=5.0)
        return mu, log_var


class Decoder(nn.Module):
    def __init__(self, latent_dim: int, output_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(latent_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 512)
        self.fc4 = nn.Linear(512, output_dim)
        self.act = nn.LeakyReLU(0.2)
        self.out = nn.Sigmoid()

    def forward(self, z):
        z = self.act(self.fc1(z))
        z = self.act(self.fc2(z))
        z = self.act(self.fc3(z))
        return self.out(self.fc4(z))


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_plot(fig, path: Path):
    fig.savefig(path, bbox_inches='tight', dpi=300)
    print(f'saved: {path}')


def sample_latent(mu, log_var):
    std = torch.exp(0.5 * log_var)
    eps = torch.randn_like(std)
    return mu + eps * std


def loss_terms(x, x_hat, mu, log_var, beta: float):
    recon = torch.mean((x_hat - x) ** 2)
    reg = -0.5 * torch.mean(torch.sum(1 + log_var - mu.pow(2) - torch.exp(log_var), dim=1))
    total = recon + beta * reg
    return total, recon, reg


def score_batch(x, encoder, decoder, beta: float):
    mu, log_var = encoder(x)
    z = mu
    x_hat = decoder(z)
    recon = torch.mean((x_hat - x) ** 2, dim=1)
    reg = 0.5 * torch.sum(mu.pow(2) + torch.exp(log_var) - 1 - log_var, dim=1)
    score = recon + beta * reg
    return score, mu, x_hat


def select_threshold(y_true: np.ndarray, scores: np.ndarray):
    p_list, r_list, t_list = precision_recall_curve(y_true, scores, pos_label=1)
    if len(t_list) == 0:
        return 0.0, 0.0, 0.0, 0.0

    f1_max = -1.0
    cut = float(t_list[0])
    p_out = 0.0
    r_out = 0.0

    for i, t in enumerate(t_list):
        p_val = float(p_list[i + 1])
        r_val = float(r_list[i + 1])
        f1_val = 2 * p_val * r_val / (p_val + r_val) if (p_val + r_val) > 0 else 0.0
        if f1_val > f1_max:
            f1_max = f1_val
            cut = float(t)
            p_out = p_val
            r_out = r_val

    return f1_max, cut, p_out, r_out


def load_data(cfg: Config):
    train_df = pd.read_csv(cfg.train_csv)
    test_df = pd.read_csv(cfg.test_csv)
    x_train = train_df[list(cfg.feature_cols)].values.astype(np.float32)
    x_test = test_df[list(cfg.feature_cols)].values.astype(np.float32)
    y_test = test_df[cfg.label_col].values.astype(np.int64)
    return x_train, x_test, y_test


def eval_epoch(encoder, decoder, loader, labels, beta: float, device):
    encoder.eval()
    decoder.eval()
    scores = []
    with torch.no_grad():
        for xb in loader:
            xb = xb.to(device)
            s, _, _ = score_batch(xb, encoder, decoder, beta=beta)
            scores.extend(s.cpu().numpy())
    scores = np.asarray(scores)
    f1_val, cut, p_val, r_val = select_threshold(labels, scores)
    auc_val = float(roc_auc_score(labels, scores))
    return {
        'tpr': r_val,
        'ppv': p_val,
        'f1': f1_val,
        'auc': auc_val,
        'cut': cut,
        'min': float(scores.min()),
        'max': float(scores.max()),
        'scores': scores,
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--vin', type=str, default='201')
    parser.add_argument('--data-root', type=str, default='./data')
    parser.add_argument('--output-root', type=str, default='./outputs')
    parser.add_argument('--train-name', type=str, default='train_features(2).csv')
    parser.add_argument('--test-name', type=str, default='test_features(2).csv')
    parser.add_argument('--exp-name', type=str, default='VAE')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--latent-dim', type=int, default=6)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--beta', type=float, default=1e-3)
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = Config(
        vin=args.vin,
        data_root=args.data_root,
        output_root=args.output_root,
        train_name=args.train_name,
        test_name=args.test_name,
        exp_name=args.exp_name,
        seed=args.seed,
        batch_size=args.batch_size,
        latent_dim=args.latent_dim,
        epochs=args.epochs,
        lr=args.lr,
        beta=args.beta,
    )

    cfg.save_dir.mkdir(parents=True, exist_ok=True)
    with open(cfg.save_dir / 'config.json', 'w', encoding='utf-8') as f:
        json.dump(asdict(cfg), f, ensure_ascii=False, indent=2)

    set_seed(cfg.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device: {device}')
    print(f'train: {cfg.train_csv}')
    print(f'test : {cfg.test_csv}')
    print(f'out  : {cfg.save_dir}')

    x_train, x_test, y_test = load_data(cfg)
    train_loader = DataLoader(ArrayDataset(x_train), batch_size=cfg.batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(ArrayDataset(x_test), batch_size=cfg.batch_size, shuffle=False)

    encoder = Encoder(input_dim=x_train.shape[1], latent_dim=cfg.latent_dim).to(device)
    decoder = Decoder(latent_dim=cfg.latent_dim, output_dim=x_train.shape[1]).to(device)
    optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=cfg.lr)

    loss_log = {'epoch': [], 'recon': [], 'reg': [], 'total': []}

    for epoch in range(cfg.epochs):
        encoder.train()
        decoder.train()

        recon_sum = 0.0
        reg_sum = 0.0
        total_sum = 0.0
        n_steps = 0

        pbar = tqdm(train_loader, desc=f'epoch {epoch + 1}/{cfg.epochs}', unit='batch')
        for xb in pbar:
            xb = xb.to(device)
            optimizer.zero_grad()

            mu, log_var = encoder(xb)
            z = sample_latent(mu, log_var)
            x_hat = decoder(z)

            total, recon, reg = loss_terms(xb, x_hat, mu, log_var, beta=cfg.beta)
            total.backward()
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(decoder.parameters(), max_norm=1.0)
            optimizer.step()

            recon_sum += recon.item()
            reg_sum += reg.item()
            total_sum += total.item()
            n_steps += 1

            pbar.set_postfix({
                'recon': f'{recon.item():.6f}',
                'reg': f'{reg.item():.6f}',
                'total': f'{total.item():.6f}',
            })

        recon_avg = recon_sum / max(n_steps, 1)
        reg_avg = reg_sum / max(n_steps, 1)
        total_avg = total_sum / max(n_steps, 1)

        loss_log['epoch'].append(epoch + 1)
        loss_log['recon'].append(recon_avg)
        loss_log['reg'].append(reg_avg)
        loss_log['total'].append(total_avg)

        print(f'epoch {epoch + 1}: recon={recon_avg:.6f}, reg={reg_avg:.6f}, total={total_avg:.6f}')

        torch.save(
            {
                'encoder': encoder.state_dict(),
                'decoder': decoder.state_dict(),
                'epoch': epoch + 1,
            },
            cfg.save_dir / f'model_{epoch + 1}.pth',
        )

    rows = []
    epoch_pick = None
    f1_pick = -1.0
    cut_pick = 0.0
    best = {}

    for epoch in range(1, cfg.epochs + 1):
        ckpt = cfg.save_dir / f'model_{epoch}.pth'
        if not ckpt.exists():
            continue

        state = torch.load(ckpt, map_location=device)
        encoder.load_state_dict(state['encoder'])
        decoder.load_state_dict(state['decoder'])

        res = eval_epoch(encoder, decoder, test_loader, y_test, beta=cfg.beta, device=device)
        rows.append({
            'epoch': epoch,
            'TPR': res['tpr'],
            'PPV': res['ppv'],
            'F1': res['f1'],
            'AUC': res['auc'],
            'threshold': res['cut'],
            'score_min': res['min'],
            'score_max': res['max'],
        })

        if res['f1'] > f1_pick:
            f1_pick = res['f1']
            cut_pick = res['cut']
            epoch_pick = epoch
            best = res

    epoch_df = pd.DataFrame(rows)
    epoch_df.to_csv(cfg.save_dir / 'epoch.csv', index=False)

    print(f'\nselected: epoch={epoch_pick}, f1={f1_pick:.6f}, threshold={cut_pick:.6f}')
    print(f"TPR={best['tpr']:.6f}, PPV={best['ppv']:.6f}, AUC={best['auc']:.6f}")

    state = torch.load(cfg.save_dir / f'model_{epoch_pick}.pth', map_location=device)
    encoder.load_state_dict(state['encoder'])
    decoder.load_state_dict(state['decoder'])
    encoder.eval()
    decoder.eval()

    scores_all = []
    latent_all = []
    with torch.no_grad():
        for xb in test_loader:
            xb = xb.to(device)
            score, mu, _ = score_batch(xb, encoder, decoder, beta=cfg.beta)
            scores_all.extend(score.cpu().numpy())
            latent_all.append(mu.cpu().numpy())

    scores_all = np.asarray(scores_all)
    latent_all = np.concatenate(latent_all, axis=0)

    summary_df = pd.DataFrame([
        {
            'epoch': epoch_pick,
            'TPR': best['tpr'],
            'PPV': best['ppv'],
            'F1': best['f1'],
            'AUC': best['auc'],
            'threshold': best['cut'],
            'score_min': float(scores_all.min()),
            'score_max': float(scores_all.max()),
        }
    ])
    summary_df.to_csv(cfg.save_dir / 'summary.csv', index=False)

    loss_df = pd.DataFrame(loss_log)
    loss_df.to_csv(cfg.save_dir / 'loss.csv', index=False)

    plt.figure(figsize=(10, 6))
    plt.scatter(latent_all[y_test == 0, 0], latent_all[y_test == 0, 1], label='Normal', alpha=0.5)
    plt.scatter(latent_all[y_test == 1, 0], latent_all[y_test == 1, 1], label='Anomaly', alpha=0.5, marker='x')
    plt.legend()
    plt.title('Latent Space')
    save_plot(plt.gcf(), cfg.save_dir / 'latent.png')
    plt.close()

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(loss_df['epoch'], loss_df['recon'], marker='o', label='recon')
    ax.plot(loss_df['epoch'], loss_df['reg'], marker='o', label='reg')
    ax.plot(loss_df['epoch'], loss_df['total'], marker='o', label='total')
    ax.set_title('Loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Value')
    ax.grid(True, alpha=0.3)
    ax.legend()
    save_plot(fig, cfg.save_dir / 'loss.png')
    plt.close(fig)

    plt.figure(figsize=(10, 6))
    plt.plot(epoch_df['epoch'], epoch_df['F1'], marker='o', linewidth=2)
    plt.title('F1')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.grid(True, alpha=0.3)
    save_plot(plt.gcf(), cfg.save_dir / 'f1.png')
    plt.close()

    plt.figure(figsize=(16, 6))
    sc = plt.scatter(
        x=np.arange(len(y_test)),
        y=scores_all,
        c=y_test,
        cmap='coolwarm',
        vmin=0,
        vmax=1,
        alpha=0.6,
        edgecolors='w',
        linewidths=0.5,
    )
    plt.colorbar(sc, label='Label')
    plt.axhline(y=best['cut'], color='r', linestyle='--', label='Threshold')
    plt.xlabel('Index')
    plt.ylabel('Score')
    plt.title('Scores')
    plt.legend()
    save_plot(plt.gcf(), cfg.save_dir / 'scores.png')
    plt.close()

    fpr, tpr, _ = roc_curve(y_test, scores_all, pos_label=1)
    roc_val = auc(fpr, tpr)
    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, color='magenta', marker='o', linewidth=2, label=f'ROC (AUC = {roc_val:.2f})')
    plt.plot([0, 1], [0, 1], color='orange', linestyle='--', label='y = x')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xticks(np.arange(0, 1.1, 0.2))
    plt.yticks(np.arange(0, 1.1, 0.2))
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC')
    plt.legend(loc='lower right')
    save_plot(plt.gcf(), cfg.save_dir / 'roc.png')
    plt.close()

    result_df = pd.DataFrame([
        {
            'tag': cfg.exp_name,
            'epoch': epoch_pick,
            'AUC': best['auc'],
            'F1': best['f1'],
            'P': best['ppv'],
            'R': best['tpr'],
            'threshold': best['cut'],
            'score_min': float(scores_all.min()),
            'score_max': float(scores_all.max()),
        }
    ])
    result_df.to_csv(cfg.save_dir / 'result.csv', index=False)

    print('\nfinished.')
    print(f'best epoch: {epoch_pick}')
    print({
        'TPR': best['tpr'],
        'PPV': best['ppv'],
        'F1': best['f1'],
        'AUC': best['auc'],
        'threshold': best['cut'],
    })


if __name__ == '__main__':
    main()

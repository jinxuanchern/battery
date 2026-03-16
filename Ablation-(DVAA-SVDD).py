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
from sklearn.metrics import precision_recall_curve, roc_auc_score, roc_curve, auc
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

matplotlib.rcParams['font.family'] = 'Times New Roman'


@dataclass
class Config:
    vin: str = '099'
    data_root: str = './data'
    output_root: str = './outputs'
    exp_name: str = 'adv_svdd'
    file_tag: str = '2'
    feature_cols: tuple = ('PC1', 'PC2', 'PC3', 'PC4', 'PC5', 'PC6')
    label_col: str = 'label'
    seed: int = 42
    batch_size: int = 512
    latent_dim: int = 6
    epochs: int = 20
    lr_gen: float = 1e-4
    lr_disc: float = 1e-4
    gp_weight: float = 2.0
    adv_weight: float = 0.1
    eps_cov: float = 1e-3
    keep_ratio: float = 0.9


class SampleSet(Dataset):
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
        x = self.bn1(self.act(self.fc1(x)))
        x = self.bn2(self.act(self.fc2(x)))
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


class Discriminator(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 256)
        self.fc4 = nn.Linear(256, 1)
        self.act = nn.LeakyReLU(0.2)
        self.out = nn.Sigmoid()

    def forward(self, x):
        x = self.act(self.fc1(x))
        x = self.act(self.fc2(x))
        x = self.act(self.fc3(x))
        return self.out(self.fc4(x))


class Runner:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.out_dir = Path(cfg.output_root) / cfg.vin / cfg.exp_name
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self._set_seed(cfg.seed)

        train_csv = Path(cfg.data_root) / cfg.vin / f'train_features({cfg.file_tag}).csv'
        test_csv = Path(cfg.data_root) / cfg.vin / f'test_features({cfg.file_tag}).csv'
        train_df = pd.read_csv(train_csv)
        test_df = pd.read_csv(test_csv)

        self.train_x = train_df[list(cfg.feature_cols)].values.astype(np.float32)
        self.test_x = test_df[list(cfg.feature_cols)].values.astype(np.float32)
        self.test_y = test_df[cfg.label_col].values.astype(np.int64)

        self.train_loader = DataLoader(SampleSet(self.train_x), batch_size=cfg.batch_size, shuffle=True, drop_last=True)
        self.test_loader = DataLoader(SampleSet(self.test_x), batch_size=cfg.batch_size, shuffle=False)

        self.enc = Encoder(self.train_x.shape[1], cfg.latent_dim).to(self.device)
        self.dec = Decoder(cfg.latent_dim, self.train_x.shape[1]).to(self.device)
        self.disc = Discriminator(self.train_x.shape[1]).to(self.device)
        self.center = nn.Parameter(torch.randn(cfg.latent_dim, device=self.device) * 0.1)

        self.opt_g = optim.Adam(list(self.enc.parameters()) + list(self.dec.parameters()) + [self.center], lr=cfg.lr_gen)
        self.opt_d = optim.Adam(self.disc.parameters(), lr=cfg.lr_disc)

        self.mse = nn.MSELoss()
        self.bce = nn.BCELoss()

    @staticmethod
    def _set_seed(seed: int):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def save_plot(self, fig, name: str):
        path = self.out_dir / name
        fig.savefig(path, bbox_inches='tight', dpi=300)
        print(f'saved: {path}')

    def grad_penalty(self, real_x, fake_x):
        alpha = torch.rand(real_x.size(0), 1, device=real_x.device)
        mix = (alpha * real_x + (1 - alpha) * fake_x).requires_grad_(True)
        out = self.disc(mix)
        grad = torch.autograd.grad(
            outputs=out,
            inputs=mix,
            grad_outputs=torch.ones_like(out),
            create_graph=True,
            retain_graph=True,
        )[0]
        return ((grad.norm(2, dim=1) - 1) ** 2).mean()

    def sample_z(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        return mu + std * torch.randn_like(std)

    def score_fn(self, mu, log_var, center):
        eye = torch.eye(mu.size(1), device=mu.device).unsqueeze(0)
        cov = torch.diag_embed(torch.exp(log_var)) + self.cfg.eps_cov * eye
        inv_cov = torch.inverse(cov)
        diff = mu - center
        prod = torch.bmm(diff.unsqueeze(1), inv_cov).squeeze(1)
        return torch.sum(prod * diff, dim=1)

    def select_threshold(self, y_true: np.ndarray, scores: np.ndarray):
        prec, rec, ths = precision_recall_curve(y_true, scores, pos_label=1)
        if len(ths) == 0:
            return 0.0, 0.0, 0.0
        f1_best = -1.0
        th_best = float(ths[0])
        p_best = 0.0
        r_best = 0.0
        for i, th in enumerate(ths):
            p = float(prec[i + 1])
            r = float(rec[i + 1])
            f1 = 2 * p * r / (p + r + 1e-9)
            if f1 > f1_best:
                f1_best = f1
                th_best = float(th)
                p_best = p
                r_best = r
        return th_best, p_best, r_best

    @torch.no_grad()
    def update_center(self):
        all_mu = []
        self.enc.eval()
        for xb in self.train_loader:
            xb = xb.to(self.device)
            mu, _ = self.enc(xb)
            all_mu.append(mu)
        new_center = torch.cat(all_mu, dim=0).mean(dim=0)
        self.center.data = 0.99 * self.center.data + 0.01 * new_center

    @torch.no_grad()
    def infer(self):
        self.enc.eval()
        scores_all = []
        mu_all = []
        for xb in self.test_loader:
            xb = xb.to(self.device)
            mu, log_var = self.enc(xb)
            scores = self.score_fn(mu, log_var, self.center)
            scores_all.append(scores.cpu().numpy())
            mu_all.append(mu.cpu().numpy())
        return np.concatenate(scores_all), np.concatenate(mu_all, axis=0)

    def train(self):
        loss_rows = []
        epoch_rows = []
        f1_run = 0.0
        auc_run = 0.0

        for epoch in range(1, self.cfg.epochs + 1):
            self.enc.train()
            self.dec.train()
            self.disc.train()

            recon_vals = []
            score_vals = []
            align_vals = []
            adv_vals = []
            total_vals = []

            pbar = tqdm(self.train_loader, desc=f'epoch {epoch}/{self.cfg.epochs}', unit='batch')
            for xb in pbar:
                xb = xb.to(self.device)
                keep_n = max(1, int(xb.size(0) * self.cfg.keep_ratio))
                xk = xb[:keep_n]

                self.opt_d.zero_grad()
                mu, log_var = self.enc(xk)
                z = self.sample_z(mu, log_var)
                x_hat = self.dec(z)
                real_out = self.disc(xk)
                fake_out = self.disc(x_hat.detach())
                gp = self.grad_penalty(xk, x_hat.detach())
                d_loss = self.cfg.adv_weight * (
                    self.bce(real_out, torch.ones_like(real_out)) +
                    self.bce(fake_out, torch.zeros_like(fake_out))
                ) + self.cfg.gp_weight * gp
                d_loss.backward()
                self.opt_d.step()

                self.opt_g.zero_grad()
                mu, log_var = self.enc(xk)
                z = self.sample_z(mu, log_var)
                x_hat = self.dec(z)
                score = self.score_fn(z, log_var, self.center).mean()
                recon = self.mse(x_hat, xk)
                mu_hat, log_var_hat = self.enc(x_hat)
                z_hat = self.sample_z(mu_hat, log_var_hat)
                align = self.mse(z, z_hat)
                fake_out_g = self.disc(x_hat)
                adv = self.cfg.adv_weight * self.bce(fake_out_g, torch.ones_like(fake_out_g))
                total = 0.25 * recon + 0.25 * score + 0.25 * align + 0.25 * adv
                total.backward()
                nn.utils.clip_grad_norm_(self.enc.parameters(), 1.0)
                nn.utils.clip_grad_norm_(self.dec.parameters(), 1.0)
                nn.utils.clip_grad_norm_(self.disc.parameters(), 1.0)
                self.opt_g.step()

                recon_vals.append(recon.item())
                score_vals.append(score.item())
                align_vals.append(align.item())
                adv_vals.append(adv.item())
                total_vals.append(total.item())
                loss_rows.append({
                    'step': len(loss_rows) + 1,
                    'recon': recon.item(),
                    'score': score.item(),
                    'align': align.item(),
                    'adv': adv.item(),
                    'total': total.item(),
                })
                pbar.set_postfix({
                    'recon': f'{recon.item():.5f}',
                    'score': f'{score.item():.5f}',
                    'align': f'{align.item():.5f}',
                    'adv': f'{adv.item():.5f}',
                })

            self.update_center()

            ckpt = {
                'enc': self.enc.state_dict(),
                'dec': self.dec.state_dict(),
                'disc': self.disc.state_dict(),
                'center': self.center.detach().cpu().clone(),
                'epoch': epoch,
            }
            torch.save(ckpt, self.out_dir / f'model_{epoch}.pth')

            scores, _ = self.infer()
            auc_now = float(roc_auc_score(self.test_y, scores))
            th_now, p_now, r_now = self.select_threshold(self.test_y, scores)
            pred = (scores > th_now).astype(int)
            tp = int(((pred == 1) & (self.test_y == 1)).sum())
            fp = int(((pred == 1) & (self.test_y == 0)).sum())
            fn = int(((pred == 0) & (self.test_y == 1)).sum())
            f1_now = 2 * p_now * r_now / (p_now + r_now + 1e-9)
            f1_run = max(f1_run, f1_now)
            auc_run = max(auc_run, auc_now)

            epoch_rows.append({
                'epoch': epoch,
                'tpr': r_now,
                'ppv': p_now,
                'f1': f1_now,
                'auc': auc_now,
                'th': th_now,
                'min': float(scores.min()),
                'max': float(scores.max()),
                'tp': tp,
                'fp': fp,
                'fn': fn,
                'f1_best': f1_run,
                'auc_best': auc_run,
                'recon': float(np.mean(recon_vals)),
                'score': float(np.mean(score_vals)),
                'align': float(np.mean(align_vals)),
                'adv': float(np.mean(adv_vals)),
                'total': float(np.mean(total_vals)),
            })
            print(f'epoch {epoch}: f1={f1_now:.6f}, auc={auc_now:.6f}, th={th_now:.6f}')

        loss_df = pd.DataFrame(loss_rows)
        epoch_df = pd.DataFrame(epoch_rows)
        loss_df.to_csv(self.out_dir / 'loss.csv', index=False)
        epoch_df.to_csv(self.out_dir / 'epoch.csv', index=False)
        return epoch_df

    def finalize(self, epoch_df: pd.DataFrame):
        idx = epoch_df['f1'].idxmax()
        row = epoch_df.loc[idx]
        epoch_best = int(row['epoch'])
        th_best = float(row['th'])
        ckpt = torch.load(self.out_dir / f'model_{epoch_best}.pth', map_location=self.device)
        self.enc.load_state_dict(ckpt['enc'])
        self.dec.load_state_dict(ckpt['dec'])
        self.disc.load_state_dict(ckpt['disc'])
        self.center.data = ckpt['center'].to(self.device)

        scores, latent = self.infer()
        pred = (scores > th_best).astype(int)
        tp = int(((pred == 1) & (self.test_y == 1)).sum())
        fp = int(((pred == 1) & (self.test_y == 0)).sum())
        fn = int(((pred == 0) & (self.test_y == 1)).sum())
        ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * ppv * tpr / (ppv + tpr + 1e-9)
        auc_val = float(roc_auc_score(self.test_y, scores))
        fpr, tpr_curve, _ = roc_curve(self.test_y, scores, pos_label=1)

        pd.DataFrame({'fpr': fpr, 'tpr': tpr_curve, 'auc': auc_val}).to_csv(self.out_dir / 'roc.csv', index=False)
        pd.DataFrame([{
            'epoch': epoch_best,
            'tpr': tpr,
            'ppv': ppv,
            'f1': f1,
            'auc': auc_val,
            'th': th_best,
            'min': float(scores.min()),
            'max': float(scores.max()),
        }]).to_csv(self.out_dir / 'result.csv', index=False)

        with open(self.out_dir / 'summary.csv', 'w', encoding='utf-8') as f:
            f.write('item,value\n')
            f.write(f'epoch,{epoch_best}\n')
            f.write(f'tpr,{tpr:.6f}\n')
            f.write(f'ppv,{ppv:.6f}\n')
            f.write(f'f1,{f1:.6f}\n')
            f.write(f'auc,{auc_val:.6f}\n')
            f.write(f'th,{th_best:.6f}\n')
            f.write(f'min,{float(scores.min()):.6f}\n')
            f.write(f'max,{float(scores.max()):.6f}\n')

        self.plot_latent(latent)
        self.plot_loss(pd.read_csv(self.out_dir / 'loss.csv'))
        self.plot_f1(epoch_df)
        self.plot_scores(scores, th_best)
        self.plot_roc(fpr, tpr_curve, auc_val)

        info = {
            'device': str(self.device),
            'train_shape': list(self.train_x.shape),
            'test_shape': list(self.test_x.shape),
            'test_pos': int((self.test_y == 1).sum()),
        }
        with open(self.out_dir / 'info.json', 'w', encoding='utf-8') as f:
            json.dump(info, f, indent=2, ensure_ascii=False)

        print(f'best epoch: {epoch_best}')
        print({'tpr': tpr, 'ppv': ppv, 'f1': f1, 'auc': auc_val, 'th': th_best})

    def plot_latent(self, latent):
        fig = plt.figure(figsize=(10, 6))
        plt.scatter(latent[self.test_y == 0, 0], latent[self.test_y == 0, 1], label='Normal', alpha=0.5)
        plt.scatter(latent[self.test_y == 1, 0], latent[self.test_y == 1, 1], label='Anomaly', alpha=0.5, marker='x')
        plt.scatter(self.center[0].item(), self.center[1].item(), c='red', s=100, label='Center')
        plt.legend()
        plt.title('Latent')
        self.save_plot(fig, 'latent.png')
        plt.close(fig)

    def plot_loss(self, loss_df: pd.DataFrame):
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(loss_df['step'], loss_df['recon'], label='recon')
        ax.plot(loss_df['step'], loss_df['score'], label='score')
        ax.plot(loss_df['step'], loss_df['align'], label='align')
        ax.plot(loss_df['step'], loss_df['adv'], label='adv')
        ax.set_title('Loss')
        ax.set_xlabel('Step')
        ax.set_ylabel('Value')
        ax.legend()
        self.save_plot(fig, 'loss.png')
        plt.close(fig)

    def plot_f1(self, epoch_df: pd.DataFrame):
        fig = plt.figure(figsize=(10, 6))
        plt.plot(epoch_df['epoch'], epoch_df['f1'], marker='o', linewidth=2)
        plt.title('F1')
        plt.xlabel('Epoch')
        plt.ylabel('Value')
        plt.grid(True, alpha=0.3)
        self.save_plot(fig, 'f1.png')
        plt.close(fig)

    def plot_scores(self, scores, threshold):
        fig = plt.figure(figsize=(16, 6))
        sc = plt.scatter(
            x=np.arange(len(self.test_y)),
            y=scores,
            c=self.test_y,
            cmap='coolwarm',
            vmin=0,
            vmax=1,
            alpha=0.6,
            edgecolors='w',
            linewidths=0.5,
        )
        plt.colorbar(sc, label='Label')
        plt.axhline(y=threshold, color='r', linestyle='--', label='Threshold')
        plt.xlabel('Index')
        plt.ylabel('Score')
        plt.title('Scores')
        plt.legend()
        self.save_plot(fig, 'scores.png')
        plt.close(fig)

    def plot_roc(self, fpr, tpr, auc_val):
        fig = plt.figure(figsize=(6, 6))
        plt.plot(fpr, tpr, color='magenta', marker='o', linewidth=2, label=f'ROC ({auc_val:.2f})')
        plt.plot([0, 1], [0, 1], color='orange', linestyle='--', label='diag')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.xticks(np.arange(0, 1.1, 0.2))
        plt.yticks(np.arange(0, 1.1, 0.2))
        plt.xlabel('FPR')
        plt.ylabel('TPR')
        plt.title('ROC')
        plt.legend(loc='lower right')
        self.save_plot(fig, 'roc.png')
        plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--vin', type=str, default='099')
    parser.add_argument('--data-root', type=str, default='./data')
    parser.add_argument('--output-root', type=str, default='./outputs')
    parser.add_argument('--exp-name', type=str, default='adv_svdd')
    parser.add_argument('--file-tag', type=str, default='2')
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--latent-dim', type=int, default=6)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--lr-gen', type=float, default=1e-4)
    parser.add_argument('--lr-disc', type=float, default=1e-4)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    return Config(
        vin=args.vin,
        data_root=args.data_root,
        output_root=args.output_root,
        exp_name=args.exp_name,
        file_tag=args.file_tag,
        batch_size=args.batch_size,
        latent_dim=args.latent_dim,
        epochs=args.epochs,
        lr_gen=args.lr_gen,
        lr_disc=args.lr_disc,
        seed=args.seed,
    )


def main():
    cfg = parse_args()
    runner = Runner(cfg)
    with open(runner.out_dir / 'config.json', 'w', encoding='utf-8') as f:
        json.dump(asdict(cfg), f, indent=2, ensure_ascii=False)
    epoch_df = runner.train()
    runner.finalize(epoch_df)


if __name__ == '__main__':
    main()

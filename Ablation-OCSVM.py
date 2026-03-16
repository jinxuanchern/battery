import argparse
import json
import os
from dataclasses import asdict, dataclass

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.metrics import auc, precision_recall_curve, roc_auc_score, roc_curve
from sklearn.svm import OneClassSVM

matplotlib.rcParams['font.family'] = 'Times New Roman'


@dataclass
class Config:
    vin: str = '293'
    data_root: str = './data'
    output_root: str = './outputs'
    exp_name: str = 'ocsvm'
    train_name: str = 'train_features(2).csv'
    test_name: str = 'test_features(2).csv'
    feature_cols: tuple = ('PC1', 'PC2', 'PC3', 'PC4', 'PC5', 'PC6')
    label_col: str = 'label'
    seed: int = 42
    kernel: str = 'rbf'
    nu: float = 0.05
    gamma: str = 'scale'
    use_tsne: bool = False

    @property
    def save_dir(self) -> str:
        return os.path.join(self.output_root, self.vin, self.exp_name)

    @property
    def train_csv(self) -> str:
        return os.path.join(self.data_root, self.vin, self.train_name)

    @property
    def test_csv(self) -> str:
        return os.path.join(self.data_root, self.vin, self.test_name)


def parse_args() -> Config:
    parser = argparse.ArgumentParser()
    parser.add_argument('--vin', type=str, default='293')
    parser.add_argument('--data-root', type=str, default='./data')
    parser.add_argument('--output-root', type=str, default='./outputs')
    parser.add_argument('--exp-name', type=str, default='ocsvm')
    parser.add_argument('--train-name', type=str, default='train_features(2).csv')
    parser.add_argument('--test-name', type=str, default='test_features(2).csv')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--kernel', type=str, default='rbf')
    parser.add_argument('--nu', type=float, default=0.05)
    parser.add_argument('--gamma', type=str, default='scale')
    parser.add_argument('--use-tsne', action='store_true')
    args = parser.parse_args()
    return Config(
        vin=args.vin,
        data_root=args.data_root,
        output_root=args.output_root,
        exp_name=args.exp_name,
        train_name=args.train_name,
        test_name=args.test_name,
        seed=args.seed,
        kernel=args.kernel,
        nu=args.nu,
        gamma=args.gamma,
        use_tsne=args.use_tsne,
    )


def set_seed(seed: int) -> None:
    np.random.seed(seed)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_plot(fig, path: str) -> None:
    fig.savefig(path, bbox_inches='tight', dpi=300)
    print(f'Saved: {path}')


def save_json(obj: dict, path: str) -> None:
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def select_threshold(y_true: np.ndarray, scores: np.ndarray):
    precision, recall, thresholds = precision_recall_curve(y_true, scores, pos_label=1)
    if len(thresholds) == 0:
        return 0.0, 0.0, 0.0, 0, 0, 0

    f1_best = -1.0
    t_best = float(thresholds[0])
    p_best = 0.0
    r_best = 0.0
    tp_best = 0
    fp_best = 0
    fn_best = 0

    for t in thresholds:
        pred = (scores > t).astype(int)
        tp = int(((pred == 1) & (y_true == 1)).sum())
        fn = int(((pred == 0) & (y_true == 1)).sum())
        fp = int(((pred == 1) & (y_true == 0)).sum())
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        if f1 > f1_best:
            f1_best = f1
            t_best = float(t)
            p_best = p
            r_best = r
            tp_best = tp
            fp_best = fp
            fn_best = fn

    return f1_best, t_best, p_best, r_best, tp_best, fp_best, fn_best


def load_data(cfg: Config):
    train_df = pd.read_csv(cfg.train_csv)
    test_df = pd.read_csv(cfg.test_csv)
    x_train = train_df[list(cfg.feature_cols)].values
    x_test = test_df[list(cfg.feature_cols)].values
    y_test = test_df[cfg.label_col].values
    return x_train, x_test, y_test


def plot_latent(cfg: Config, x_test: np.ndarray, y_test: np.ndarray) -> None:
    if cfg.use_tsne:
        emb = TSNE(n_components=2, random_state=cfg.seed, init='pca', learning_rate='auto').fit_transform(x_test)
    else:
        emb = x_test[:, :2]

    fig = plt.figure(figsize=(10, 6))
    plt.scatter(emb[y_test == 0, 0], emb[y_test == 0, 1], label='Normal', alpha=0.5)
    plt.scatter(emb[y_test == 1, 0], emb[y_test == 1, 1], label='Anomaly', alpha=0.5, marker='x')
    plt.legend()
    plt.title('Feature Space')
    save_plot(fig, os.path.join(cfg.save_dir, 'latent.png'))
    plt.close(fig)


def plot_loss(cfg: Config) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis('off')
    ax.text(0.5, 0.58, 'OCSVM has no iterative loss curve.', ha='center', va='center', fontsize=16)
    ax.text(0.5, 0.40, f'kernel={cfg.kernel}, nu={cfg.nu}, gamma={cfg.gamma}', ha='center', va='center', fontsize=13)
    ax.set_title('Loss')
    save_plot(fig, os.path.join(cfg.save_dir, 'loss.png'))
    plt.close(fig)


def plot_f1(cfg: Config, df: pd.DataFrame) -> None:
    fig = plt.figure(figsize=(10, 6))
    plt.plot(df['epoch'], df['F1'], marker='o', linewidth=2)
    plt.title('F1')
    plt.xlabel('Index')
    plt.ylabel('F1')
    plt.ylim(0.0, 1.0)
    plt.grid(True, alpha=0.3)
    save_plot(fig, os.path.join(cfg.save_dir, 'f1.png'))
    plt.close(fig)


def plot_scores(cfg: Config, y_test: np.ndarray, scores: np.ndarray, threshold: float) -> None:
    fig = plt.figure(figsize=(16, 6))
    sc = plt.scatter(
        x=np.arange(len(y_test)),
        y=scores,
        c=y_test,
        cmap='coolwarm',
        vmin=0,
        vmax=1,
        alpha=0.6,
        edgecolors='w',
        linewidths=0.5,
    )
    plt.colorbar(sc, label='True Label (0=Normal, 1=Anomaly)')
    plt.axhline(y=threshold, color='r', linestyle='--', label='Threshold')
    plt.xlabel('Sample Index')
    plt.ylabel('Score')
    plt.title('Scores')
    plt.legend()
    save_plot(fig, os.path.join(cfg.save_dir, 'scores.png'))
    plt.close(fig)


def plot_roc(cfg: Config, y_test: np.ndarray, scores: np.ndarray):
    fpr, tpr, _ = roc_curve(y_test, scores, pos_label=1)
    auc_val = auc(fpr, tpr)
    fig = plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, color='magenta', marker='o', linewidth=2, label=f'ROC (AUC = {auc_val:.2f})')
    plt.plot([0, 1], [0, 1], color='orange', linestyle='--', label='y = x')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xticks(np.arange(0, 1.1, 0.2))
    plt.yticks(np.arange(0, 1.1, 0.2))
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC')
    plt.legend(loc='lower right')
    save_plot(fig, os.path.join(cfg.save_dir, 'roc.png'))
    plt.close(fig)
    return fpr, tpr, auc_val


def main() -> None:
    cfg = parse_args()
    set_seed(cfg.seed)
    ensure_dir(cfg.save_dir)
    save_json(asdict(cfg), os.path.join(cfg.save_dir, 'config.json'))

    x_train, x_test, y_test = load_data(cfg)
    print(f'Train shape: {x_train.shape}')
    print(f'Test shape : {x_test.shape}')
    print(f'Test positives: {int((y_test == 1).sum())}')

    model = OneClassSVM(kernel=cfg.kernel, nu=cfg.nu, gamma=cfg.gamma)
    model.fit(x_train)

    scores = -model.decision_function(x_test).ravel()
    f1_val, threshold, ppv, tpr_val, tp, fp, fn = select_threshold(y_test, scores)
    auc_val = roc_auc_score(y_test, scores)

    epoch_df = pd.DataFrame([
        {
            'epoch': 1,
            'TPR': tpr_val,
            'PPV': ppv,
            'F1': f1_val,
            'AUC': auc_val,
            'threshold': threshold,
            'score_min': float(scores.min()),
            'score_max': float(scores.max()),
        }
    ])
    epoch_df.to_csv(os.path.join(cfg.save_dir, 'epoch.csv'), index=False)
    epoch_df.to_csv(os.path.join(cfg.save_dir, 'summary.csv'), index=False)

    fpr, tpr_curve, roc_val = plot_roc(cfg, y_test, scores)
    roc_df = pd.DataFrame({'fpr': fpr, 'tpr': tpr_curve})
    roc_df['auc'] = roc_val
    roc_df.to_csv(os.path.join(cfg.save_dir, 'roc.csv'), index=False)

    plot_latent(cfg, x_test, y_test)
    plot_loss(cfg)
    plot_f1(cfg, epoch_df)
    plot_scores(cfg, y_test, scores, threshold)

    result_df = pd.DataFrame([
        {
            'tag': 'ocsvm',
            'AUC': auc_val,
            'F1': f1_val,
            'P': ppv,
            'R': tpr_val,
            'threshold': threshold,
            'score_min': float(scores.min()),
            'score_max': float(scores.max()),
        }
    ])
    result_df.to_csv(os.path.join(cfg.save_dir, 'result.csv'), index=False)

    info = {
        'kernel': cfg.kernel,
        'nu': cfg.nu,
        'gamma': cfg.gamma,
        'n_support_vectors': int(len(model.support_)),
        'TPR': tpr_val,
        'PPV': ppv,
        'F1': f1_val,
        'AUC': auc_val,
        'threshold': threshold,
        'TP': tp,
        'FP': fp,
        'FN': fn,
    }
    save_json(info, os.path.join(cfg.save_dir, 'info.json'))

    print('\nOCSVM finished.')
    print(result_df.iloc[0].to_dict())


if __name__ == '__main__':
    main()

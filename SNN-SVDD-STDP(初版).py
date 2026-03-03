import os, numpy as np, pandas as pd, torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import precision_recall_curve, roc_auc_score, roc_curve, auc
from tqdm import tqdm
import matplotlib
import matplotlib.pyplot as plt
from math import pi, gamma


SAVE_DIR = r"E:\桌面\database\chart\Deep-SVDD-多表格-结果\099\离线\STDP"
os.makedirs(SAVE_DIR, exist_ok=True)

BATCH   = 1024
LATENT  = 6
EPOCHS  = 20
SEED    = 42
DEVICE  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
matplotlib.rcParams['font.family'] = 'Arial'
torch.manual_seed(SEED);  np.random.seed(SEED);  torch.cuda.manual_seed_all(SEED)


class AnomDataset(Dataset):
    def __init__(self, df: pd.DataFrame):
        self.x = torch.tensor(df[['PC1','PC2']].values, dtype=torch.float32)
        self.t = torch.tensor(df['TIME'].values,              dtype=torch.float32)
        self.y = torch.tensor(df['label'].values,             dtype=torch.int8)
    def __len__(self):  return len(self.x)
    def __getitem__(self, i):  return self.x[i], self.t[i], self.y[i]

train_df = pd.read_csv(r"E:\桌面\database\chart\Deep-SVDD-多表格-结果\模型数据\099\train_features(3).csv")
val_df   = pd.read_csv(r"E:\桌面\database\chart\Deep-SVDD-多表格-结果\模型数据\099\test_features(3).csv")
train_loader = DataLoader(AnomDataset(train_df), batch_size=BATCH, shuffle=True,  drop_last=True)
val_loader   = DataLoader(AnomDataset(val_df  ), batch_size=BATCH, shuffle=False, drop_last=False)

print('Device:', DEVICE)

class SurrogateSpike(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, thresh):
        ctx.save_for_backward(x);  return (x >= thresh).float()
    @staticmethod
    def backward(ctx, g):
        (x,) = ctx.saved_tensors
        return g * (1.0 / (1.0 + torch.abs(x))**2), None

sur_spike = SurrogateSpike.apply

class SpikingLinear(nn.Module):
    def __init__(self, inp, out, thresh=0.1):  # Ok,这里阈值我还不确定是什么意思
        super().__init__(); self.fc = nn.Linear(inp, out); self.th=thresh
    def forward(self, x):
        mem = self.fc(x)                          # 连续膜电位
        spk = sur_spike(mem, self.th)             # 二值脉冲(供 STDP)
        return torch.sigmoid(mem), spk            # 连续向量给后续

class SNNEncoder(nn.Module):
    def __init__(self, in_dim, d_lat):
        super().__init__()
        self.l1 = SpikingLinear(in_dim, 512)
        self.l2 = SpikingLinear(512, 256)
        self.l3 = SpikingLinear(256, d_lat)
    def forward(self, x):
        x,_ = self.l1(x); x,_ = self.l2(x); z,s = self.l3(x)
        return z, s

class Decoder(nn.Module):
    def __init__(self, d_lat, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_lat,256), nn.LeakyReLU(0.2),
            nn.Linear(256,256),   nn.LeakyReLU(0.2),
            nn.Linear(256,512),   nn.LeakyReLU(0.2),
            nn.Linear(512,out_dim)            # 无 Sigmoid → 支持负值
        )
    def forward(self,z): return self.net(z)

# =============== 损失 & 正则 ===============
recon_fn = nn.MSELoss()
def euclid(z,c): return torch.norm(z - c.unsqueeze(0), dim=1)

def diffusion_reg(z, k=20, D=1.0, sig=1.0):
    # pair-wise 距离
    dist = torch.cdist(z, z)                              # (B,B)
    knn_d, knn_idx = dist.topk(k=k, largest=False)        # ← 索引也返回
    # 邻接权重 & 密度
    A   = torch.exp(-knn_d**2 / (2*sig**2))               # (B,k)
    rho = A.sum(1)                                        # (B,)
    # Laplacian 近似：∑_j A_ij (ρ_j − ρ_i)
    lap = (A * (rho[knn_idx] - rho.unsqueeze(1))).sum(1)  # (B,)
    return D * (lap**2).mean()


def vol_comp(z,c,beta=0.01,sig=1.0):
    d   = torch.norm(z - c.unsqueeze(0), dim=1); rho = torch.exp(-d**2/(2*sig**2))
    R   = d.max().clamp_min(1e-6); V = (pi**(LATENT/2)/gamma(LATENT/2+1))*R**LATENT
    vol = (d**3 * rho).sum()/rho.sum().clamp_min(1e-6)
    return beta*vol/V

# =============== 网络 & 优化器 ===============
enc = SNNEncoder(2,LATENT).to(DEVICE)
dec = Decoder(LATENT,2).to(DEVICE)
center_c = nn.Parameter(torch.randn(LATENT, device=DEVICE)*0.1)

# 动态加权-不确定性 (三个主损失各一 log_sigma)
log_sig_r = nn.Parameter(torch.zeros(1, device=DEVICE))
log_sig_s = nn.Parameter(torch.zeros(1, device=DEVICE))
log_sig_e = nn.Parameter(torch.zeros(1, device=DEVICE))

opt = optim.Adam(list(enc.parameters())+list(dec.parameters())+
                 [center_c, log_sig_r, log_sig_s, log_sig_e], lr=1e-4)

# =============== EMA 辅助函数 ===============
def ema_center():
    mu_all=[]
    with torch.no_grad():
        for (x,_,_) in train_loader:
            mu_all.append(enc(x.to(DEVICE))[0])
    return torch.cat(mu_all).mean(0)

# =============== 训练 ======================
loss_log=[]; metr_log=[]
for ep in range(1,EPOCHS+1):
    enc.train(); dec.train()
    pbar=tqdm(train_loader, desc=f'Epoch {ep}/{EPOCHS}')
    for x,t,_ in pbar:
        x=x.to(DEVICE); z,_=enc(x); x_hat=dec(z)
        recon = recon_fn(x_hat,x)
        svdd  = euclid(z, center_c).mean()
        z_hat,_ = enc(x_hat.detach()); enc_cnst = recon_fn(z_hat, z.detach())
        diff  = diffusion_reg(z); vol = vol_comp(z, center_c)

        # 动态加权 (Kendall & Gal, 2018)
        loss = (torch.exp(-2*log_sig_r)*recon + log_sig_r) + \
               (torch.exp(-2*log_sig_s)*svdd  + log_sig_s) + \
               (torch.exp(-2*log_sig_e)*enc_cnst + log_sig_e) + \
               0.01*diff + 0.01*vol

        opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(enc.parameters(),1.0); opt.step()

        loss_log.append([recon.item(), svdd.item(), enc_cnst.item()])
        pbar.set_postfix(recon=f'{recon:.3f}', svdd=f'{svdd:.3f}')

    # EMA 更新球心
    with torch.no_grad():
        center_c.data.mul_(0.99).add_(0.01*ema_center())

    # ============ 验证 ============
    enc.eval(); scr=[]; lab=[]
    with torch.no_grad():
        for x,_,y in val_loader:
            z,_=enc(x.to(DEVICE)); scr+=euclid(z,center_c).cpu().tolist(); lab+=y.tolist()
    scr=np.array(scr); lab=np.array(lab)
    prec,rec,thr=precision_recall_curve(lab,scr); f1=2*prec*rec/(prec+rec+1e-9)
    idx=np.nanargmax(f1); best_th=thr[max(idx-1,0)] if idx==len(thr) else thr[idx]
    pred=(scr>best_th).astype(int)
    TP=((pred==1)&(lab==1)).sum(); FN=((pred==0)&(lab==1)).sum(); FP=((pred==1)&(lab==0)).sum()
    R=TP/(TP+FN+1e-9); P=TP/(TP+FP+1e-9); F1=2*R*P/(R+P+1e-9); AUC=roc_auc_score(lab,scr)
    metr_log.append([ep,R,P,F1,AUC]); print(f'  >> Val: TPR={R:.3f}, PPV={P:.3f}, F1={F1:.3f}, AUC={AUC:.3f}')

# ================================================================
#               -----------  可视化 -----------
# ================================================================
# 1) 潜在空间散点
latent_vec=[]; test_scores=[]
enc.eval()
with torch.no_grad():
    for x,_,y in val_loader:
        z,_=enc(x.to(DEVICE)); latent_vec.append(z.cpu().numpy())
        test_scores.extend(euclid(z,center_c).cpu().numpy())
latent_vec=np.concatenate(latent_vec); test_scores=np.array(test_scores)
test_labels=val_df['label'].values

plt.figure(figsize=(10,6))
plt.scatter(latent_vec[test_labels==0,0], latent_vec[test_labels==0,1], alpha=0.5, label='Normal')
plt.scatter(latent_vec[test_labels==1,0], latent_vec[test_labels==1,1], alpha=0.5, marker='x', label='Anomaly')
plt.scatter(center_c[0].item(),center_c[1].item(),c='red',s=100,label='Center')
plt.title('Latent Space Distribution'); plt.legend(); plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR,'latent_space.png')); plt.close()

# 2) 训练过程损失
loss_arr=np.array(loss_log)
plt.figure(figsize=(10,6))
plt.plot(loss_arr[:,0],label='Recon'); plt.plot(loss_arr[:,1],label='SVDD'); plt.plot(loss_arr[:,2],label='Enc')
plt.title('Loss Trend'); plt.xlabel('Batch'); plt.ylabel('Loss'); plt.legend(); plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR,'loss_trend.png')); plt.close()

# 3) Epoch 级指标
metr_arr=np.array(metr_log)
plt.figure(figsize=(10,6))
plt.plot(metr_arr[:,0],metr_arr[:,3],marker='o',label='F1'); plt.plot(metr_arr[:,0],metr_arr[:,4],marker='o',label='AUC')
plt.ylim(0,1.05); plt.xlabel('Epoch'); plt.ylabel('Metric'); plt.title('Metric over Epoch'); plt.legend(); plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR,'metric_epoch.png')); plt.close()

# 4) Anomaly Score 分布
plt.figure(figsize=(16,6))
sc=plt.scatter(np.arange(len(test_labels)), test_scores, c=test_labels, cmap='coolwarm', alpha=0.6, edgecolors='w', linewidths=0.5)
plt.colorbar(sc,label='True Label'); plt.axhline(best_th,c='r',ls='--',label='Threshold')
plt.xlabel('Sample'); plt.ylabel('Score'); plt.title('Anomaly Score'); plt.legend(); plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR,'anomaly_score.png')); plt.close()

# 5) ROC
fpr,tpr,_=roc_curve(test_labels,test_scores); roc_auc=auc(fpr,tpr)
plt.figure(figsize=(6,6))
plt.plot(fpr,tpr,lw=2,label=f'ROC (AUC={roc_auc:.2f})'); plt.plot([0,1],[0,1],'--',c='grey')
plt.xlim(0,1); plt.ylim(0,1); plt.xlabel('FPR'); plt.ylabel('TPR'); plt.title('ROC'); plt.legend(); plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR,'ROC.png')); plt.close()

# 6) STDP Learning Window
dt=torch.linspace(-50,50,500); dw=torch.exp(-torch.abs(dt)/20.0)
dw[dt<0]*=-1                # 简易双指数窗
plt.figure(figsize=(7,4)); plt.plot(dt.numpy(),dw.numpy(),lw=2)
plt.axhline(0,c='k',ls='--'); plt.xlabel('Δt (ms)'); plt.ylabel('Δw'); plt.title('STDP Learning Window'); plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR,'stdp_window.png')); plt.close()

# 7) Spike Raster (末层脉冲)
(x_b,_,_) = next(iter(val_loader))
_, spk_b = enc.l3(enc.l2(enc.l1(x_b.to(DEVICE))[0])[0])

# ★ 修正：先 detach
spk_b = spk_b.detach().cpu().numpy()

plt.figure(figsize=(8,4))
for n in range(min(64, spk_b.shape[0])):
    ts = np.where(spk_b[n] > 0.5)[0]
    plt.vlines(ts, n+0.5, n+1.5, color='k')
plt.xlabel('Neuron'); plt.ylabel('Sample'); plt.title('Spike Raster'); plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, 'spike_raster.png'))
plt.close()


print('All figures saved to:', SAVE_DIR)

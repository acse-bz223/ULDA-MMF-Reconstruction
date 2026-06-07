#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_continual_mixda_labelsup_plus_ablate.py

Supports:
- Residual Align layer (W ~ I) + optional bias, with orthogonality regularization (optional).
- Class-conditional bias (small per-class shift).
- Global and/or classwise CORAL (optional).
- Linear scheduling of selected losses within each domain.
- Uncertainty-map consistency with three teacher modes:
    * id: same-sample teacher from source domain (requires matching IDs)
    * classproto: decoded prototypes per class (no ID requirement)
    * ema: mean-teacher (EMA of align/bias), no ID requirement
- Domain selection:
    * --train_domains "1-9,11"   # domains to train/adapt sequentially
    * --zs_domains "10"           # domains to evaluate zero-shot (no training)

Assumes VAE+Head are pretrained on bending0 (256x256), then frozen.
"""

import os, math, csv, json, argparse, random
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Headless plotting safety
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ------------------------- utils -------------------------

def log(msg: str, *, flush=True):
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=flush)

def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True

def label_to_digit(labels: np.ndarray, per_digit_hint: int) -> np.ndarray:
    z = labels.astype(np.int64) - 1
    return (z % (per_digit_hint * 10)) // per_digit_hint

def to_tensor_resized_X(batch_np: np.ndarray, size: int, device) -> torch.Tensor:
    t = torch.from_numpy(batch_np.astype(np.float32, copy=True))
    if t.ndim == 2:   t = t.unsqueeze(0).unsqueeze(0)
    elif t.ndim == 3: t = t.unsqueeze(1)
    elif t.ndim == 4: pass
    else: raise ValueError(f"speckle ndim={t.ndim}")
    B = t.shape[0]
    t2 = t.view(B, -1)
    t2 = (t2 - t2.min(dim=1, keepdim=True).values) / (t2.max(dim=1, keepdim=True).values - t2.min(dim=1, keepdim=True).values + 1e-8)
    t = t2.view(B, 1, t.shape[-2], t.shape[-1])
    if t.shape[-2] != size or t.shape[-1] != size:
        t = F.interpolate(t, size=(size, size), mode='bilinear', align_corners=False)
    return t.to(device, non_blocking=True)

def to_tensor_resized_Y(batch_np: np.ndarray, size: int, device) -> torch.Tensor:
    t = torch.from_numpy(batch_np.astype(np.float32, copy=True))
    if t.ndim == 2:   t = t.unsqueeze(0).unsqueeze(0)
    elif t.ndim == 3: t = t.unsqueeze(1)
    elif t.ndim == 4: pass
    else: raise ValueError(f"gt ndim={t.ndim}")
    if t.shape[-2] != size or t.shape[-1] != size:
        t = F.interpolate(t, size=(size, size), mode='bilinear', align_corners=False)
    return t.to(device, non_blocking=True)

def psnr_from_mse(mse: torch.Tensor) -> torch.Tensor:
    return 10.0 * torch.log10(1.0 / mse.clamp_min(1e-12))

@torch.no_grad()
def ssim_2d(x: torch.Tensor, y: torch.Tensor, sigma=1.5, r=11) -> torch.Tensor:
    pad = r // 2
    xs = torch.arange(r, device=x.device, dtype=x.dtype) - pad
    g  = torch.exp(-(xs**2)/(2*sigma**2))
    g  = (g/g.sum()).view(1,1,-1,1)
    ker= g @ g.transpose(-2,-1)
    yp = F.pad(y,(pad,pad,pad,pad),'reflect')
    xp = F.pad(x,(pad,pad,pad,pad),'reflect')
    mu1=F.conv2d(xp,ker); mu2=F.conv2d(yp,ker)
    s1 =F.conv2d(xp*xp,ker)-mu1*mu1
    s2 =F.conv2d(yp*yp,ker)-mu2*mu2
    s12=F.conv2d(xp*yp,ker)-mu1*mu2
    C1=x.new_tensor(0.01**2); C2=x.new_tensor(0.03**2)
    s=((2*mu1*mu2+C1)*(2*s12+C2))/((mu1*mu1+mu2*mu2+C1)*(s1+s2+C2))
    return s.flatten(1).mean(1)

def batch_covar(z: torch.Tensor) -> torch.Tensor:
    zc = z - z.mean(0, keepdim=True)
    if zc.size(0) > 1:
        C = (zc.T @ zc) / (zc.size(0) - 1)
    else:
        C = torch.zeros(z.size(1), z.size(1), device=z.device, dtype=z.dtype)
    return (C + C.T) / 2.0

# ---------------------- data helpers ----------------------

def load_bending_arrays(root: Path, bend_id: int):
    p = root / f"bending{bend_id}_sorted"
    X = np.load(p/"speckles_sorted.npy", mmap_mode='r')
    Y = np.load(p/"images_sorted.npy",   mmap_mode='r')
    L = np.load(p/"labels_sorted.npy",   mmap_mode='r')
    return X, Y, L

def make_label2idx(L: np.ndarray) -> Dict[int, int]:
    d = {}
    Llist = L.tolist()
    for i, lab in enumerate(Llist):
        if lab not in d: d[lab] = i
    return d

# ------------------------- model -------------------------

def conv_block(in_ch, out_ch, p_drop):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, 1, 1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.Dropout2d(p_drop),
        nn.MaxPool2d(2, 2)
    )

def deconv_block(in_ch, out_ch, p_drop):
    return nn.Sequential(
        nn.ConvTranspose2d(in_ch, out_ch, 4, 2, 1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.Dropout2d(p_drop)
    )

class SimpleVAEWithUncertainty(nn.Module):
    def __init__(self, input_hw=(256,256), latent_dim=512, dropout_p_enc=0.2, dropout_p_dec=0.1):
        super().__init__()
        H, W = input_hw
        assert H==W==256, "This VAE expects 256x256 inputs."
        self.enc1 = conv_block(1,   32, dropout_p_enc)
        self.enc2 = conv_block(32,  64, dropout_p_enc)
        self.enc3 = conv_block(64, 128, dropout_p_enc)
        self.enc4 = conv_block(128,256, dropout_p_enc)
        with torch.no_grad():
            d = torch.zeros(1,1,H,W)
            h = self._enc(d)
            self.enc_shape = h.shape[1:]
            self.flat_dim  = int(h.view(1,-1).size(1))
        self.fc_mu     = nn.Linear(self.flat_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.flat_dim, latent_dim)
        self.fc_dec    = nn.Linear(latent_dim, self.flat_dim)
        C,_,_ = self.enc_shape
        self.dec1 = deconv_block(C,   128, dropout_p_dec)
        self.dec2 = deconv_block(128, 64,  dropout_p_dec)
        self.dec3 = deconv_block(64,  32,  dropout_p_dec)
        self.dec4 = deconv_block(32,  16,  dropout_p_dec)
        self.out_mu     = nn.Conv2d(16, 1, 3, 1, 1)
        self.out_logvar = nn.Conv2d(16, 1, 3, 1, 1)

    def _enc(self, x):
        x = self.enc1(x); x = self.enc2(x); x = self.enc3(x); x = self.enc4(x)
        return x

    def encode(self, x):
        if x.dim()==3: x = x.unsqueeze(1)
        h = self._enc(x).flatten(1)
        mu = self.fc_mu(h); logvar = self.fc_logvar(h)
        return mu, logvar

    def decode(self, z, out_hw=None):
        h = self.fc_dec(z).view(-1, *self.enc_shape)
        h = self.dec1(h); h = self.dec2(h); h = self.dec3(h); h = self.dec4(h)
        mu = torch.sigmoid(self.out_mu(h))
        logvar = self.out_logvar(h)
        if out_hw is not None and (mu.shape[-2], mu.shape[-1]) != tuple(out_hw):
            mu = F.interpolate(mu, size=out_hw, mode='bilinear', align_corners=False)
            logvar = F.interpolate(logvar, size=out_hw, mode='bilinear', align_corners=False)
        return mu, logvar

class LatentDigitHead(nn.Module):
    def __init__(self, latent_dim=512, p_drop=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.BatchNorm1d(latent_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p_drop),
            nn.Linear(latent_dim, 10)
        )
    def forward(self, mu): return self.net(mu)

class AlignLayer(nn.Module):
    """Residual linear aligner whose effective transform starts at Identity."""
    def __init__(self, latent_dim=512, use_bias=True, residual=True):
        super().__init__()
        self.fc = nn.Linear(latent_dim, latent_dim, bias=use_bias)
        with torch.no_grad():
            if residual:
                nn.init.zeros_(self.fc.weight)
            else:
                nn.init.eye_(self.fc.weight)
            if use_bias: nn.init.zeros_(self.fc.bias)
        self.residual = residual
    def forward(self, z):
        y = self.fc(z)
        return z + y if self.residual else y

class ClassCondBias(nn.Module):
    """Class-conditional bias: small per-class shift in latent."""
    def __init__(self, num_classes=10, dim=512):
        super().__init__()
        self.bias = nn.Embedding(num_classes, dim)
        nn.init.zeros_(self.bias.weight)
    def forward(self, z, y):
        return z + self.bias(y)

# --------------------- losses ---------------------

def supcon_loss(z: torch.Tensor, y: torch.Tensor, T=0.07):
    z = F.normalize(z, dim=1)
    sim = z @ z.t() / T
    N = z.size(0)
    mask = torch.eq(y.unsqueeze(1), y.unsqueeze(0)).float()
    logits = sim - torch.eye(N, device=z.device) * 1e9
    exp = torch.exp(logits)
    log_prob = logits - torch.log(exp.sum(1, keepdim=True))
    pos = mask - torch.eye(N, device=z.device)
    pos_cnt = pos.sum(1).clamp_min(1.0)
    loss = - (pos * log_prob).sum(1) / pos_cnt
    return loss.mean()

def coral_loss(cov_src: torch.Tensor, cov_tgt: torch.Tensor):
    D = cov_src.size(0)
    return ((cov_src - cov_tgt)**2).sum() / (4.0 * D * D)

def orth_reg(align: AlignLayer, lam=1e-3):
    W = align.fc.weight
    I = torch.eye(W.size(0), device=W.device, dtype=W.dtype)
    effective_W = I + W if align.residual else W
    return lam * (effective_W.t() @ effective_W - I).pow(2).mean()

def gaussian_sym_kl(mu1, logv1, mu2, logv2):
    """Symmetric KL between N(mu1,s1^2) and N(mu2,s2^2), averaged over pixels."""
    logv1 = torch.clamp(logv1, -6, 3)
    logv2 = torch.clamp(logv2, -6, 3)
    v1 = torch.exp(logv1)
    v2 = torch.exp(logv2)
    kl12 = 0.5 * ( (v1 + (mu1 - mu2)**2) / v2 - 1.0 + (logv2 - logv1) )
    kl21 = 0.5 * ( (v2 + (mu2 - mu1)**2) / v1 - 1.0 + (logv1 - logv2) )
    skl = (kl12 + kl21) * 0.5
    return skl.flatten(1).mean(1).mean()

# ----------- prototypes / covariance -----------

@torch.no_grad()
def compute_prototypes(vae: nn.Module, X: np.ndarray, L: np.ndarray,
                       per_digit_hint: int, device, size: int,
                       batch: int, amp_on: bool, align: Optional[nn.Module]=None) -> torch.Tensor:
    digits = label_to_digit(L, per_digit_hint).astype(np.int64)
    latent_dim = int(vae.fc_mu.out_features)
    sums = torch.zeros(10, latent_dim, device=device)
    cnt  = torch.zeros(10, device=device)
    for s in range(0, len(L), batch):
        e = min(s+batch, len(L))
        xb = to_tensor_resized_X(X[s:e], size, device)
        with torch.amp.autocast("cuda", enabled=amp_on and device.type=="cuda"):
            mu = vae.encode(xb)[0].float()
        if align is not None:
            mu = align(mu)
        d = torch.from_numpy(digits[s:e]).to(device, non_blocking=True)
        for k in range(10):
            m = (d == k)
            if m.any():
                sums[k] += mu[m].sum(0)
                cnt[k]  += m.sum()
        del xb, mu, d
    return sums / cnt.clamp_min(1.0).unsqueeze(1)

@torch.no_grad()
def compute_covariance_over_X(vae: nn.Module, X: np.ndarray, device, size: int,
                              batch: int, amp_on: bool, align: Optional[nn.Module]=None) -> torch.Tensor:
    latent_dim = int(vae.fc_mu.out_features)
    sum_mu  = torch.zeros(latent_dim, device=device)
    sum_xxt = torch.zeros(latent_dim, latent_dim, device=device)
    N = 0
    for s in range(0, len(X), batch):
        e  = min(s+batch, len(X))
        xb = to_tensor_resized_X(X[s:e], size, device)
        with torch.amp.autocast("cuda", enabled=amp_on and device.type=="cuda"):
            mu = vae.encode(xb)[0].float()
        if align is not None:
            mu = align(mu)
        sum_mu  += mu.sum(0)
        sum_xxt += mu.T @ mu
        N += mu.size(0)
        del xb, mu
    mean = sum_mu / max(1, N)
    cov  = (sum_xxt - N * (mean.unsqueeze(1) @ mean.unsqueeze(0))) / max(1, N-1)
    return (cov + cov.T)/2.0

@torch.no_grad()
def compute_classwise_cov(vae: nn.Module, X: np.ndarray, L: np.ndarray, per_digit_hint: int,
                          device, size: int, batch: int, amp_on: bool,
                          align: Optional[nn.Module]=None) -> List[torch.Tensor]:
    covs = [None]*10
    bufs = [ [] for _ in range(10) ]
    for s in range(0, len(L), batch):
        e = min(s+batch, len(L))
        xb = to_tensor_resized_X(X[s:e], size, device)
        d  = label_to_digit(L[s:e], per_digit_hint).astype(np.int64)
        with torch.amp.autocast("cuda", enabled=amp_on and device.type=="cuda"):
            mu = vae.encode(xb)[0].float()
        if align is not None:
            mu = align(mu)
        mu_cpu = mu.detach().cpu()
        for i in range(mu_cpu.size(0)):
            bufs[int(d[i])].append(mu_cpu[i:i+1])
        del xb, mu
    for k in range(10):
        if len(bufs[k]) >= 2:
            Z = torch.cat(bufs[k], 0).to(device)
            covs[k] = batch_covar(Z).detach()
        else:
            D = int(vae.fc_mu.out_features)
            covs[k] = torch.zeros(D, D, device=device)
    return covs

# ------------------- visualization helpers -------------------

@torch.no_grad()
def _pick_digit_indices(L: np.ndarray, per_digit_hint: int, max_per_digit: int, fallback_k: int = 10):
    digits = label_to_digit(L, per_digit_hint).astype(np.int64)
    sel = []
    for d in range(10):
        ids = np.where(digits == d)[0][:max_per_digit]
        sel.append(ids)
    picked = np.concatenate([s for s in sel if len(s)>0]) if any(len(s)>0 for s in sel) else np.array([], dtype=np.int64)
    if len(picked) == 0:
        picked = np.arange(min(fallback_k, len(L)))
    return picked

@torch.no_grad()
def viz_recon_grid(vae, align, class_bias, X, Y, L, device, size, save_path: Path,
                   amp_on: bool, per_digit_hint: int, rows: int = 10):
    idx = _pick_digit_indices(L, per_digit_hint, max_per_digit=1, fallback_k=rows)[:rows]
    xs = to_tensor_resized_X(X[idx], size, device)
    ys = to_tensor_resized_Y(Y[idx], size, device)
    yb = torch.from_numpy(label_to_digit(L[idx], per_digit_hint).astype(np.int64)).to(device)
    with torch.amp.autocast("cuda", enabled=amp_on and device.type=="cuda"):
        mu = vae.encode(xs)[0].float()
        rec_pre,_ = vae.decode(mu, out_hw=(size,size))
        mu2 = align(mu)
        if class_bias is not None:
            mu2 = class_bias(mu2, yb)
        rec_post,_ = vae.decode(mu2, out_hw=(size,size))
    xs  = xs.float().cpu().numpy()
    rec_pre  = rec_pre.clamp(0,1).float().cpu().numpy()
    rec_post = rec_post.clamp(0,1).float().cpu().numpy()
    ys  = ys.float().cpu().numpy()

    fig, axes = plt.subplots(rows, 4, figsize=(8, 2*rows))
    if rows == 1:
        axes = np.expand_dims(axes, 0)
    for r in range(xs.shape[0]):
        axes[r,0].imshow(xs[r,0], vmin=0, vmax=1);    axes[r,0].set_title("Speckle"); axes[r,0].axis('off')
        axes[r,1].imshow(rec_pre[r,0], vmin=0, vmax=1);  axes[r,1].set_title("Reconstruction (pre-DA)"); axes[r,1].axis('off')
        axes[r,2].imshow(rec_post[r,0], vmin=0, vmax=1); axes[r,2].set_title("Reconstruction (post-DA)"); axes[r,2].axis('off')
        axes[r,3].imshow(ys[r,0], vmin=0, vmax=1);    axes[r,3].set_title("Ground Truth"); axes[r,3].axis('off')
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)

@torch.no_grad()
def viz_tsne_pair(vae, head, align, class_bias, X, L, device, size, out_before: Path, out_after: Path,
                  amp_on: bool, per_digit_hint: int, per_digit: int = 800, batch: int = 256):
    try:
        from sklearn.manifold import TSNE
    except Exception as e:
        log(f"[viz] sklearn not available for t-SNE: {e}. Skipping t-SNE viz.")
        return
    digits = label_to_digit(L, per_digit_hint).astype(np.int64)
    wanted = []
    counters = {d:0 for d in range(10)}
    for i, d in enumerate(digits):
        if counters[int(d)] < per_digit:
            wanted.append(i); counters[int(d)] += 1
        if all(c >= per_digit for c in counters.values()):
            break
    if len(wanted) == 0:
        wanted = list(range(min(1000, len(L))))
    wanted = np.array(wanted)

    Z0 = []
    Z1 = []
    Yd = []
    for s in range(0, len(wanted), batch):
        e = min(s+batch, len(wanted))
        ids = wanted[s:e]
        xb = to_tensor_resized_X(X[ids], size, device)
        yb = torch.from_numpy(digits[ids]).to(device)
        with torch.amp.autocast("cuda", enabled=amp_on and device.type=="cuda"):
            mu = vae.encode(xb)[0].float()
            mu2 = align(mu)
            if class_bias is not None:
                mu2 = class_bias(mu2, yb)
        Z0.append(mu.detach().cpu().numpy())
        Z1.append(mu2.detach().cpu().numpy())
        Yd.append(digits[ids])
    Z0 = np.concatenate(Z0, 0)
    Z1 = np.concatenate(Z1, 0)
    Yd = np.concatenate(Yd, 0)

    tsne0 = TSNE(n_components=2, init='random', learning_rate='auto', perplexity=30)
    E0 = tsne0.fit_transform(Z0)
    tsne1 = TSNE(n_components=2, init='random', learning_rate='auto', perplexity=30)
    E1 = tsne1.fit_transform(Z1)

    def _scatter(E, Y, title, path):
        plt.figure(figsize=(6,5))
        for d in range(10):
            m = (Y==d)
            if m.sum()>0:
                plt.scatter(E[m,0], E[m,1], s=4, label=str(d), alpha=0.7)
        plt.legend(markerscale=3, prop={'size':8})
        plt.title(title)
        plt.tight_layout()
        path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(path, dpi=150)
        plt.close()

    _scatter(E0, Yd, 't-SNE (latent mu, pre-DA)', out_before)
    _scatter(E1, Yd, 't-SNE (aligned mu, post-DA)', out_after)

# ------------------- evaluation -------------------

@torch.no_grad()
def eval_on_arrays(vae, head, align, X, Y, L, device, size, batch=256, amp_on=False, per_digit_hint=800, class_bias: Optional[nn.Module]=None):
    vae.eval(); head.eval(); align.eval()
    if class_bias is not None: class_bias.eval()
    s_mse0=s_psn0=s_ssm0=0.0; s_mse1=s_psn1=s_ssm1=0.0; n=0
    acc0_ok=acc1_ok=0
    digits = label_to_digit(L, per_digit_hint).astype(np.int64)
    for s in range(0, len(L), batch):
        e = min(s+batch, len(L))
        xs = to_tensor_resized_X(X[s:e], size, device)
        ys = to_tensor_resized_Y(Y[s:e], size, device)
        yb = torch.from_numpy(digits[s:e]).to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=amp_on and device.type=="cuda"):
            mu = vae.encode(xs)[0].float()
            rec0,_ = vae.decode(mu, out_hw=(size,size))
            mu2 = align(mu)
            if class_bias is not None:
                mu2 = class_bias(mu2, yb)
            rec1,_ = vae.decode(mu2, out_hw=(size,size))
            logit0 = head(mu)
            logit1 = head(mu2)
        rec0 = rec0.clamp(0,1).float()
        rec1 = rec1.clamp(0,1).float()
        ys   = ys.float()
        mse0 = ((rec0-ys)**2).flatten(1).mean(1); psn0 = psnr_from_mse(mse0); ssm0 = ssim_2d(rec0, ys)
        mse1 = ((rec1-ys)**2).flatten(1).mean(1); psn1 = psnr_from_mse(mse1); ssm1 = ssim_2d(rec1, ys)
        s_mse0 += float(mse0.sum()); s_psn0 += float(psn0.sum()); s_ssm0 += float(ssm0.sum())
        s_mse1 += float(mse1.sum()); s_psn1 += float(psn1.sum()); s_ssm1 += float(ssm1.sum())
        n += (e - s)
        dnp = digits[s:e]
        acc0_ok += int((logit0.argmax(1).detach().cpu().numpy() == dnp).sum())
        acc1_ok += int((logit1.argmax(1).detach().cpu().numpy() == dnp).sum())
        del xs, ys, mu, mu2, rec0, rec1, logit0, logit1, yb
    base = dict(MSE=s_mse0/max(1,n), PSNR=s_psn0/max(1,n), SSIM=s_ssm0/max(1,n), Acc=100.0*acc0_ok/max(1,n), N=n)
    da   = dict(MSE=s_mse1/max(1,n), PSNR=s_psn1/max(1,n), SSIM=s_ssm1/max(1,n), Acc=100.0*acc1_ok/max(1,n), N=n)
    return base, da

# --------------------------- args ---------------------------

def parse_args():
    ap = argparse.ArgumentParser("Continual DA with ablations + domain selection + viz")
    ap.add_argument("--data_root", type=str, required=True)
    ap.add_argument("--vae_ckpt", type=str, required=True)
    ap.add_argument("--initial_domain", type=int, default=0)
    ap.add_argument("--final_domain", type=int, default=10)
    ap.add_argument("--domains", type=str, default="", help="[legacy] Train domains, e.g., '1-3,5'.")
    ap.add_argument("--train_domains", type=str, default="", help="Domains to train/adapt, e.g., '1-9'. Overrides --domains if set.")
    ap.add_argument("--zs_domains", type=str, default="", help="Zero-shot evaluation domains (no training).")
    ap.add_argument("--per_digit_hint", type=int, default=800)
    ap.add_argument("--resize_hw", type=int, default=256)

    ap.add_argument("--epochs_per_domain", type=int, default=30)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--replay_ratio", type=float, default=0.25)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)

    # base loss weights (scheduled)
    ap.add_argument("--w_ce",    type=float, default=1.0)
    ap.add_argument("--w_prot",  type=float, default=2.0)
    ap.add_argument("--w_coral", type=float, default=0.5)

    # switches
    ap.add_argument("--use_classwise_coral", action="store_true")
    ap.add_argument("--use_supcon", action="store_true")
    ap.add_argument("--w_supcon", type=float, default=0.1)

    # uncertainty consistency
    ap.add_argument("--use_uncert_consistency", action="store_true")
    ap.add_argument("--unc_mode", type=str, default="id", choices=["id","classproto","ema"])
    ap.add_argument("--w_unc", type=float, default=0.5)
    ap.add_argument("--w_unc_mu", type=float, default=0.0)
    ap.add_argument("--unc_teacher", type=str, default="source", choices=["source","prev"])
    ap.add_argument("--ema_momentum", type=float, default=0.996)

    # align & regularization
    ap.add_argument("--orth_lambda", type=float, default=1e-3)
    ap.add_argument("--use_class_bias", action="store_true")
    ap.add_argument("--no_align_residual", action="store_true")

    # explicit per-loss OFF toggles (default ON)
    ap.add_argument("--no_ce", action="store_true")
    ap.add_argument("--no_prot", action="store_true")
    ap.add_argument("--no_coral_g", action="store_true")
    ap.add_argument("--no_coral_c", action="store_true")
    ap.add_argument("--no_supcon", action="store_true")
    ap.add_argument("--no_orth", action="store_true")
    ap.add_argument("--no_unc_kl", action="store_true")
    ap.add_argument("--no_unc_mu", action="store_true")

    # viz
    ap.add_argument("--viz_every", type=int, default=10)
    ap.add_argument("--viz_rows",  type=int, default=10)
    ap.add_argument("--tsne_per_digit", type=int, default=800)
    ap.add_argument("--tsne_batch",     type=int, default=256)

    ap.add_argument("--save_dir", type=str, default="runs/continual_labelsup_plus")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--amp", action="store_true")
    return ap.parse_args()

def strip_module(sd):
    return { (k[7:] if k.startswith("module.") else k): v for k,v in sd.items() }

def load_vae_head(vae, head, ckpt_path, strict_vae=False, strict_head=False):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    vae_sd  = ckpt.get("vae",  ckpt)
    head_sd = ckpt.get("head", {})
    m,u = vae.load_state_dict(strip_module(vae_sd), strict=strict_vae)
    log(f"[load] VAE missing={len(m)} unexpected={len(u)}")
    if isinstance(head_sd, dict) and len(head_sd)>0:
        m2,u2 = head.load_state_dict(strip_module(head_sd), strict=strict_head)
        log(f"[load] HEAD missing={len(m2)} unexpected={len(u2)}")
    else:
        log("[load] HEAD not found in ckpt; using current head weights.")
    return vae, head

def _parse_domain_list(dom_str: str, init_d: int, final_d: int):
    dom_str = (dom_str or "").strip()
    if not dom_str:
        return list(range(init_d+1, final_d+1))
    out = []
    for part in dom_str.split(','):
        part = part.strip()
        if not part:
            continue
        if '-' in part:
            a,b = part.split('-',1)
            a = int(a); b = int(b)
            out.extend(list(range(min(a,b), max(a,b)+1)))
        else:
            out.append(int(part))
    # unique in order
    seen = set(); res=[]
    for d in out:
        if d not in seen:
            res.append(d); seen.add(d)
    return res

# --------------------------- main ---------------------------

def main():
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_on = args.amp and (device.type == "cuda")
    SIZE = int(args.resize_hw)

    root = Path(args.data_root)
    X0, Y0, L0 = load_bending_arrays(root, args.initial_domain)
    label2idx_src = make_label2idx(L0)

    # build & load VAE+Head (frozen)
    vae  = SimpleVAEWithUncertainty(input_hw=(SIZE,SIZE), latent_dim=512).to(device).eval()
    head = LatentDigitHead(latent_dim=512, p_drop=0.2).to(device).eval()
    vae, head = load_vae_head(vae, head, args.vae_ckpt, strict_vae=False, strict_head=False)
    for p in vae.parameters():  p.requires_grad_(False)
    for p in head.parameters(): p.requires_grad_(False)

    # aligner + optional class-bias
    align = AlignLayer(latent_dim=512, use_bias=True, residual=(not args.no_align_residual)).to(device).train()
    class_bias = ClassCondBias(num_classes=10, dim=512).to(device).train() if args.use_class_bias else None

    # run dirs
    run_dir = Path(args.save_dir) / datetime.now().strftime("%Y%m%d_%H%M%S")
    (run_dir/"checkpoints").mkdir(parents=True, exist_ok=True)
    (run_dir/"viz").mkdir(parents=True, exist_ok=True)
    hist_csv = run_dir/"history_epoch_losses.csv"
    val_csv  = run_dir/"domain_eval.csv"
    with open(hist_csv, "w", newline="") as f:
        csv.writer(f).writerow(["domain","epoch","CE","PROT","CORAL_G","CORAL_C","SupCon","Orth","UncKL","UncMu","Total"])
    with open(val_csv, "w", newline="") as f:
        csv.writer(f).writerow(["domain","split","MSE","PSNR","SSIM","Acc","N"])

    # source anchors (prototypes, covariances)
    proto_src = compute_prototypes(vae, X0, L0, args.per_digit_hint, device, SIZE, args.tsne_batch, amp_on, align=None).detach()
    cov_src_g = compute_covariance_over_X(vae, X0, device, SIZE, args.tsne_batch, amp_on, align=None).detach()
    cov_src_c = compute_classwise_cov(vae, X0, L0, args.per_digit_hint, device, SIZE, args.tsne_batch, amp_on, align=None) if args.use_classwise_coral and not args.no_coral_c else None

    # pre-training eval on domain 0
    base0, da0 = eval_on_arrays(vae, head, align, X0, Y0, L0, device, SIZE, batch=args.tsne_batch, amp_on=amp_on, per_digit_hint=args.per_digit_hint, class_bias=class_bias)
    with open(val_csv, "a", newline="") as f:
        csv.writer(f).writerow([args.initial_domain, "before", base0["MSE"], base0["PSNR"], base0["SSIM"], base0["Acc"], base0["N"]])
        csv.writer(f).writerow([args.initial_domain, "before_da", da0["MSE"], da0["PSNR"], da0["SSIM"], da0["Acc"], da0["N"]])
    log(f"[bending{args.initial_domain}] pre-train base: MSE={base0['MSE']:.6f} PSNR={base0['PSNR']:.2f} SSIM={base0['SSIM']:.4f} Acc={base0['Acc']:.2f}%")
    log(f"[bending{args.initial_domain}] pre-train DA:   MSE={da0['MSE']:.6f} PSNR={da0['PSNR']:.2f} SSIM={da0['SSIM']:.4f} Acc={da0['Acc']:.2f}%")

    # optimizer
    params = list(align.parameters()) + (list(class_bias.parameters()) if class_bias is not None else [])
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)

    # domain lists
    def parse_list(s): return _parse_domain_list(s, args.initial_domain, args.final_domain)
    train_domains = parse_list(args.train_domains or args.domains)
    zs_domains    = parse_list(args.zs_domains) if args.zs_domains else []
    if zs_domains:
        train_domains = [d for d in train_domains if d not in set(zs_domains)]
    log(f"[domains] train = {train_domains}; zero-shot = {zs_domains}")

    # replay buffers
    src_X_list = [X0]; src_L_list = [L0]
    ema = 0.2  # for anchors update

    # EMA teacher optional
    from copy import deepcopy
    def ema_clone(module: nn.Module):
        m2 = deepcopy(module).eval()
        for p in m2.parameters(): p.requires_grad_(False)
        return m2
    @torch.no_grad()
    def ema_update(target: nn.Module, source: nn.Module, momentum: float = 0.996):
        for (tp, sp) in zip(target.parameters(), source.parameters()):
            tp.data.mul_(momentum).add_(sp.data, alpha=(1.0 - momentum))
        for (tb, sb) in zip(target.buffers(), source.buffers()):
            tb.copy_(sb)

    ema_align = None; ema_class_bias = None
    if args.use_uncert_consistency and args.unc_mode == 'ema':
        ema_align = ema_clone(align).to(device)
        if args.use_class_bias:
            ema_class_bias = ema_clone(class_bias).to(device)
        log(f"[unc] mode=EMA, momentum={args.ema_momentum}")

    # =============== train/adapt over train_domains ===============
    for target_domain in train_domains:
        log(f"\n=== Adapting to bending{target_domain} (label-supervised) ===")
        Xt, Yt, Lt = load_bending_arrays(root, target_domain)

        # eval before training on target
        base, da = eval_on_arrays(vae, head, align, Xt, Yt, Lt, device, SIZE, batch=args.tsne_batch, amp_on=amp_on, per_digit_hint=args.per_digit_hint, class_bias=class_bias)
        with open(val_csv, "a", newline="") as f:
            csv.writer(f).writerow([target_domain, "before", base["MSE"], base["PSNR"], base["SSIM"], base["Acc"], base["N"]])
            csv.writer(f).writerow([target_domain, "before_da", da["MSE"], da["PSNR"], da["SSIM"], da["Acc"], da["N"]])
        log(f"[before] MSE={base['MSE']:.6f} PSNR={base['PSNR']:.2f} SSIM={base['SSIM']:.4f} Acc={base['Acc']:.2f}%")

        viz_dir = run_dir/"viz"/f"bending{target_domain}"
        viz_dir.mkdir(parents=True, exist_ok=True)

        # classproto teacher (decoded class prototypes)
        class_teacher_mu_c = class_teacher_lv_c = None
        if args.use_uncert_consistency and args.unc_mode == 'classproto':
            class_teacher_mu_c, class_teacher_lv_c = build_class_teacher_maps(vae=None, proto_src=None, size=None, device=None, amp_on=None)  # placeholder
            # Actually we will build below after proto_src is available (we do have it):
            class_teacher_mu_c, class_teacher_lv_c = build_class_teacher_maps_from_proto(vae, proto_src, SIZE, device, amp_on)
            log("[unc] mode=CLASSPROTO ready")

        # combine source for replay
        Xsrc = np.concatenate(src_X_list, axis=0)
        Lsrc = np.concatenate(src_L_list, axis=0)
        N_tgt = len(Lt); N_src = len(Lsrc)
        steps_per_epoch = max(1, math.ceil(max(N_tgt, N_src) / args.batch_size))

        # switches effective
        def eff(flag, weight_gt0=True): return flag and weight_gt0
        use_CE       = eff(not args.no_ce,        args.w_ce > 0)
        use_PROT     = eff(not args.no_prot,      args.w_prot > 0)
        use_CORAL_G  = eff(not args.no_coral_g,   args.w_coral > 0)
        use_CORAL_Cf = eff(args.use_classwise_coral and (not args.no_coral_c), args.w_coral > 0)
        use_SUPCON   = eff(args.use_supcon and (not args.no_supcon), args.w_supcon > 0)
        use_ORTH     = eff(not args.no_orth,      args.orth_lambda > 0)
        use_UNC_KL   = eff(args.use_uncert_consistency and (not args.no_unc_kl), args.w_unc > 0)
        use_UNC_MU   = eff(args.use_uncert_consistency and (not args.no_unc_mu), args.w_unc_mu > 0)

        log(f"[loss switches] CE={use_CE} PROT={use_PROT} CORAL_G={use_CORAL_G} CORAL_C={use_CORAL_Cf} SUPCON={use_SUPCON} ORTH={use_ORTH} UNCKL={use_UNC_KL} UNCMU={use_UNC_MU} (unc_mode={args.unc_mode})")

        for ep in range(1, args.epochs_per_domain+1):
            s = ep / float(args.epochs_per_domain)
            lam_ce   = args.w_ce if use_CE else 0.0
            lam_prot = args.w_prot if use_PROT else 0.0
            lam_coral= (args.w_coral * s) if (use_CORAL_G or use_CORAL_Cf) else 0.0
            lam_sup  = (args.w_supcon * max(0.0, (s - 0.3)/0.7)) if use_SUPCON else 0.0
            lam_unc  = (args.w_unc * s) if use_UNC_KL else 0.0
            lam_uncm = (args.w_unc_mu * s) if use_UNC_MU else 0.0

            idx_tgt = np.random.permutation(N_tgt)
            idx_src = np.random.permutation(N_src) if N_src>0 else np.array([], dtype=np.int64)
            src_bs = int(args.batch_size * args.replay_ratio) if N_src>0 else 0
            tgt_bs = max(1, args.batch_size - src_bs)

            sCE=sPR=sCOg=sCOc=sSC=sORTH=sUKL=sUMU=sTOT=0.0; nb=0

            for step in range(steps_per_epoch):
                t_s = step*tgt_bs; t_e = min((step+1)*tgt_bs, N_tgt)
                if t_s >= t_e: break
                tid = idx_tgt[t_s:t_e]
                xb_t = to_tensor_resized_X(Xt[tid], SIZE, device)
                yb_t = torch.from_numpy(label_to_digit(Lt[tid], args.per_digit_hint).astype(np.int64)).to(device, non_blocking=True)
                labid_t = torch.from_numpy(Lt[tid].astype(np.int64))

                xb_s = None; yb_s = None
                if src_bs > 0:
                    s_s = step*src_bs; s_e = min((step+1)*src_bs, N_src)
                    if s_s < s_e:
                        sid = idx_src[s_s:s_e]
                        xb_s = to_tensor_resized_X(Xsrc[sid], SIZE, device)
                        yb_s = torch.from_numpy(label_to_digit(Lsrc[sid], args.per_digit_hint).astype(np.int64)).to(device, non_blocking=True)

                with torch.amp.autocast("cuda", enabled=amp_on):
                    mu_t = vae.encode(xb_t)[0].float()
                    mu_t2= align(mu_t)
                    if class_bias is not None:
                        mu_t2 = class_bias(mu_t2, yb_t)

                    ce = torch.tensor(0.0, device=device)
                    prot = torch.tensor(0.0, device=device)
                    coral_g = torch.tensor(0.0, device=device)
                    coral_c = torch.tensor(0.0, device=device)
                    sc = torch.tensor(0.0, device=device)
                    ukl = torch.tensor(0.0, device=device)
                    umu = torch.tensor(0.0, device=device)
                    oreg = torch.tensor(0.0, device=device)

                    if use_CE:
                        logits_t = head(mu_t2)
                        ce = F.cross_entropy(logits_t, yb_t)

                    if use_PROT:
                        prot = F.mse_loss(mu_t2, proto_src[yb_t], reduction='mean')

                    if use_CORAL_G or use_CORAL_Cf:
                        cov_t = batch_covar(mu_t2)
                        if use_CORAL_G:
                            coral_g = coral_loss(cov_src_g.to(device).to(dtype=cov_t.dtype), cov_t)
                        if use_CORAL_Cf and cov_src_c is not None:
                            present = yb_t.unique(); cnt = 0
                            for c in present:
                                m = (yb_t == c)
                                if int(m.sum()) >= 2:
                                    cov_tc = batch_covar(mu_t2[m])
                                    coral_c = coral_c + coral_loss(cov_src_c[int(c)].to(device).to(dtype=cov_tc.dtype), cov_tc)
                                    cnt += 1
                            if cnt > 0: coral_c = coral_c / cnt

                    if use_SUPCON:
                        if xb_s is not None:
                            mu_s2 = align(vae.encode(xb_s)[0].float())
                            if class_bias is not None:
                                mu_s2 = class_bias(mu_s2, yb_s)
                            z = torch.cat([mu_s2, mu_t2], 0)
                            lab = torch.cat([yb_s, yb_t], 0)
                        else:
                            z = mu_t2; lab = yb_t
                        if z.size(0) >= 16:
                            sc = supcon_loss(z, lab, T=0.07)

                    if (args.use_uncert_consistency and (use_UNC_KL or use_UNC_MU)):
                        if args.unc_mode == 'id':
                            idxs = [label2idx_src.get(int(l), -1) for l in labid_t.tolist()]
                            mask = [i for i,ii in enumerate(idxs) if ii >= 0]
                            if len(mask) > 0:
                                xs_teacher = to_tensor_resized_X(X0[[idxs[i] for i in mask]], SIZE, device)
                                with torch.no_grad():
                                    mu_src = vae.encode(xs_teacher)[0].float()
                                    rec_src_mu, rec_src_lv = vae.decode(mu_src, out_hw=(SIZE,SIZE))
                                mu_t2_sel = mu_t2[mask]
                                rec_tgt_mu, rec_tgt_lv = vae.decode(mu_t2_sel, out_hw=(SIZE,SIZE))
                                rec_src_mu = rec_src_mu.clamp(0,1).detach(); rec_tgt_mu = rec_tgt_mu.clamp(0,1)
                                if use_UNC_KL:
                                    ukl = gaussian_sym_kl(rec_tgt_mu, rec_tgt_lv, rec_src_mu, rec_src_lv)
                                if use_UNC_MU:
                                    umu = F.mse_loss(rec_tgt_mu, rec_src_mu)
                        elif args.unc_mode == 'classproto':
                            with torch.no_grad():
                                t_mu = class_teacher_mu_c[yb_t]
                                t_lv = class_teacher_lv_c[yb_t]
                            rec_tgt_mu, rec_tgt_lv = vae.decode(mu_t2, out_hw=(SIZE,SIZE))
                            rec_tgt_mu = rec_tgt_mu.clamp(0,1); t_mu = t_mu.clamp(0,1)
                            if use_UNC_KL:
                                ukl = gaussian_sym_kl(rec_tgt_mu, rec_tgt_lv, t_mu, t_lv)
                            if use_UNC_MU:
                                umu = F.mse_loss(rec_tgt_mu, t_mu)
                        elif args.unc_mode == 'ema' and (ema_align is not None):
                            with torch.no_grad():
                                mu_t2_teacher = ema_align(mu_t)
                                if ema_class_bias is not None:
                                    mu_t2_teacher = ema_class_bias(mu_t2_teacher, yb_t)
                                t_mu, t_lv = vae.decode(mu_t2_teacher, out_hw=(SIZE,SIZE))
                            rec_tgt_mu, rec_tgt_lv = vae.decode(mu_t2, out_hw=(SIZE,SIZE))
                            rec_tgt_mu = rec_tgt_mu.clamp(0,1); t_mu = t_mu.clamp(0,1)
                            if use_UNC_KL:
                                ukl = gaussian_sym_kl(rec_tgt_mu, rec_tgt_lv, t_mu, t_lv)
                            if use_UNC_MU:
                                umu = F.mse_loss(rec_tgt_mu, t_mu)

                    if use_ORTH:
                        oreg = orth_reg(align, lam=args.orth_lambda)

                    loss = lam_ce*ce + lam_prot*prot + lam_coral*(coral_g + coral_c) + lam_sup*sc + oreg + lam_unc*ukl + lam_uncm*umu

                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()

                if ema_align is not None:
                    with torch.no_grad():
                        ema_update(ema_align, align, momentum=args.ema_momentum)
                        if ema_class_bias is not None:
                            ema_update(ema_class_bias, class_bias, momentum=args.ema_momentum)

                sCE+=float(ce.item()); sPR+=float(prot.item()); sCOg+=float(coral_g.item()); sCOc+=float(coral_c.item()); sSC+=float(sc.item()); sORTH+=float(oreg.item()); sUKL+=float(ukl.item()); sUMU+=float(umu.item()); sTOT+=float(loss.item()); nb+=1

                del xb_t, yb_t, labid_t
                if xb_s is not None: del xb_s, yb_s

            with open(hist_csv, "a", newline="") as f:
                csv.writer(f).writerow([target_domain, ep, sCE/max(1,nb), sPR/max(1,nb), sCOg/max(1,nb), sCOc/max(1,nb), sSC/max(1,nb), sORTH/max(1,nb), sUKL/max(1,nb), sUMU/max(1,nb), sTOT/max(1,nb)])
            log(f"[D{target_domain} ep{ep}/{args.epochs_per_domain}] CE={sCE/max(1,nb):.4f} Prot={sPR/max(1,nb):.4f} "
                f"CORALg={sCOg/max(1,nb):.6f} CORALc={sCOc/max(1,nb):.6f} "
                f"{('(SupCon='+format(sSC/max(1,nb),'.4f')+')') if (args.use_supcon and args.w_supcon>0 and not args.no_supcon) else ''} "
                f"Orth={sORTH/max(1,nb):.4f} UncKL={sUKL/max(1,nb):.4f} UncMu={sUMU/max(1,nb):.4f} Total={sTOT/max(1,nb):.4f}")

            # quick eval per epoch
            base, da = eval_on_arrays(vae, head, align, Xt, Yt, Lt, device, SIZE, batch=args.tsne_batch, amp_on=amp_on, per_digit_hint=args.per_digit_hint, class_bias=class_bias)
            with open(val_csv, "a", newline="") as f:
                csv.writer(f).writerow([target_domain, f"epoch{ep}_base", base["MSE"], base["PSNR"], base["SSIM"], base["Acc"], base["N"]])
                csv.writer(f).writerow([target_domain, f"epoch{ep}_da",   da["MSE"],   da["PSNR"],   da["SSIM"],   da["Acc"],   da["N"]])

            # periodic viz
            if args.viz_every > 0 and (ep % args.viz_every == 0):
                try:
                    recon_path = viz_dir / f"epoch{ep}_recon_grid.png"
                    viz_recon_grid(vae, align, class_bias, Xt, Yt, Lt, device, SIZE, recon_path,
                                   amp_on=amp_on, per_digit_hint=args.per_digit_hint, rows=args.viz_rows)
                    viz_tsne_pair(vae, head, align, class_bias, Xt, Lt, device, SIZE,
                                  out_before=viz_dir / f"epoch{ep}_tsne_before.png",
                                  out_after =viz_dir / f"epoch{ep}_tsne_after.png",
                                  amp_on=amp_on, per_digit_hint=args.per_digit_hint,
                                  per_digit=args.tsne_per_digit, batch=args.tsne_batch)
                    log(f"[viz] saved recon+tsne at epoch {ep} -> {viz_dir}")
                except Exception as e:
                    log(f"[viz] failed at epoch {ep}: {e}")

        # after-domain eval
        final = eval_on_arrays(vae, head, align, Xt, Yt, Lt, device, SIZE, batch=args.tsne_batch, amp_on=amp_on, per_digit_hint=args.per_digit_hint, class_bias=class_bias)[1]
        with open(val_csv, "a", newline="") as f:
            csv.writer(f).writerow([target_domain, "after", final["MSE"], final["PSNR"], final["SSIM"], final["Acc"], final["N"]])
        log(f"[after] D{target_domain} post-DA MSE={final['MSE']:.6f} PSNR={final['PSNR']:.2f} SSIM={final['SSIM']:.4f} Acc={final['Acc']:.2f}%")
        torch.save(align.state_dict(), run_dir/"checkpoints"/f"align_after_domain{target_domain}.pth")
        if class_bias is not None:
            torch.save(class_bias.state_dict(), run_dir/"checkpoints"/f"classbias_after_domain{target_domain}.pth")

        # update anchors with aligned target (EMA)
        with torch.no_grad():
            proto_t_aligned = compute_prototypes(vae, Xt, Lt, args.per_digit_hint, device, SIZE, args.tsne_batch, amp_on, align=align)
            cov_t_aligned_g = compute_covariance_over_X(vae, Xt, device, SIZE, args.tsne_batch, amp_on, align=align)
            proto_src = (1-ema)*proto_src + ema*proto_t_aligned
            cov_src_g = (1-ema)*cov_src_g + ema*cov_t_aligned_g
            if args.use_classwise_coral and not args.no_coral_c:
                cov_t_aligned_c = compute_classwise_cov(vae, Xt, Lt, args.per_digit_hint, device, SIZE, args.tsne_batch, amp_on, align=align)
                if isinstance(cov_src_c, list):
                    cov_src_c = [(1-ema)*cov_src_c[k] + ema*cov_t_aligned_c[k] for k in range(10)]
                else:
                    cov_src_c = cov_t_aligned_c

        # add target to replay buffers
        src_X_list.append(Xt); src_L_list.append(Lt)

    # =============== zero-shot evaluation ===============
    for zs_dom in zs_domains:
        log(f"\n=== ZERO-SHOT evaluate bending{zs_dom} (no training) ===")
        Xz, Yz, Lz = load_bending_arrays(root, zs_dom)
        base, da = eval_on_arrays(vae, head, align, Xz, Yz, Lz, device, SIZE, batch=args.tsne_batch, amp_on=amp_on, per_digit_hint=args.per_digit_hint, class_bias=class_bias)
        with open(val_csv, "a", newline="") as f:
            csv.writer(f).writerow([zs_dom, "zero_shot_base", base["MSE"], base["PSNR"], base["SSIM"], base["Acc"], base["N"]])
            csv.writer(f).writerow([zs_dom, "zero_shot",      da["MSE"],   da["PSNR"],   da["SSIM"],   da["Acc"],   da["N"]])
        log(f"[zero-shot] bend{zs_dom} base: MSE={base['MSE']:.6f} PSNR={base['PSNR']:.2f} SSIM={base['SSIM']:.4f} Acc={base['Acc']:.2f}%")
        log(f"[zero-shot] bend{zs_dom}  DA : MSE={da['MSE']:.6f} PSNR={da['PSNR']:.2f} SSIM={da['SSIM']:.4f} Acc={da['Acc']:.2f}%")
        zsviz = run_dir/"viz"/f"bending{zs_dom}_zero_shot"
        try:
            viz_recon_grid(vae, align, class_bias, Xz, Yz, Lz, device, SIZE, zsviz/"recon_grid.png",
                           amp_on=amp_on, per_digit_hint=args.per_digit_hint, rows=args.viz_rows)
            viz_tsne_pair(vae, head, align, class_bias, Xz, Lz, device, SIZE,
                          out_before=zsviz/"tsne_before.png", out_after=zsviz/"tsne_after.png",
                          amp_on=amp_on, per_digit_hint=args.per_digit_hint,
                          per_digit=args.tsne_per_digit, batch=args.tsne_batch)
        except Exception as e:
            log(f"[viz zero-shot] failed: {e}")

    with open(run_dir/"done.json","w") as f:
        json.dump({"finished_at": datetime.now().isoformat(timespec="seconds")}, f, indent=2)
    log(f"All done. Logs: {run_dir}")

# ---- helper to build classproto teacher from current prototypes ----
@torch.no_grad()
def build_class_teacher_maps_from_proto(vae, proto_src, size, device, amp_on):
    # decode per-class prototype mu into image-space Gaussian params
    mu_dec = []
    lv_dec = []
    y_idx = torch.arange(10, dtype=torch.long, device=device)
    z = proto_src  # [10, latent]
    with torch.amp.autocast("cuda", enabled=amp_on and device.type=="cuda"):
        m, lv = vae.decode(z, out_hw=(size,size))
    mu_dec = m.detach().clamp(0,1)
    lv_dec = lv.detach()
    return mu_dec, lv_dec

if __name__ == "__main__":
    main()

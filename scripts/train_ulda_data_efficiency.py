#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ulda_subtrain_sweep_target_tsne_alien.py

Key fix:
- t-SNE after DA MUST use latent AFTER alien layer (AlignLayer [+ optional ClassCondBias]).
  i.e., z_alien = alien(mu); (and if class_bias enabled, z_alien = class_bias(z_alien, y))

Pipeline:
- Target domain split: train/val/test = 8:1:1 (stratified by digit)
- Train on TRAIN subset fractions 10%..100% (nested, per class)
- Each fraction:
  - Train Align(+ClassBias) with replay from source (bending0)
  - Evaluate on VAL every epoch and TEST after training (base vs DA)
  - From TEST: pick 1 sample per digit (0-9) => save recon before/after + uncertainty maps
  - From TEST: save t-SNE on base latent and on alien-latent (DA latent)

Outputs:
  runs/.../<timestamp>/frac010 ... frac100/
    viz/recon_unc_grid.png
    viz/tsne_test_base.png
    viz/tsne_test_alien.png   <-- fixed: uses alien layer output
"""

import os, math, csv, json, argparse, random
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Tuple
from contextlib import nullcontext

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ------------------------- utils -------------------------

def log(msg: str, *, flush=True):
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=flush)

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True

def autocast_ctx(device: torch.device, enabled: bool):
    if device.type == "cuda":
        return torch.amp.autocast("cuda", enabled=enabled)
    return nullcontext()

def label_to_digit(labels: np.ndarray, per_digit_hint: int) -> np.ndarray:
    z = labels.astype(np.int64) - 1
    return (z % (per_digit_hint * 10)) // per_digit_hint

def to_tensor_resized_X(batch_np: np.ndarray, size: int, device) -> torch.Tensor:
    t = torch.from_numpy(batch_np.astype(np.float32, copy=True))
    if t.ndim == 2:
        t = t.unsqueeze(0).unsqueeze(0)
    elif t.ndim == 3:
        t = t.unsqueeze(1)
    elif t.ndim == 4:
        pass
    else:
        raise ValueError(f"speckle ndim={t.ndim}")

    B = t.shape[0]
    t2 = t.view(B, -1)
    mn = t2.min(dim=1, keepdim=True).values
    mx = t2.max(dim=1, keepdim=True).values
    t2 = (t2 - mn) / (mx - mn + 1e-8)
    t = t2.view(B, 1, t.shape[-2], t.shape[-1])

    if t.shape[-2] != size or t.shape[-1] != size:
        t = F.interpolate(t, size=(size, size), mode="bilinear", align_corners=False)

    return t.to(device, non_blocking=True)

def to_tensor_resized_Y(batch_np: np.ndarray, size: int, device) -> torch.Tensor:
    t = torch.from_numpy(batch_np.astype(np.float32, copy=True))
    if t.ndim == 2:
        t = t.unsqueeze(0).unsqueeze(0)
    elif t.ndim == 3:
        t = t.unsqueeze(1)
    elif t.ndim == 4:
        pass
    else:
        raise ValueError(f"gt ndim={t.ndim}")

    if t.shape[-2] != size or t.shape[-1] != size:
        t = F.interpolate(t, size=(size, size), mode="bilinear", align_corners=False)
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
    X = np.load(p/"speckles_sorted.npy", mmap_mode="r")
    Y = np.load(p/"images_sorted.npy",   mmap_mode="r")
    L = np.load(p/"labels_sorted.npy",   mmap_mode="r")
    return X, Y, L

def make_label2idx(L: np.ndarray) -> Dict[int, int]:
    d = {}
    Llist = L.tolist()
    for i, lab in enumerate(Llist):
        if lab not in d:
            d[lab] = i
    return d

def stratified_split_indices(L: np.ndarray, per_digit_hint: int, seed: int,
                             ratios=(0.8, 0.1, 0.1),
                             min_per_class_val: int = 1,
                             min_per_class_test: int = 1) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    assert abs(sum(ratios) - 1.0) < 1e-6
    rng = np.random.default_rng(seed)
    digits = label_to_digit(L, per_digit_hint).astype(np.int64)

    tr, va, te = [], [], []
    for c in range(10):
        idx_c = np.where(digits == c)[0]
        if len(idx_c) == 0:
            continue
        rng.shuffle(idx_c)

        n = len(idx_c)
        n_val = max(min_per_class_val, int(round(ratios[1] * n)))
        n_test = max(min_per_class_test, int(round(ratios[2] * n)))
        n_val = min(n_val, n)
        n_test = min(n_test, n - n_val) if (n - n_val) > 0 else 0

        n_train = n - n_val - n_test
        if n_train <= 0:
            if n > 2:
                n_train = 1
                n_val = min(n_val, n - n_train)
                n_test = n - n_train - n_val
            else:
                n_train = max(1, n - 1)
                n_val = n - n_train
                n_test = 0

        tr.append(idx_c[:n_train])
        va.append(idx_c[n_train:n_train+n_val])
        te.append(idx_c[n_train+n_val:n_train+n_val+n_test])

    train_idx = np.concatenate(tr).astype(np.int64) if len(tr) else np.array([], dtype=np.int64)
    val_idx   = np.concatenate(va).astype(np.int64) if len(va) else np.array([], dtype=np.int64)
    test_idx  = np.concatenate(te).astype(np.int64) if len(te) else np.array([], dtype=np.int64)

    rng.shuffle(train_idx); rng.shuffle(val_idx); rng.shuffle(test_idx)
    return train_idx, val_idx, test_idx

def build_nested_train_orders(train_idx: np.ndarray, L_full: np.ndarray, per_digit_hint: int, seed: int):
    rng = np.random.default_rng(seed)
    digits = label_to_digit(L_full[train_idx], per_digit_hint).astype(np.int64)
    orders = {}
    for c in range(10):
        idx_c = train_idx[np.where(digits == c)[0]]
        idx_c = idx_c.copy()
        rng.shuffle(idx_c)
        orders[c] = idx_c
    return orders

def subset_from_orders(orders: Dict[int, np.ndarray], frac: float, min_per_class: int = 1) -> np.ndarray:
    frac = float(frac)
    assert 0.0 < frac <= 1.0
    picks = []
    for c, arr in orders.items():
        if len(arr) == 0:
            continue
        k = int(math.floor(frac * len(arr)))
        k = max(k, min_per_class)
        k = min(k, len(arr))
        picks.append(arr[:k])
    out = np.concatenate(picks).astype(np.int64) if len(picks) else np.array([], dtype=np.int64)
    rng = np.random.default_rng(12345)
    rng.shuffle(out)
    return out


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
            self.enc_shape = h.shape[1:]           # (256,16,16)
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
            mu = F.interpolate(mu, size=out_hw, mode="bilinear", align_corners=False)
            logvar = F.interpolate(logvar, size=out_hw, mode="bilinear", align_corners=False)
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
            if use_bias:
                nn.init.zeros_(self.fc.bias)
        self.residual = residual
    def forward(self, z):
        y = self.fc(z)
        return z + y if self.residual else y

class ClassCondBias(nn.Module):
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
    logv1 = torch.clamp(logv1, -6, 3)
    logv2 = torch.clamp(logv2, -6, 3)
    v1 = torch.exp(logv1)
    v2 = torch.exp(logv2)
    kl12 = 0.5 * ((v1 + (mu1 - mu2)**2) / v2 - 1.0 + (logv2 - logv1))
    kl21 = 0.5 * ((v2 + (mu2 - mu1)**2) / v1 - 1.0 + (logv1 - logv2))
    skl = (kl12 + kl21) * 0.5
    return skl.flatten(1).mean(1).mean()


# ----------- anchors (source) -----------

def strip_module(sd):
    return { (k[7:] if k.startswith("module.") else k): v for k,v in sd.items() }

def load_vae_head(vae, head, ckpt_path, strict_vae=False, strict_head=False):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    vae_sd  = ckpt.get("vae",  ckpt)
    head_sd = ckpt.get("head", {})
    m,u = vae.load_state_dict(strip_module(vae_sd), strict=strict_vae)
    log(f"[load] VAE missing={len(m)} unexpected={len(u)}")
    if isinstance(head_sd, dict) and len(head_sd) > 0:
        m2,u2 = head.load_state_dict(strip_module(head_sd), strict=strict_head)
        log(f"[load] HEAD missing={len(m2)} unexpected={len(u2)}")
    else:
        log("[load] HEAD not found in ckpt; using current head weights.")
    return vae, head

@torch.no_grad()
def compute_prototypes(vae: nn.Module, X: np.ndarray, L: np.ndarray,
                       per_digit_hint: int, device, size: int,
                       batch: int, amp_on: bool) -> torch.Tensor:
    digits = label_to_digit(L, per_digit_hint).astype(np.int64)
    latent_dim = int(vae.fc_mu.out_features)
    sums = torch.zeros(10, latent_dim, device=device)
    cnt  = torch.zeros(10, device=device)
    for s in range(0, len(L), batch):
        e = min(s+batch, len(L))
        xb = to_tensor_resized_X(X[s:e], size, device)
        with autocast_ctx(device, amp_on):
            mu = vae.encode(xb)[0].float()
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
                              batch: int, amp_on: bool) -> torch.Tensor:
    latent_dim = int(vae.fc_mu.out_features)
    sum_mu  = torch.zeros(latent_dim, device=device)
    sum_xxt = torch.zeros(latent_dim, latent_dim, device=device)
    N = 0
    for s in range(0, len(X), batch):
        e  = min(s+batch, len(X))
        xb = to_tensor_resized_X(X[s:e], size, device)
        with autocast_ctx(device, amp_on):
            mu = vae.encode(xb)[0].float()
        sum_mu  += mu.sum(0)
        sum_xxt += mu.T @ mu
        N += mu.size(0)
        del xb, mu
    mean = sum_mu / max(1, N)
    cov  = (sum_xxt - N * (mean.unsqueeze(1) @ mean.unsqueeze(0))) / max(1, N-1)
    return (cov + cov.T)/2.0

@torch.no_grad()
def compute_classwise_cov(vae: nn.Module, X: np.ndarray, L: np.ndarray, per_digit_hint: int,
                          device, size: int, batch: int, amp_on: bool) -> List[torch.Tensor]:
    covs = [None]*10
    bufs = [ [] for _ in range(10) ]
    for s in range(0, len(L), batch):
        e = min(s+batch, len(L))
        xb = to_tensor_resized_X(X[s:e], size, device)
        d  = label_to_digit(L[s:e], per_digit_hint).astype(np.int64)
        with autocast_ctx(device, amp_on):
            mu = vae.encode(xb)[0].float()
        mu_cpu = mu.detach().cpu()
        for i in range(mu_cpu.size(0)):
            bufs[int(d[i])].append(mu_cpu[i:i+1])
        del xb, mu
    for k in range(10):
        D = int(vae.fc_mu.out_features)
        if len(bufs[k]) >= 2:
            Z = torch.cat(bufs[k], 0).to(device)
            covs[k] = batch_covar(Z).detach()
        else:
            covs[k] = torch.zeros(D, D, device=device)
    return covs


# ------------------- evaluation -------------------

@torch.no_grad()
def eval_on_indices(vae, head, alien, X, Y, L, idx: np.ndarray, device, size, batch=256, amp_on=False,
                    per_digit_hint=800, class_bias: Optional[nn.Module]=None):
    vae.eval(); head.eval(); alien.eval()
    if class_bias is not None:
        class_bias.eval()

    s_mse0=s_psn0=s_ssm0=0.0
    s_mse1=s_psn1=s_ssm1=0.0
    n=0
    acc0_ok=acc1_ok=0

    digits_full = label_to_digit(L[idx], per_digit_hint).astype(np.int64)

    for s in range(0, len(idx), batch):
        e = min(s+batch, len(idx))
        sel = idx[s:e]
        xs = to_tensor_resized_X(X[sel], size, device)
        ys = to_tensor_resized_Y(Y[sel], size, device)
        yb = torch.from_numpy(digits_full[s:e]).to(device, non_blocking=True)

        with autocast_ctx(device, amp_on):
            mu = vae.encode(xs)[0].float()
            rec0,_ = vae.decode(mu, out_hw=(size,size))

            z_alien = alien(mu)
            if class_bias is not None:
                z_alien = class_bias(z_alien, yb)
            rec1,_ = vae.decode(z_alien, out_hw=(size,size))

            logit0 = head(mu)
            logit1 = head(z_alien)

        rec0 = rec0.clamp(0,1).float()
        rec1 = rec1.clamp(0,1).float()
        ys   = ys.float()

        mse0 = ((rec0-ys)**2).flatten(1).mean(1)
        psn0 = psnr_from_mse(mse0)
        ssm0 = ssim_2d(rec0, ys)

        mse1 = ((rec1-ys)**2).flatten(1).mean(1)
        psn1 = psnr_from_mse(mse1)
        ssm1 = ssim_2d(rec1, ys)

        s_mse0 += float(mse0.sum()); s_psn0 += float(psn0.sum()); s_ssm0 += float(ssm0.sum())
        s_mse1 += float(mse1.sum()); s_psn1 += float(psn1.sum()); s_ssm1 += float(ssm1.sum())
        n += (e - s)

        dnp = digits_full[s:e]
        acc0_ok += int((logit0.argmax(1).detach().cpu().numpy() == dnp).sum())
        acc1_ok += int((logit1.argmax(1).detach().cpu().numpy() == dnp).sum())

        del xs, ys, mu, z_alien, rec0, rec1, logit0, logit1, yb

    base = dict(MSE=s_mse0/max(1,n), PSNR=s_psn0/max(1,n), SSIM=s_ssm0/max(1,n),
                Acc=100.0*acc0_ok/max(1,n), N=n)
    da   = dict(MSE=s_mse1/max(1,n), PSNR=s_psn1/max(1,n), SSIM=s_ssm1/max(1,n),
                Acc=100.0*acc1_ok/max(1,n), N=n)
    return base, da


# ------------------- t-SNE (FIXED: use alien output) -------------------

@torch.no_grad()
def extract_latents_test_for_tsne(
    vae, alien, X, L, idx: np.ndarray,
    device, size, batch=256, amp_on=False, per_digit_hint=800,
    class_bias: Optional[nn.Module]=None,
    standardize: bool=True
):
    """
    Return:
      Z_base  = mu (no alien)
      Z_alien = alien(mu) [+ class_bias]
    """
    vae.eval(); alien.eval()
    if class_bias is not None:
        class_bias.eval()

    digits = label_to_digit(L[idx], per_digit_hint).astype(np.int64)
    Zb_list, Za_list, Y_list = [], [], []

    for s in range(0, len(idx), batch):
        e = min(s+batch, len(idx))
        sel = idx[s:e]
        xs = to_tensor_resized_X(X[sel], size, device)
        yb = torch.from_numpy(digits[s:e]).to(device, non_blocking=True)

        with autocast_ctx(device, amp_on):
            mu = vae.encode(xs)[0].float()
            z_alien = alien(mu)
            if class_bias is not None:
                z_alien = class_bias(z_alien, yb)

        Zb_list.append(mu.detach().cpu().numpy())
        Za_list.append(z_alien.detach().cpu().numpy())
        Y_list.append(digits[s:e])

        del xs, yb, mu, z_alien

    Zb = np.concatenate(Zb_list, 0)
    Za = np.concatenate(Za_list, 0)
    yy = np.concatenate(Y_list, 0)

    if standardize:
        # standardize per-dimension (helps t-SNE stability)
        def stdize(Z):
            m = Z.mean(0, keepdims=True)
            s = Z.std(0, keepdims=True) + 1e-8
            return (Z - m) / s
        Zb = stdize(Zb)
        Za = stdize(Za)

    return Zb, Za, yy

def try_tsne(Z: np.ndarray, seed: int, perplexity: float = 30.0):
    try:
        from sklearn.manifold import TSNE
        n = Z.shape[0]
        p = min(perplexity, max(5.0, (n - 1) / 3.0))
        tsne = TSNE(n_components=2, init="pca", random_state=seed, learning_rate="auto",
                    perplexity=float(p))
        return tsne.fit_transform(Z)
    except Exception as e:
        log(f"[tSNE] skipped (sklearn not available or error): {repr(e)}")
        return None

def save_tsne_plot(emb2d: np.ndarray, y: np.ndarray, out_png: Path, title: str):
    plt.figure(figsize=(7.0, 6.0), dpi=200)
    for c in range(10):
        m = (y == c)
        if m.any():
            plt.scatter(emb2d[m, 0], emb2d[m, 1], s=6, alpha=0.75, label=str(c))
    plt.title(title)
    plt.xticks([]); plt.yticks([])
    plt.legend(markerscale=2, fontsize=8, loc="best", frameon=True)
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()


# ------------------- recon + uncertainty (same as before) -------------------

def pick_one_per_digit(idx: np.ndarray, L_full: np.ndarray, per_digit_hint: int, seed: int) -> Dict[int, int]:
    rng = np.random.default_rng(seed)
    digits = label_to_digit(L_full[idx], per_digit_hint).astype(np.int64)
    picks = {}
    for c in range(10):
        pos = np.where(digits == c)[0]
        if len(pos) == 0:
            continue
        j = int(rng.integers(low=0, high=len(pos)))
        picks[c] = int(idx[pos[j]])
    return picks

@torch.no_grad()
def recon_and_uncertainty(vae, alien, X, Y, L, sample_indices: List[int], device, size, amp_on: bool,
                         per_digit_hint: int, class_bias: Optional[nn.Module]=None):
    vae.eval(); alien.eval()
    if class_bias is not None:
        class_bias.eval()

    sel = np.array(sample_indices, dtype=np.int64)
    xs = to_tensor_resized_X(X[sel], size, device)
    ys = to_tensor_resized_Y(Y[sel], size, device)
    digits = label_to_digit(L[sel], per_digit_hint).astype(np.int64)
    yb = torch.from_numpy(digits).to(device, non_blocking=True)

    with autocast_ctx(device, amp_on):
        mu = vae.encode(xs)[0].float()
        rec0_mu, rec0_lv = vae.decode(mu, out_hw=(size, size))

        z_alien = alien(mu)
        if class_bias is not None:
            z_alien = class_bias(z_alien, yb)
        rec1_mu, rec1_lv = vae.decode(z_alien, out_hw=(size, size))

    rec0_mu = rec0_mu.clamp(0, 1).float()
    rec1_mu = rec1_mu.clamp(0, 1).float()
    ys = ys.float().clamp(0, 1)

    rec0_lv = torch.clamp(rec0_lv.float(), -6, 3)
    rec1_lv = torch.clamp(rec1_lv.float(), -6, 3)
    unc0 = torch.exp(0.5 * rec0_lv)
    unc1 = torch.exp(0.5 * rec1_lv)

    return {
        "digits": digits,
        "gt": ys.detach().cpu(),
        "rec_before": rec0_mu.detach().cpu(),
        "unc_before": unc0.detach().cpu(),
        "rec_after": rec1_mu.detach().cpu(),
        "unc_after": unc1.detach().cpu(),
    }

def save_recon_unc_grid(pack: dict, out_png: Path, unc_cmap: str = "turbo", show_gt: bool = True):
    digits = pack["digits"]
    order = np.argsort(digits)
    gt = pack["gt"][order]
    rb = pack["rec_before"][order]
    ub = pack["unc_before"][order]
    ra = pack["rec_after"][order]
    ua = pack["unc_after"][order]
    digits_sorted = digits[order]

    def norm01(t):
        x = t.squeeze(1).numpy()
        out = []
        for i in range(x.shape[0]):
            a = x[i]
            mn, mx = float(a.min()), float(a.max())
            out.append((a - mn) / (mx - mn + 1e-8))
        return np.stack(out, 0)

    ubn = norm01(ub)
    uan = norm01(ua)

    cols = 5 if show_gt else 4
    fig = plt.figure(figsize=(cols * 2.1, 10 * 1.15), dpi=220)
    titles = (["GT"] if show_gt else []) + ["Rec (base)", "Unc (base)", "Rec (alien)", "Unc (alien)"]

    for r in range(min(10, len(digits_sorted))):
        if show_gt:
            ax = plt.subplot(10, cols, r*cols + 1)
            ax.imshow(gt[r,0].numpy(), cmap="gray", vmin=0, vmax=1)
            ax.set_title(titles[0] if r == 0 else "", fontsize=9)
            ax.set_ylabel(f"{digits_sorted[r]}", rotation=0, labelpad=12, fontsize=10)
            ax.axis("off")

            ax = plt.subplot(10, cols, r*cols + 2)
            ax.imshow(rb[r,0].numpy(), cmap="gray", vmin=0, vmax=1)
            ax.set_title(titles[1] if r == 0 else "", fontsize=9)
            ax.axis("off")

            ax = plt.subplot(10, cols, r*cols + 3)
            ax.imshow(ubn[r], cmap=unc_cmap, vmin=0, vmax=1)
            ax.set_title(titles[2] if r == 0 else "", fontsize=9)
            ax.axis("off")

            ax = plt.subplot(10, cols, r*cols + 4)
            ax.imshow(ra[r,0].numpy(), cmap="gray", vmin=0, vmax=1)
            ax.set_title(titles[3] if r == 0 else "", fontsize=9)
            ax.axis("off")

            ax = plt.subplot(10, cols, r*cols + 5)
            ax.imshow(uan[r], cmap=unc_cmap, vmin=0, vmax=1)
            ax.set_title(titles[4] if r == 0 else "", fontsize=9)
            ax.axis("off")
        else:
            ax = plt.subplot(10, cols, r*cols + 1)
            ax.imshow(rb[r,0].numpy(), cmap="gray", vmin=0, vmax=1)
            ax.set_title(titles[0] if r == 0 else "", fontsize=9)
            ax.set_ylabel(f"{digits_sorted[r]}", rotation=0, labelpad=12, fontsize=10)
            ax.axis("off")

            ax = plt.subplot(10, cols, r*cols + 2)
            ax.imshow(ubn[r], cmap=unc_cmap, vmin=0, vmax=1)
            ax.set_title(titles[1] if r == 0 else "", fontsize=9)
            ax.axis("off")

            ax = plt.subplot(10, cols, r*cols + 3)
            ax.imshow(ra[r,0].numpy(), cmap="gray", vmin=0, vmax=1)
            ax.set_title(titles[2] if r == 0 else "", fontsize=9)
            ax.axis("off")

            ax = plt.subplot(10, cols, r*cols + 4)
            ax.imshow(uan[r], cmap=unc_cmap, vmin=0, vmax=1)
            ax.set_title(titles[3] if r == 0 else "", fontsize=9)
            ax.axis("off")

    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()


# --------------------------- args ---------------------------

def parse_args():
    ap = argparse.ArgumentParser("ULDA subset sweep (FIX tsne: use alien-latent)")
    ap.add_argument("--data_root", type=str, required=True)
    ap.add_argument("--vae_ckpt", type=str, required=True)

    ap.add_argument("--source_domain", type=int, default=0)
    ap.add_argument("--target_domain", type=int, default=1)

    ap.add_argument("--per_digit_hint", type=int, default=800)
    ap.add_argument("--resize_hw", type=int, default=256)

    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--amp", action="store_true")

    ap.add_argument("--train_ratio", type=float, default=0.8)
    ap.add_argument("--val_ratio", type=float, default=0.1)
    ap.add_argument("--test_ratio", type=float, default=0.1)

    ap.add_argument("--min_frac", type=float, default=0.1)
    ap.add_argument("--max_frac", type=float, default=1.0)
    ap.add_argument("--step_frac", type=float, default=0.1)
    ap.add_argument("--min_per_class_subset", type=int, default=1)

    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--replay_ratio", type=float, default=0.25)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)

    ap.add_argument("--w_ce",    type=float, default=1.0)
    ap.add_argument("--w_prot",  type=float, default=2.0)
    ap.add_argument("--w_coral", type=float, default=0.5)
    ap.add_argument("--use_classwise_coral", action="store_true")

    ap.add_argument("--use_supcon", action="store_true")
    ap.add_argument("--w_supcon", type=float, default=0.1)

    ap.add_argument("--use_uncert_consistency", action="store_true")
    ap.add_argument("--w_unc", type=float, default=0.5)
    ap.add_argument("--w_unc_mu", type=float, default=0.0)
    ap.add_argument("--unc_teacher", type=str, default="source", choices=["source","prev"])

    ap.add_argument("--orth_lambda", type=float, default=1e-3)
    ap.add_argument("--use_class_bias", action="store_true")
    ap.add_argument("--no_align_residual", action="store_true")

    ap.add_argument("--eval_batch", type=int, default=256)
    ap.add_argument("--anchor_batch", type=int, default=256)

    ap.add_argument("--tsne_perplexity", type=float, default=30.0)
    ap.add_argument("--tsne_standardize", action="store_true", help="standardize latent before TSNE (recommended)")
    ap.add_argument("--unc_cmap", type=str, default="turbo")
    ap.add_argument("--show_gt_in_grid", action="store_true")

    ap.add_argument("--save_dir", type=str, default="runs/ulda_subtrain_sweep_tsne_alien")
    return ap.parse_args()


# --------------------------- training per fraction ---------------------------

def train_one_fraction(
    frac: float,
    args,
    device,
    amp_on: bool,
    SIZE: int,
    run_dir: Path,
    # data
    X0, Y0, L0,
    Xt, Yt, Lt,
    train_orders: Dict[int, np.ndarray],
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    # anchors
    proto_src: torch.Tensor,
    cov_src_g: torch.Tensor,
    cov_src_c: Optional[List[torch.Tensor]],
    label2idx_src: Dict[int, int],
):
    frac_tag = int(round(frac * 100))
    sub_dir = run_dir / f"frac{frac_tag:03d}"
    (sub_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (sub_dir / "viz").mkdir(parents=True, exist_ok=True)

    set_seed(args.seed)

    train_sub_idx = subset_from_orders(train_orders, frac=frac, min_per_class=args.min_per_class_subset)
    np.save(sub_dir / f"subset_indices_frac{frac_tag:03d}.npy", train_sub_idx)
    log(f"[frac {frac:.2f}] train subset: {len(train_sub_idx)}")

    # build models for this fraction
    vae  = SimpleVAEWithUncertainty(input_hw=(SIZE,SIZE), latent_dim=512).to(device).eval()
    head = LatentDigitHead(latent_dim=512, p_drop=0.2).to(device).eval()
    vae, head = load_vae_head(vae, head, args.vae_ckpt, strict_vae=False, strict_head=False)
    for p in vae.parameters():  p.requires_grad_(False)
    for p in head.parameters(): p.requires_grad_(False)

    alien = AlignLayer(latent_dim=512, use_bias=True, residual=(not args.no_align_residual)).to(device).train()
    class_bias = ClassCondBias(num_classes=10, dim=512).to(device).train() if args.use_class_bias else None

    params = list(alien.parameters()) + (list(class_bias.parameters()) if class_bias is not None else [])
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)

    hist_csv = sub_dir / "history_epoch_losses.csv"
    eval_csv = sub_dir / "domain_eval.csv"
    with open(hist_csv, "w", newline="") as f:
        csv.writer(f).writerow(["frac","epoch","CE","PROT","CORAL_G","CORAL_C","SupCon","Orth","UncKL","UncMu","Total"])
    with open(eval_csv, "w", newline="") as f:
        csv.writer(f).writerow(["frac","split","MSE","PSNR","SSIM","Acc","N"])

    # before eval
    base_v0, da_v0 = eval_on_indices(vae, head, alien, Xt, Yt, Lt, val_idx, device, SIZE,
                                     batch=args.eval_batch, amp_on=amp_on, per_digit_hint=args.per_digit_hint, class_bias=class_bias)
    base_t0, da_t0 = eval_on_indices(vae, head, alien, Xt, Yt, Lt, test_idx, device, SIZE,
                                     batch=args.eval_batch, amp_on=amp_on, per_digit_hint=args.per_digit_hint, class_bias=class_bias)
    with open(eval_csv, "a", newline="") as f:
        w = csv.writer(f)
        w.writerow([frac, "val_before_base", base_v0["MSE"], base_v0["PSNR"], base_v0["SSIM"], base_v0["Acc"], base_v0["N"]])
        w.writerow([frac, "val_before_alien", da_v0["MSE"], da_v0["PSNR"], da_v0["SSIM"], da_v0["Acc"], da_v0["N"]])
        w.writerow([frac, "test_before_base", base_t0["MSE"], base_t0["PSNR"], base_t0["SSIM"], base_t0["Acc"], base_t0["N"]])
        w.writerow([frac, "test_before_alien", da_t0["MSE"], da_t0["PSNR"], da_t0["SSIM"], da_t0["Acc"], da_t0["N"]])

    # training sizes
    N_tgt = len(train_sub_idx)
    N_src = len(L0)
    src_bs = int(args.batch_size * args.replay_ratio) if N_src > 0 else 0
    tgt_bs = max(1, args.batch_size - src_bs)
    steps_per_epoch = max(1, math.ceil(N_tgt / tgt_bs))

    rng_src = np.random.default_rng(args.seed + 999)

    for ep in range(1, args.epochs + 1):
        alien.train()
        if class_bias is not None:
            class_bias.train()

        sCE=sPR=sCOg=sCOc=sSC=sORTH=sUKL=sUMU=sTOT=0.0
        nb = 0

        sprog = ep / float(args.epochs)
        lam_coral = args.w_coral * sprog
        lam_sup   = (args.w_supcon * max(0.0, (sprog - 0.3)/0.7)) if args.use_supcon else 0.0
        lam_unc   = (args.w_unc * sprog) if args.use_uncert_consistency else 0.0
        lam_unc_mu= (args.w_unc_mu * sprog) if args.use_uncert_consistency else 0.0

        rng_ep = np.random.default_rng(args.seed + ep * 13)
        perm = rng_ep.permutation(N_tgt)

        for step in range(steps_per_epoch):
            t_s = step * tgt_bs
            t_e = min((step + 1) * tgt_bs, N_tgt)
            if t_s >= t_e:
                break

            sel_t = train_sub_idx[perm[t_s:t_e]]
            xb_t = to_tensor_resized_X(Xt[sel_t], SIZE, device)
            yb_t_np = label_to_digit(Lt[sel_t], args.per_digit_hint).astype(np.int64)
            yb_t = torch.from_numpy(yb_t_np).to(device, non_blocking=True)
            labid_t = torch.from_numpy(Lt[sel_t].astype(np.int64))

            xb_s = None
            yb_s = None
            if src_bs > 0:
                replace = (N_src < src_bs)
                sid = rng_src.choice(N_src, size=src_bs, replace=replace).astype(np.int64)
                xb_s = to_tensor_resized_X(X0[sid], SIZE, device)
                yb_s = torch.from_numpy(label_to_digit(L0[sid], args.per_digit_hint).astype(np.int64)).to(device, non_blocking=True)

            with autocast_ctx(device, amp_on):
                mu_t = vae.encode(xb_t)[0].float()
                z_alien = alien(mu_t)
                if class_bias is not None:
                    z_alien = class_bias(z_alien, yb_t)

                logits_t = head(z_alien)
                ce = F.cross_entropy(logits_t, yb_t)

                # prototype anchor (bias-aware)
                if class_bias is not None:
                    prot_anchor = proto_src[yb_t] + class_bias.bias(yb_t)
                else:
                    prot_anchor = proto_src[yb_t]
                prot = F.mse_loss(z_alien, prot_anchor, reduction="mean")

                cov_t = batch_covar(z_alien)
                coral_g = coral_loss(cov_src_g.to(device).to(dtype=cov_t.dtype), cov_t)

                coral_c = torch.tensor(0.0, device=device)
                if args.use_classwise_coral and (cov_src_c is not None):
                    present = yb_t.unique()
                    cnt = 0
                    for c in present:
                        m = (yb_t == c)
                        if int(m.sum()) >= 2:
                            cov_tc = batch_covar(z_alien[m])
                            coral_c = coral_c + coral_loss(cov_src_c[int(c)].to(device).to(dtype=cov_tc.dtype), cov_tc)
                            cnt += 1
                    if cnt > 0:
                        coral_c = coral_c / cnt

                sc = torch.tensor(0.0, device=device)
                if args.use_supcon:
                    if xb_s is not None:
                        mu_s = vae.encode(xb_s)[0].float()
                        zs = alien(mu_s)
                        if class_bias is not None:
                            zs = class_bias(zs, yb_s)
                        z = torch.cat([zs, z_alien], 0)
                        lab = torch.cat([yb_s, yb_t], 0)
                    else:
                        z = z_alien
                        lab = yb_t
                    if z.size(0) >= 16:
                        sc = supcon_loss(z, lab, T=0.07)

                ukl = torch.tensor(0.0, device=device)
                umu = torch.tensor(0.0, device=device)
                if args.use_uncert_consistency:
                    idxs = [label2idx_src.get(int(l), -1) for l in labid_t.tolist()]
                    mask = [i for i, ii in enumerate(idxs) if ii >= 0]
                    if len(mask) > 0:
                        idxs_valid = [idxs[i] for i in mask]
                        xs_teacher = to_tensor_resized_X(X0[idxs_valid], SIZE, device)

                        mu_src = vae.encode(xs_teacher)[0].float()
                        z_src = mu_src
                        # teacher should be in SAME latent space as student (alien space)
                        z_src = alien(z_src)
                        if class_bias is not None:
                            y_teacher = torch.from_numpy(label_to_digit(L0[idxs_valid], args.per_digit_hint).astype(np.int64)).to(device, non_blocking=True)
                            z_src = class_bias(z_src, y_teacher)

                        rec_src_mu, rec_src_lv = vae.decode(z_src, out_hw=(SIZE, SIZE))

                        z_sel = z_alien[mask]
                        rec_tgt_mu, rec_tgt_lv = vae.decode(z_sel, out_hw=(SIZE, SIZE))

                        rec_src_mu = rec_src_mu.clamp(0, 1)
                        rec_tgt_mu = rec_tgt_mu.clamp(0, 1)

                        ukl = gaussian_sym_kl(rec_tgt_mu, rec_tgt_lv, rec_src_mu, rec_src_lv)
                        if args.w_unc_mu > 0:
                            umu = F.mse_loss(rec_tgt_mu, rec_src_mu)

                oreg = orth_reg(alien, lam=args.orth_lambda)

                loss = (args.w_ce * ce
                        + args.w_prot * prot
                        + lam_coral * (coral_g + coral_c)
                        + lam_sup * sc
                        + oreg
                        + lam_unc * ukl
                        + lam_unc_mu * umu)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            sCE+=float(ce.item()); sPR+=float(prot.item())
            sCOg+=float(coral_g.item()); sCOc+=float(coral_c.item())
            sSC+=float(sc.item()); sORTH+=float(oreg.item())
            sUKL+=float(ukl.item()); sUMU+=float(umu.item())
            sTOT+=float(loss.item()); nb+=1

            del xb_t, yb_t, labid_t
            if xb_s is not None:
                del xb_s, yb_s

        with open(hist_csv, "a", newline="") as f:
            csv.writer(f).writerow([frac, ep,
                                    sCE/max(1,nb), sPR/max(1,nb),
                                    sCOg/max(1,nb), sCOc/max(1,nb),
                                    sSC/max(1,nb), sORTH/max(1,nb),
                                    sUKL/max(1,nb), sUMU/max(1,nb),
                                    sTOT/max(1,nb)])
        log(f"[frac {frac:.2f} ep{ep}/{args.epochs}] Total={sTOT/max(1,nb):.4f}")

        # val eval each epoch
        base_v, da_v = eval_on_indices(vae, head, alien, Xt, Yt, Lt, val_idx, device, SIZE,
                                       batch=args.eval_batch, amp_on=amp_on, per_digit_hint=args.per_digit_hint, class_bias=class_bias)
        with open(eval_csv, "a", newline="") as f:
            w = csv.writer(f)
            w.writerow([frac, f"val_ep{ep}_base", base_v["MSE"], base_v["PSNR"], base_v["SSIM"], base_v["Acc"], base_v["N"]])
            w.writerow([frac, f"val_ep{ep}_alien", da_v["MSE"], da_v["PSNR"], da_v["SSIM"], da_v["Acc"], da_v["N"]])

    # final test eval
    base_t, da_t = eval_on_indices(vae, head, alien, Xt, Yt, Lt, test_idx, device, SIZE,
                                   batch=args.eval_batch, amp_on=amp_on, per_digit_hint=args.per_digit_hint, class_bias=class_bias)
    with open(eval_csv, "a", newline="") as f:
        w = csv.writer(f)
        w.writerow([frac, "test_after_base", base_t["MSE"], base_t["PSNR"], base_t["SSIM"], base_t["Acc"], base_t["N"]])
        w.writerow([frac, "test_after_alien", da_t["MSE"], da_t["PSNR"], da_t["SSIM"], da_t["Acc"], da_t["N"]])

    torch.save(alien.state_dict(), sub_dir/"checkpoints"/"alien_final.pth")
    if class_bias is not None:
        torch.save(class_bias.state_dict(), sub_dir/"checkpoints"/"classbias_final.pth")

    # recon grid
    picks = pick_one_per_digit(test_idx, Lt, args.per_digit_hint, seed=args.seed + frac_tag)
    sample_indices = [picks[d] for d in range(10) if d in picks]
    pack = recon_and_uncertainty(vae, alien, Xt, Yt, Lt, sample_indices, device, SIZE, amp_on,
                                 per_digit_hint=args.per_digit_hint, class_bias=class_bias)
    save_recon_unc_grid(pack, sub_dir/"viz"/"recon_unc_grid.png",
                        unc_cmap=args.unc_cmap, show_gt=bool(args.show_gt_in_grid))

    # -------- t-SNE (FIXED) --------
    Zb, Za, yy = extract_latents_test_for_tsne(
        vae, alien, Xt, Lt, test_idx,
        device, SIZE, batch=args.eval_batch, amp_on=amp_on,
        per_digit_hint=args.per_digit_hint, class_bias=class_bias,
        standardize=bool(args.tsne_standardize)
    )

    emb_b = try_tsne(Zb, seed=args.seed + 1, perplexity=args.tsne_perplexity)
    if emb_b is not None:
        save_tsne_plot(emb_b, yy, sub_dir/"viz"/"tsne_test_base.png",
                       title=f"t-SNE TEST (base mu) | frac={frac:.2f}")

    emb_a = try_tsne(Za, seed=args.seed + 2, perplexity=args.tsne_perplexity)
    if emb_a is not None:
        save_tsne_plot(emb_a, yy, sub_dir/"viz"/"tsne_test_alien.png",
                       title=f"t-SNE TEST (after AlienLayer) | frac={frac:.2f}")

    log(f"[frac {frac:.2f}] done. TEST base PSNR={base_t['PSNR']:.2f} | TEST alien PSNR={da_t['PSNR']:.2f}")
    return base_t, da_t


# --------------------------- main ---------------------------

def main():
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_on = bool(args.amp and device.type == "cuda")
    SIZE = int(args.resize_hw)

    root = Path(args.data_root)
    X0, Y0, L0 = load_bending_arrays(root, args.source_domain)
    Xt, Yt, Lt = load_bending_arrays(root, args.target_domain)
    label2idx_src = make_label2idx(L0)

    run_dir = Path(args.save_dir) / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)

    with open(run_dir/"args.json","w") as f:
        json.dump(vars(args), f, indent=2)

    ratios = (args.train_ratio, args.val_ratio, args.test_ratio)
    train_idx, val_idx, test_idx = stratified_split_indices(Lt, args.per_digit_hint, seed=args.seed, ratios=ratios)
    np.savez(run_dir/"split_indices.npz", train_idx=train_idx, val_idx=val_idx, test_idx=test_idx)

    log(f"[split] bending{args.target_domain}: train={len(train_idx)} val={len(val_idx)} test={len(test_idx)}")

    train_orders = build_nested_train_orders(train_idx, Lt, args.per_digit_hint, seed=args.seed)

    # compute source anchors once
    log("[anchors] computing source anchors once...")
    vae_tmp  = SimpleVAEWithUncertainty(input_hw=(SIZE,SIZE), latent_dim=512).to(device).eval()
    head_tmp = LatentDigitHead(latent_dim=512, p_drop=0.2).to(device).eval()
    vae_tmp, head_tmp = load_vae_head(vae_tmp, head_tmp, args.vae_ckpt, strict_vae=False, strict_head=False)
    for p in vae_tmp.parameters():  p.requires_grad_(False)
    for p in head_tmp.parameters(): p.requires_grad_(False)

    proto_src = compute_prototypes(vae_tmp, X0, L0, args.per_digit_hint, device, SIZE, batch=args.anchor_batch, amp_on=amp_on).detach()
    cov_src_g = compute_covariance_over_X(vae_tmp, X0, device, SIZE, batch=args.anchor_batch, amp_on=amp_on).detach()
    cov_src_c = None
    if args.use_classwise_coral:
        cov_src_c = compute_classwise_cov(vae_tmp, X0, L0, args.per_digit_hint, device, SIZE, batch=args.anchor_batch, amp_on=amp_on)

    del vae_tmp, head_tmp

    # sweep list
    fracs = []
    f = args.min_frac
    while f <= args.max_frac + 1e-9:
        fracs.append(round(f, 4))
        f += args.step_frac

    summary_csv = run_dir / "sweep_summary_test.csv"
    with open(summary_csv, "w", newline="") as fsum:
        csv.writer(fsum).writerow(["frac", "test_base_PSNR", "test_alien_PSNR", "test_base_SSIM", "test_alien_SSIM", "test_base_MSE", "test_alien_MSE", "test_base_Acc", "test_alien_Acc"])

    for frac in fracs:
        base_t, alien_t = train_one_fraction(
            frac=frac, args=args, device=device, amp_on=amp_on, SIZE=SIZE, run_dir=run_dir,
            X0=X0, Y0=Y0, L0=L0, Xt=Xt, Yt=Yt, Lt=Lt,
            train_orders=train_orders, val_idx=val_idx, test_idx=test_idx,
            proto_src=proto_src, cov_src_g=cov_src_g, cov_src_c=cov_src_c,
            label2idx_src=label2idx_src
        )
        with open(summary_csv, "a", newline="") as fsum:
            csv.writer(fsum).writerow([
                frac,
                base_t["PSNR"], alien_t["PSNR"],
                base_t["SSIM"], alien_t["SSIM"],
                base_t["MSE"],  alien_t["MSE"],
                base_t["Acc"],  alien_t["Acc"]
            ])

    log(f"All sweep done. Results in: {run_dir}")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_mlp_recon.py

- Baseline: plain fully-connected (MLP) speckle-to-image reconstruction
- Input/target are both resized to HW x HW (default 128 to keep params manageable)
- Loss: MSE (+ optional L1)
- Logging: history.csv, val_metrics.csv, checkpoints/{last.pt,best.pt}, viz/*
- If val_ratio == 0, evaluate on train split
"""

import os
os.environ["MPLBACKEND"] = "Agg"
os.environ["QT_QPA_PLATFORM"] = "offscreen"

import argparse, re, json, csv, random
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

# ---------------- Utils ----------------
def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True

def choose_device(s: str):
    if s == "auto": return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(s)

def log(msg): print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)

def normalize_per_image(a: np.ndarray) -> np.ndarray:
    a = a.astype(np.float32, copy=False)
    mx, mn = float(a.max()), float(a.min())
    return (a - mn) / (mx - mn + 1e-8)

_bending_pat = re.compile(r"^bending(\d+)(?:_sorted)?$", re.IGNORECASE)

def detect_bending_dirs(root: Path, ids: Optional[List[int]]) -> List[Tuple[int, Path]]:
    out = []
    for p in sorted([d for d in root.iterdir() if d.is_dir()]):
        m = _bending_pat.match(p.name)
        if not m: continue
        i = int(m.group(1))
        if (ids is None) or (i in ids):
            out.append((i, p))
    if not out:
        raise FileNotFoundError(f"No bending folders in {root} for ids={ids or 'ALL'}")
    out.sort(key=lambda t: t[0])
    return out

def find_npy(dirp: Path, name: str) -> Path:
    p = dirp / name
    if p.exists(): return p
    # fallback to closest filename
    cands = list(dirp.glob("*.npy"))
    key = "speckle" if "speckle" in name else ("image" if "image" in name else "label")
    for q in cands:
        if key in q.name.lower():
            return q
    raise FileNotFoundError(f"{name} not found in {dirp}")


# ---------------- Dataset ----------------
class BendConcatDataset(Dataset):
    """
    Read multiple bending folders. In __getitem__, X/Y -> [1,HW,HW].
    Return: x:[1,HW,HW], y:[1,HW,HW], digit(0..9), bend_index
    """
    def __init__(self, pairs: List[Tuple[Path,Path,Path]], per_digit_hint=800,
                 limit_per_bending=0, resize_hw=128):
        self.pairs = pairs
        self.per_digit_hint = per_digit_hint
        self.limit = limit_per_bending
        self.resize_hw = int(resize_hw)

        self.X_files = [p[0] for p in pairs]
        self.Y_files = [p[1] for p in pairs]
        self.L_files = [p[2] for p in pairs]

        self.X_arrs = [None]*len(self.X_files)
        self.Y_arrs = [None]*len(self.Y_files)
        self.L_arrs = [None]*len(self.L_files)

        self.lengths, self.offsets = [], []
        cum = 0
        for _,(_,_,l) in enumerate(pairs):
            L = np.load(l, mmap_mode='r')
            N = min(len(L), self.limit) if self.limit>0 else len(L)
            self.lengths.append(N); self.offsets.append(cum); cum += N
            del L
        self.total = cum

    def __len__(self): return self.total

    def _ensure(self, i):
        if self.X_arrs[i] is None: self.X_arrs[i] = np.load(self.X_files[i], mmap_mode='r')
        if self.Y_arrs[i] is None: self.Y_arrs[i] = np.load(self.Y_files[i], mmap_mode='r')
        if self.L_arrs[i] is None: self.L_arrs[i] = np.load(self.L_files[i], mmap_mode='r')

    def _map(self, idx):
        for fi,off in enumerate(self.offsets):
            if idx < off + self.lengths[fi]:
                return fi, idx - off
        raise IndexError(idx)

    def __getitem__(self, idx):
        fi, k = self._map(idx); self._ensure(fi)
        X,Y,L = self.X_arrs[fi], self.Y_arrs[fi], self.L_arrs[fi]

        x = X[k]; y = Y[k]
        if x.ndim==3 and x.shape[0]==1: x=x[0]
        if y.ndim==3 and y.shape[0]==1: y=y[0]
        x = normalize_per_image(x).astype(np.float32, copy=False)
        y = y.astype(np.float32, copy=False)

        x = torch.from_numpy(x).unsqueeze(0)
        y = torch.from_numpy(y).unsqueeze(0)

        if self.resize_hw > 0:
            s = (self.resize_hw, self.resize_hw)
            x = F.interpolate(x.unsqueeze(0), size=s, mode="bilinear", align_corners=False).squeeze(0)
            y = F.interpolate(y.unsqueeze(0), size=s, mode="bilinear", align_corners=False).squeeze(0)

        d = int(((int(L[k]) - 1) % (self.per_digit_hint * 10)) // self.per_digit_hint)
        return x, y, torch.tensor(d, dtype=torch.long), torch.tensor(fi, dtype=torch.long)


# ---------------- MLP model ----------------
class MLPRecon(nn.Module):
    """
    Flatten -> Linear ... -> Linear -> reshape to [B,1,H,W] -> sigmoid
    widths: list of hidden layer sizes, e.g. [4096,4096]
    """
    def __init__(self, hw=128, widths=(4096,4096), dropout=0.1, use_bn=True, activation="gelu"):
        super().__init__()
        self.hw = int(hw)
        self.in_dim = self.hw * self.hw
        self.out_dim = self.in_dim

        act = nn.GELU() if activation.lower()=="gelu" else nn.ReLU(inplace=True)
        layers = []
        last = self.in_dim
        for w in widths:
            layers.append(nn.Linear(last, w))
            if use_bn: layers.append(nn.BatchNorm1d(w))
            layers.append(act)
            if dropout>0: layers.append(nn.Dropout(dropout))
            last = w
        layers.append(nn.Linear(last, self.out_dim))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        b = x.size(0)
        flat = x.view(b, -1)               # [B, H*W]
        y = self.mlp(flat)                 # [B, H*W]
        y = y.view(b, 1, self.hw, self.hw)
        return torch.sigmoid(y)


# ---------------- Metrics & Viz ----------------
@torch.no_grad()
def ssim_2d(pred, target, sigma=1.5, r=11):
    pad = r // 2
    xs = torch.arange(r, device=pred.device, dtype=pred.dtype) - pad
    g = torch.exp(-(xs**2)/(2*sigma**2))
    g = (g/g.sum()).view(1,1,-1,1)
    ker = g @ g.transpose(-2,-1)
    yp = F.pad(target, (pad,pad,pad,pad), "reflect")
    pp = F.pad(pred,   (pad,pad,pad,pad), "reflect")
    mu1 = F.conv2d(pp, ker); mu2 = F.conv2d(yp, ker)
    s1  = F.conv2d(pp*pp, ker) - mu1*mu1
    s2  = F.conv2d(yp*yp, ker) - mu2*mu2
    s12 = F.conv2d(pp*yp, ker) - mu1*mu2
    C1 = 0.01**2; C2 = 0.03**2
    ssim = ((2*mu1*mu2+C1)*(2*s12+C2))/((mu1*mu1+mu2*mu2+C1)*(s1+s2+C2))
    return ssim.flatten(1).mean(1)

@torch.no_grad()
def eval_recon_metrics(model, loader, device):
    if loader is None:
        return dict(MSE=float("nan"), PSNR=float("nan"), SSIM=float("nan"), N=0)
    model.eval()
    s_mse=s_psnr=s_ssim=0.0; n=0
    for x,y,_,_ in loader:
        x=x.to(device); y=y.to(device)
        pred = model(x).clamp(0,1)
        mse = ((pred-y)**2).flatten(1).mean(1)
        psn = 10.0*torch.log10(1.0/mse.clamp_min(1e-12))
        ssm = ssim_2d(pred,y)
        s_mse += mse.sum().item(); s_psnr += psn.sum().item(); s_ssim += ssm.sum().item()
        n += x.size(0)
    if n==0:
        return dict(MSE=float("nan"), PSNR=float("nan"), SSIM=float("nan"), N=0)
    return dict(MSE=s_mse/n, PSNR=s_psnr/n, SSIM=s_ssim/n, N=n)

@torch.no_grad()
def save_recon_grid(model, loader, device, save_path: Path, n=8):
    if loader is None: return
    model.eval()
    xs=[]; ys=[]
    for x,y,_,_ in loader:
        xs.append(x); ys.append(y)
        if len(torch.cat(xs))>=n: break
    if not xs: return
    x = torch.cat(xs)[:n].to(device)
    y = torch.cat(ys)[:n].to(device)
    pred = model(x).clamp(0,1)
    x = x.cpu().numpy(); y=y.cpu().numpy(); pred=pred.cpu().numpy()

    cols=3; rows=min(n, x.shape[0])
    fig, axes = plt.subplots(rows, cols, figsize=(cols*2.6, rows*2.6))
    if rows==1: axes = axes[None,...]
    for i in range(rows):
        axes[i,0].imshow(x[i,0], cmap='gray'); axes[i,0].set_title("Speckle")
        axes[i,1].imshow(pred[i,0], cmap='gray', vmin=0, vmax=1); axes[i,1].set_title("Recon")
        axes[i,2].imshow(y[i,0], cmap='gray', vmin=0, vmax=1);  axes[i,2].set_title("GT")
        for j in range(cols): axes[i,j].axis('off')
    fig.tight_layout(); save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150); plt.close(fig)


# ---------------- Main ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, required=True)
    ap.add_argument("--bending_ids", type=str, default="0")
    ap.add_argument("--per_digit_hint", type=int, default=800)
    ap.add_argument("--limit_per_bending", type=int, default=0)
    ap.add_argument("--resize_hw", type=int, default=128, help="Use 128 for MLP to keep params manageable; 256 works with smaller widths.")

    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--val_ratio", type=float, default=0.1)
    ap.add_argument("--test_ratio", type=float, default=0.1)

    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--w_l1", type=float, default=0.0)
    ap.add_argument("--widths", type=str, default="4096,4096", help="comma-separated hidden sizes, e.g. '4096,4096' or '2048,2048,2048'")
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--use_bn", action="store_true")
    ap.add_argument("--activation", type=str, default="gelu", choices=["gelu","relu"])

    ap.add_argument("--device", type=str, default="auto", choices=["auto","cpu","cuda"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--pin_memory", action="store_true")
    ap.add_argument("--save_dir", type=str, default="runs/mlp_recon")

    ap.add_argument("--val_vis_every", type=int, default=10)
    ap.add_argument("--viz_samples", type=int, default=8)
    args = ap.parse_args()

    set_seed(args.seed)
    device = choose_device(args.device)
    root = Path(args.data_root)
    ids = [int(s) for s in args.bending_ids.split(",") if s.strip().isdigit()] if args.bending_ids else None
    bends = detect_bending_dirs(root, ids)

    pairs=[]
    for bid, p in bends:
        x = find_npy(p, "speckles_sorted.npy")
        y = find_npy(p, "images_sorted.npy")
        l = find_npy(p, "labels_sorted.npy")
        log(f"Use bending{bid}: {x.name}, {y.name}, {l.name}")
        pairs.append((x,y,l))

    ds = BendConcatDataset(pairs,
                           per_digit_hint=args.per_digit_hint,
                           limit_per_bending=args.limit_per_bending,
                           resize_hw=args.resize_hw)
    x0,_,_,_ = ds[0]
    H,W = x0.shape[-2], x0.shape[-1]
    assert (H,W)==(args.resize_hw,args.resize_hw)
    log(f"Total samples: {len(ds)}, HW={H}x{W}")

    # splits
    N=len(ds)
    n_val=int(N*args.val_ratio)
    n_test=int(N*args.test_ratio)
    n_train=N-n_val-n_test
    tr,va,te = random_split(ds, [n_train,n_val,n_test],
                            generator=torch.Generator().manual_seed(args.seed))
    log(f"Split train:{n_train}  val:{n_val}  test:{n_test}")

    dl_kw = dict(batch_size=args.batch_size,
                 num_workers=args.num_workers,
                 pin_memory=(args.pin_memory and device.type=="cuda"),
                 drop_last=False)
    if args.num_workers > 0:
        dl_kw["persistent_workers"] = True
    train_loader = DataLoader(tr, shuffle=True, **dl_kw)
    val_loader   = DataLoader(va, shuffle=False, **dl_kw) if n_val>0 else None
    test_loader  = DataLoader(te, shuffle=False, **dl_kw) if n_test>0 else None

    widths = [int(s) for s in args.widths.split(",") if s.strip().isdigit()]
    model = MLPRecon(hw=H, widths=widths, dropout=args.dropout,
                     use_bn=args.use_bn, activation=args.activation).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    save_dir = Path(args.save_dir); (save_dir/"checkpoints").mkdir(parents=True, exist_ok=True)
    (save_dir/"viz").mkdir(parents=True, exist_ok=True)
    hist = save_dir/"history.csv"
    if not hist.exists():
        with open(hist,"w",newline="") as f:
            csv.writer(f).writerow(["epoch","train_loss","eval_loss","eval_split"])
    vcsv = save_dir/"val_metrics.csv"
    if not vcsv.exists():
        with open(vcsv,"w",newline="") as f:
            csv.writer(f).writerow(["epoch","split","MSE","PSNR","SSIM","N"])

    best_val = float("inf")

    for ep in range(1, args.epochs+1):
        model.train()
        sL=0.0; n=0
        for x,y,_,_ in tqdm(train_loader, desc=f"Epoch {ep}/{args.epochs}"):
            x=x.to(device); y=y.to(device)
            pred = model(x)
            loss = F.mse_loss(pred, y)
            if args.w_l1 > 0:
                loss = loss + args.w_l1 * F.l1_loss(pred, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            b=x.size(0); sL+=loss.item()*b; n+=b
        train_loss = sL/max(1,n)

        # eval
        eval_loader = val_loader if val_loader is not None else train_loader
        eval_split  = "val" if val_loader is not None else "train"
        model.eval()
        sE=0.0; mN=0
        with torch.no_grad():
            for x,y,_,_ in eval_loader:
                x=x.to(device); y=y.to(device)
                pred = model(x)
                loss = F.mse_loss(pred, y)
                if args.w_l1 > 0:
                    loss = loss + args.w_l1 * F.l1_loss(pred, y)
                b=x.size(0); sE+=loss.item()*b; mN+=b
        eval_loss = sE/max(1,mN)
        log(f"Epoch {ep:03d} | train={train_loss:.4f} | {eval_split}={eval_loss:.4f}")

        with open(hist,"a",newline="") as f:
            csv.writer(f).writerow([ep, train_loss, eval_loss, eval_split])

        # checkpoints
        torch.save({"epoch":ep,"model":model.state_dict()},
                   save_dir/"checkpoints"/"last.pt")
        if eval_loss < best_val:
            best_val = eval_loss
            torch.save({"epoch":ep,"model":model.state_dict()},
                       save_dir/"checkpoints"/"best.pt")

        # periodic metrics & viz
        if args.val_vis_every>0 and (ep % args.val_vis_every == 0):
            m = eval_recon_metrics(model, eval_loader, device)
            with open(vcsv,"a",newline="") as f:
                csv.writer(f).writerow([ep, eval_split, m["MSE"], m["PSNR"], m["SSIM"], m["N"]])
            log(f"[Eval {eval_split} @ ep{ep}] MSE={m['MSE']:.6f}, PSNR={m['PSNR']:.2f}, SSIM={m['SSIM']:.4f}")
            save_recon_grid(model, eval_loader, device,
                            save_dir/"viz"/f"{eval_split}_epoch_{ep:03d}_recon.png",
                            n=args.viz_samples)

    if test_loader is not None:
        m = eval_recon_metrics(model, test_loader, device)
        log(f"[Test] MSE={m['MSE']:.6f}, PSNR={m['PSNR']:.2f}, SSIM={m['SSIM']:.4f} (N={m['N']})")

    meta = {"finished_at": datetime.now().isoformat(timespec="seconds"),
            "args": vars(args),
            "best_eval_loss": best_val}
    with open(save_dir/"metrics.json","w") as f: json.dump(meta, f, indent=2)
    log(f"Saved to {save_dir}")

if __name__ == "__main__":
    main()

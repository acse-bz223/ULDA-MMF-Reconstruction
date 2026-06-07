#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_rtmnet_sino_paper_repro.py  —— 作者设置对齐版

- 输入/GT：sinogram 域（默认 [θ, s] -> [B,1,θ,s]，最后一维 = s）
- 模型：RTMnet（SF_1D + RTpad），在 s 轴做 1D FFT 滤波
- 损失：
  * 主损失：--loss {l1,l2,npcc,vae}（默认 l2）
  * 强度配准：对每样本归一化 + output 按总能量缩放
  * TV 正则：--tv_w（默认 0）
  * 图像域辅助：--sinoproject α（默认 None 关闭），用 torch_radon 的 FBP
- 优化：Adam(lr, weight_decay=3e-5) + StepLR(step=1, gamma=0.95)（--lr_decay 开启）
"""

import os, csv, json, argparse, time, math
from pathlib import Path
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from torch.fft import rfft, irfft
# pip install einops torch-radon

# -------------------------
# 网络（作者版）
# -------------------------
class RTpad(nn.Module):
    def __init__(self, pad_width, if_zero=False):
        super().__init__()
        self.pad_width = pad_width
        self.if_zero = if_zero
    def forward(self, tensor):
        # tensor: [B,C,θ,s]
        if self.if_zero:
            tensor = F.pad(tensor, pad=(self.pad_width, self.pad_width, 0, 0),
                           mode='constant', value=0)
        pad_left  = torch.flip(tensor[:, :, -self.pad_width:, :], dims=[3])
        pad_right = torch.flip(tensor[:, :, 0:self.pad_width, :],   dims=[3])
        tensor_pad1 = torch.cat([pad_left, tensor, pad_right], dim=2)
        return tensor_pad1

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, pad_width=1, RT=False):
        super().__init__()
        if RT:
            self.conv1 = nn.Sequential(
                RTpad(pad_width=pad_width, if_zero=True),
                nn.Conv2d(in_channels, out_channels, kernel_size, padding=0, stride=1)
            )
            self.conv2 = nn.Sequential(
                RTpad(pad_width=1, if_zero=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=0, stride=1)
            )
        else:
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=pad_width, stride=1)
            self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, stride=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.LeakyReLU(inplace=True)
    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        return x

class Filter_1D(nn.Module):
    def __init__(self, filter_h, filter_w, channel=1):
        super().__init__()
        self.scale = 1.0/max(1,channel)
        self.pad = filter_w//2
        self.complex_weight = nn.Parameter(
            torch.randn(channel, filter_h, filter_w + 1, 2, dtype=torch.float32) * self.scale
        )
    def forward(self, x):
        # x: [B,C,θ,s]  —— 在最后一维 s 上做 1D FFT
        x = F.pad(x, pad=(self.pad, self.pad, 0, 0), mode='constant', value=0)
        X = rfft(x, dim=-1, norm='ortho')                       # [B,C,θ,s//2+1] complex
        W = torch.view_as_complex(self.complex_weight.contiguous())  # [C,θ,s//2+1]
        assert W.shape[1:3] == X.shape[2:4], f"filter shape {W.shape} vs input {X.shape}"
        Y = X * W
        y = irfft(Y, dim=-1, norm='ortho')                      # [B,C,θ, s+2*pad]
        return y[:, :, :, self.pad:3*self.pad]                  # 裁回原幅度

class SF_1D(nn.Module):
    def __init__(self, in_channels, out_channels, filter_h, filter_w, RT_pad=False):
        super().__init__()
        self.iniF = nn.Conv2d(in_channels, out_channels//2, kernel_size=3, stride=1, padding=1)
        self.F    = Filter_1D(filter_h, filter_w, out_channels//2)
        self.S    = DoubleConv(in_channels, out_channels//2, kernel_size=3, pad_width=1, RT=RT_pad)
    def forward(self, x):
        xf0 = self.iniF(x)  # [B, out/2, θ, s]
        xf  = self.F(xf0)   # 1D-FFT filter on s-axis
        xs  = self.S(x)
        return torch.cat([xs, xf], dim=1)

class RTMnet(nn.Module):
    def __init__(self, RT_pad=False, dim=16, filter_size=256):
        super().__init__()
        S = filter_size  # 最后一维长度（s）
        self.conv0 = SF_1D(1,     dim,   filter_h=S,   filter_w=S,   RT_pad=RT_pad)
        self.pool1 = nn.MaxPool2d(2)
        self.conv1 = SF_1D(dim,   dim*2, filter_h=S//2,filter_w=S//2,RT_pad=RT_pad)
        self.pool2 = nn.MaxPool2d(2)
        self.conv2 = SF_1D(dim*2, dim*4, filter_h=S//4,filter_w=S//4,RT_pad=RT_pad)
        self.pool3 = nn.MaxPool2d(2)
        self.conv3 = SF_1D(dim*4, dim*4, filter_h=S//8,filter_w=S//8,RT_pad=RT_pad)  # bottleneck
        self.up3   = nn.ConvTranspose2d(dim*4, dim*4, kernel_size=2, stride=2)
        self.upconv3 = DoubleConv(dim*8, dim*2, kernel_size=3, pad_width=1, RT=RT_pad)
        self.up2   = nn.ConvTranspose2d(dim*2, dim*2, kernel_size=2, stride=2)
        self.upconv2 = DoubleConv(dim*4, dim,   kernel_size=3, pad_width=1, RT=RT_pad)
        self.up1   = nn.ConvTranspose2d(dim, dim, kernel_size=2, stride=2)
        self.upconv1 = DoubleConv(dim*2, dim//2, kernel_size=3, pad_width=1, RT=RT_pad)
        self.out   = DoubleConv(dim//2, 1, kernel_size=3, pad_width=1, RT=RT_pad)
    def forward(self, x):
        x0 = self.conv0(x)
        x1 = self.conv1(self.pool1(x0))
        x2 = self.conv2(self.pool2(x1))
        x3 = self.conv3(self.pool3(x2))
        x  = self.up3(x3);  x = self.upconv3(torch.cat([x2, x], dim=1))
        x  = self.up2(x);   x = self.upconv2(torch.cat([x1, x], dim=1))
        x  = self.up1(x);   x = self.upconv1(torch.cat([x0, x], dim=1))
        x  = self.out(x)
        return x

# -------------------------
# 工具/指标/可视化
# -------------------------
def set_seed(seed=42):
    import random
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True

def log(msg): print(msg, flush=True)

def to01(x: np.ndarray):
    x = x.astype(np.float32, copy=False)
    mx = float(x.max())
    return x / mx if mx > 0 else x

@torch.no_grad()
def psnr_from_mse(mse_t):  # [B]
    return 10.0 * torch.log10(1.0 / mse_t.clamp_min(1e-12))

@torch.no_grad()
def ssim_2d(yhat, y, sigma=1.5, r=11):
    pad=r//2
    xs = torch.arange(r, device=y.device, dtype=y.dtype) - pad
    g  = torch.exp(-(xs**2)/(2*sigma**2)); g=(g/g.sum()).view(1,1,-1,1)
    ker= g @ g.transpose(-2,-1)
    yp = F.pad(y,    (pad,pad,pad,pad), 'reflect')
    hp = F.pad(yhat, (pad,pad,pad,pad), 'reflect')
    mu1 = F.conv2d(hp, ker); mu2 = F.conv2d(yp, ker)
    s1  = F.conv2d(hp*hp, ker) - mu1*mu1
    s2  = F.conv2d(yp*yp, ker) - mu2*mu2
    s12 = F.conv2d(hp*yp, ker) - mu1*mu2
    C1=0.01**2; C2=0.03**2
    ssim = ((2*mu1*mu2+C1)*(2*s12+C2))/((mu1*mu1+mu2*mu2+C1)*(s1+s2+C2))
    return ssim.flatten(1).mean(1)

def save_grid_sino(x_in, x_pred, x_gt, save_path, n=8):
    B = min(n, x_in.shape[0])
    fig, axes = plt.subplots(B, 3, figsize=(10, 3*B))
    for i in range(B):
        axes[i,0].imshow(x_in[i,0], cmap='gray', aspect='auto');  axes[i,0].set_title("Input sino"); axes[i,0].axis('off')
        axes[i,1].imshow(x_pred[i,0], cmap='gray', aspect='auto'); axes[i,1].set_title("Pred sino");  axes[i,1].axis('off')
        axes[i,2].imshow(x_gt[i,0],  cmap='gray', aspect='auto'); axes[i,2].set_title("GT sino");    axes[i,2].axis('off')
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150); plt.close(fig)

# -------------------------
# 数据集（bend0）
# -------------------------
class RadonMMFDataset(Dataset):
    """
    读取 bend=0 的 sinogram：speckles_sino.npy (输入), images_sino.npy (GT)
    默认不转置：磁盘 [θ,s] -> 网络 [B,1,θ,s]，最后一维=s（与论文一致）。
    如你坚持要转置，请用 --transpose_sino，但会把滤波轴变成 θ（不建议）。
    """
    def __init__(self, data_root, bend=0, split="train",
                 transpose_sino=False, map_digit=True, seed=42):
        folder = os.path.join(data_root, f"bending{bend}_sino")
        self.X = np.load(os.path.join(folder, "speckles_sino.npy"), mmap_mode="r")
        self.Y = np.load(os.path.join(folder, "images_sino.npy"),   mmap_mode="r")
        L = np.load(os.path.join(folder, "labels.npy"),             mmap_mode="r")

        self.transpose_sino = transpose_sino
        self.lab = self._map_digit(L) if map_digit else L.astype(np.int64)

        N = len(self.lab)
        rng = np.random.RandomState(seed)
        idx = np.arange(N); rng.shuffle(idx)
        n_tr = int(0.8*N); n_v = int(0.1*N)
        if split=="train": self.ids = idx[:n_tr]
        elif split=="val": self.ids = idx[n_tr:n_tr+n_v]
        else: self.ids = idx[n_tr+n_v:]

    @staticmethod
    def _map_digit(global_idx):
        global_idx = np.array(global_idx)
        zero_based = global_idx - 1
        within_sequence = zero_based % 8000
        return (within_sequence // 800).astype(np.int64)

    def __len__(self): return len(self.ids)

    def __getitem__(self, i):
        k = self.ids[i]
        x = to01(self.X[k]); y = to01(self.Y[k])
        if self.transpose_sino:
            x = x.T
            y = y.T
        x = np.expand_dims(x.astype(np.float32), 0)  # [1,θ,s] or [1,s,θ]
        y = np.expand_dims(y.astype(np.float32), 0)
        return torch.from_numpy(x), torch.from_numpy(y), 0

# -------------------------
# 损失/正则/图像域项
# -------------------------
class NPCCLoss(nn.Module):
    """1 - Pearson correlation"""
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps
    def forward(self, pred, target):
        B = pred.shape[0]
        p = pred.flatten(1); t = target.flatten(1)
        p = p - p.mean(dim=1, keepdim=True)
        t = t - t.mean(dim=1, keepdim=True)
        num = (p*t).sum(dim=1)
        den = torch.sqrt((p*p).sum(dim=1)* (t*t).sum(dim=1) + self.eps)
        corr = num / den
        return (1.0 - corr).mean()

def tv_gradients(x):
    # x: [B,1,θ,s]
    dx = x[:, :, :, 1:] - x[:, :, :, :-1]
    dy = x[:, :, 1:, :] - x[:, :, :-1, :]
    # pad回原尺寸
    dx = F.pad(dx, (0,1,0,0))
    dy = F.pad(dy, (0,0,0,1))
    return dx, dy

class VAELossWrap(nn.Module):
    """作者：用预训练 VAE 的特征/重建误差作为感知项；需 --vae_ckpt"""
    def __init__(self, vae, lambda_latent=2.0, lambda_recon=0.05):
        super().__init__()
        self.vae = vae.eval()
        for p in self.vae.parameters():
            p.requires_grad = False
        self.lz = lambda_latent
        self.lr = lambda_recon
        self.mse = nn.MSELoss()
    def forward(self, pred, target):
        # 送入 VAE，累积两个项
        with torch.no_grad():
            z_t, rec_t = self.vae.encode_and_decode(target)  # 需你的 VAE 实现提供该接口
        z_p, rec_p = self.vae.encode_and_decode(pred)
        loss_lat = self.mse(z_p, z_t)
        loss_rec = self.mse(rec_p, rec_t)
        return self.lz*loss_lat + self.lr*loss_rec

# -------------------------
# 评估（sinogram 域）
# -------------------------
@torch.no_grad()
def evaluate(model, loader, device, transpose_sino=False):
    model.eval()
    s_mse=s_psnr=s_ssim=0.0; n=0
    for x,y,_ in loader:
        x,y=x.to(device),y.to(device)
        # 强度配准 & 归一化与训练一致
        y = y / (y.amax(dim=(-1,-2), keepdim=True).clamp_min(1e-8))
        pred = model(x)
        scale = (y.sum(dim=(-2,-1), keepdim=True) /
                 pred.sum(dim=(-2,-1), keepdim=True).clamp_min(1e-8))
        pred = pred * scale
        mse  = ((pred - y)**2).flatten(1).mean(1)
        psnr = psnr_from_mse(mse)
        ssim = ssim_2d(pred, y)
        s_mse += float(mse.sum()); s_psnr += float(psnr.sum()); s_ssim += float(ssim.sum()); n += x.size(0)
    return dict(MSE=s_mse/n, PSNR=s_psnr/n, SSIM=s_ssim/n, N=n)

# -------------------------
# 训练
# -------------------------
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--data_root",   type=str, required=True)
    ap.add_argument("--bend",        type=int, default=0)
    ap.add_argument("--epochs",      type=int, default=16)
    ap.add_argument("--batch",       type=int, default=32)
    ap.add_argument("--lr",          type=float, default=1e-3)
    ap.add_argument("--wd",          type=float, default=3e-5)
    ap.add_argument("--save_dir",    type=str, default="runs/rtmnet_sino_paper_repro")
    ap.add_argument("--transpose_sino", action="store_true", help="不建议；会把滤波轴变成 θ")
    ap.add_argument("--viz_every",   type=int, default=4)
    ap.add_argument("--dim",         type=int, default=16)
    ap.add_argument("--loss",        type=str, default="l2", choices=["l1","l2","npcc"])
    ap.add_argument("--tv_w",        type=float, default=0.0)
    ap.add_argument("--sinoproject", type=float, default=None, help="图像域项权重 α（None 关闭）")
    ap.add_argument("--lr_decay",    type=int, default=1, help="1=StepLR(gamma=0.95)")
    ap.add_argument("--amp",         action="store_true")
    args=ap.parse_args()

    set_seed(42)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")

    # 数据
    train_set=RadonMMFDataset(args.data_root,args.bend,"train",transpose_sino=args.transpose_sino)
    val_set  =RadonMMFDataset(args.data_root,args.bend,"val",  transpose_sino=args.transpose_sino)
    test_set =RadonMMFDataset(args.data_root,args.bend,"test", transpose_sino=args.transpose_sino)

    train_loader=DataLoader(train_set,batch_size=args.batch,shuffle=True,num_workers=0,pin_memory=True, drop_last=True)
    val_loader  =DataLoader(val_set,  batch_size=args.batch,shuffle=False,num_workers=0,pin_memory=True, drop_last=False)
    test_loader =DataLoader(test_set, batch_size=args.batch,shuffle=False,num_workers=0,pin_memory=True, drop_last=False)

    # 模型
    H, W = train_set[0][0].shape[-2], train_set[0][0].shape[-1]
    # 若不转置：[θ,s] => W=s（最后一维）；若转置：[s,θ] => W=θ（会滤到 θ 上）
    model = RTMnet(RT_pad=True, dim=args.dim, filter_size=W).to(device)

    # 损失
    if args.loss == "l1":
        main_crit = nn.L1Loss()
    elif args.loss == "l2":
        main_crit = nn.MSELoss()
    elif args.loss == "npcc":
        main_crit = NPCCLoss()
    else:
        main_crit = NPCCLoss()

    opt   = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    if args.lr_decay:
        sch = torch.optim.lr_scheduler.StepLR(opt, step_size=1, gamma=0.95)
    scaler = torch.cuda.amp.GradScaler(enabled=(args.amp and device.type=="cuda"))

    # torch_radon（图像域项）
    if args.sinoproject is not None:
        try:
            from torch_radon import Radon
        except ImportError as e:
            raise ImportError("需要 torch-radon：pip install torch-radon") from e
        angles = np.linspace(0, np.pi, W, endpoint=False)  # W 对应 s 或 θ
        RTi = Radon(H, angles, clip_to_circle=True).to(device)  # 用于 GT 尺寸
        RTs = Radon(H, angles, clip_to_circle=True).to(device)  # 用于 pred；这里 H 相同即可

    run_dir=Path(args.save_dir)/time.strftime("%Y%m%d_%H%M%S")
    (run_dir/"checkpoints").mkdir(parents=True,exist_ok=True)
    (run_dir/"viz").mkdir(parents=True,exist_ok=True)
    with open(run_dir/"history.csv","w",newline="") as f:
        csv.writer(f).writerow(["epoch","loss","main","tv","img"])
    with open(run_dir/"metrics.csv","w",newline="") as f:
        csv.writer(f).writerow(["epoch","split","MSE","PSNR","SSIM","N"])

    best_psnr=-1e9

    for ep in range(1,args.epochs+1):
        model.train(); sL=sMain=sTV=sIMG=0.0; n=0
        for x,y,_ in train_loader:
            x = x.to(device); y = y.to(device)

            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", enabled=(args.amp and device.type=="cuda")):
                pred = model(x)

                # ------- 强度配准（作者做法） -------
                y_norm = y / (y.amax(dim=(-1,-2), keepdim=True).clamp_min(1e-8))
                scale  = (y_norm.sum(dim=(-2,-1), keepdim=True) /
                          pred.sum(dim=(-2,-1), keepdim=True).clamp_min(1e-8))
                pred_s = pred * scale

                # ------- 主损失 -------
                main_loss = main_crit(pred_s, y_norm)

                # ------- TV 正则 -------
                if args.tv_w > 0:
                    dx_p, dy_p = tv_gradients(pred_s)
                    dx_y, dy_y = tv_gradients(y_norm)
                    tv_loss = F.l1_loss(dx_p, dx_y) + F.l1_loss(dy_p, dy_y)
                    tv_loss = args.tv_w * tv_loss
                else:
                    tv_loss = pred_s.new_zeros(())

                # ------- 图像域辅助 -------
                if args.sinoproject is not None:
                    # ramp filter + backprojection（作者风格）
                    out_f = RTs.filter_sinogram(pred_s)
                    img_f = RTi.filter_sinogram(y_norm)
                    out_bp = RTs.backprojection(out_f)
                    img_bp = RTi.backprojection(img_f)
                    out_bp = out_bp / out_bp.amax(dim=(-1,-2), keepdim=True).clamp_min(1e-8)
                    img_bp = img_bp / img_bp.amax(dim=(-1,-2), keepdim=True).clamp_min(1e-8)
                    img_loss = F.mse_loss(out_bp, img_bp) * float(args.sinoproject)
                else:
                    img_loss = pred_s.new_zeros(())

                loss = main_loss + tv_loss + img_loss

            scaler.scale(loss).backward()
            scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(opt); scaler.update()

            bs = x.size(0); n += bs
            sL   += float(loss.item())     * bs
            sMain+= float(main_loss.item())* bs
            sTV  += float(tv_loss.item())  * bs
            sIMG += float(img_loss.item()) * bs

        if args.lr_decay:
            sch.step()

        epoch_loss = dict(loss=sL/n, main=sMain/n, tv=sTV/n, img=sIMG/n)
        print(f"[Epoch {ep}] {epoch_loss}", flush=True)
        with open(run_dir/"history.csv","a",newline="") as f:
            csv.writer(f).writerow([ep, epoch_loss['loss'], epoch_loss['main'], epoch_loss['tv'], epoch_loss['img']])

        # 验证（与训练同口径：sinogram 域 + 强度配准）
        valm=evaluate(model,val_loader,device,transpose_sino=args.transpose_sino)
        with open(run_dir/"metrics.csv","a",newline="") as f:
            csv.writer(f).writerow([ep,"val",valm["MSE"],valm["PSNR"],valm["SSIM"],valm["N"]])

        # 可视化
        if ep % args.viz_every == 0:
            model.eval()
            with torch.no_grad():
                xb, yb, _ = next(iter(val_loader))
                xb = xb.to(device); yb = yb.to(device)
                pb = model(xb)
                yb_n = yb / (yb.amax(dim=(-1,-2), keepdim=True).clamp_min(1e-8))
                sc   = (yb_n.sum(dim=(-2,-1), keepdim=True) /
                        pb.sum(dim=(-2,-1), keepdim=True).clamp_min(1e-8))
                pb_s = pb * sc
            save_grid_sino(
                xb.cpu().numpy(), pb_s.cpu().numpy(), yb_n.cpu().numpy(),
                run_dir/"viz"/f"ep{ep:03d}.png", n=min(8, xb.size(0))
            )

        # 保存 checkpoint（沿用你的标准，用 val PSNR 选 best）
        if valm["PSNR"]>best_psnr:
            best_psnr=valm["PSNR"]; torch.save(model.state_dict(),run_dir/"checkpoints"/"best.pth")
        torch.save(model.state_dict(),run_dir/"checkpoints"/"last.pth")

    # 测试
    testm=evaluate(model,test_loader,device,transpose_sino=args.transpose_sino)
    with open(run_dir/"metrics.csv","a",newline="") as f:
        csv.writer(f).writerow([args.epochs,"test",testm["MSE"],testm["PSNR"],testm["SSIM"],testm["N"]])
    with open(run_dir/"done.json","w") as f:
        json.dump({"finished_at":time.strftime("%Y-%m-%d %H:%M:%S")}, f, indent=2)
    print(f"[Test] {testm}")
    print(f"Saved to {run_dir}")

if __name__=="__main__":
    main()

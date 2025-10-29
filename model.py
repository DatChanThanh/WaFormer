

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import math, torch, torch.nn as nn
from torch.nn import functional as F
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.utils.data import DataLoader, random_split, Dataset
from torchvision import datasets, transforms
import pathlib
from tqdm import tqdm
import time
import matplotlib.pyplot as plt
import os
import numpy as np
from ptflops import get_model_complexity_info


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using", device)



train_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) 
])
val_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) 
])


train_dir = pathlib.Path('path/to/your/train')
val_dir = pathlib.Path('path/to/your/test')

train_dataset = datasets.ImageFolder(root=str(train_dir), transform=train_transform)
val_dataset = datasets.ImageFolder(root=str(val_dir), transform=val_transform)

DataLoader cho huấn luyện và test
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True,
                          num_workers=4, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False,
                        num_workers=4, pin_memory=True)

class_names = full_dataset.classes
num_classes = len(class_names)

print(f"Classes: {class_names}")
print(f"Training samples: {train_size}, Validation samples: {val_size}")
print(f"Test samples: {len(val_dataset)}")



def count_params(module, trainable_only=True):
    if trainable_only:
        return sum(p.numel() for p in module.parameters() if p.requires_grad)
    return sum(p.numel() for p in module.parameters())


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep) * random_tensor

# ------------------------------------------------------------
# RMSNorm2d (channelwise) for (B,C,H,W)
# ------------------------------------------------------------
class RMSNorm2d(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.pow(2).mean(dim=1, keepdim=True).add_(self.eps).rsqrt_()
        x = x * norm
        return x * self.weight.view(1, -1, 1, 1)


class ADA(nn.Module):
    def __init__(self, c: int, k: int = 3, dilation: int = 1):
        super().__init__()
        p = (k // 2) * dilation
        self.dw_iso = nn.Conv2d(c, c, k, padding=p, dilation=dilation, groups=c, bias=False)
        self.dw_v   = nn.Conv2d(c, c, (k, 1), padding=(p, 0), dilation=dilation, groups=c, bias=False)
        self.dw_h   = nn.Conv2d(c, c, (1, k), padding=(0, p), dilation=dilation, groups=c, bias=False)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (self.dw_iso(x) + self.dw_v(x) + self.dw_h(x)) /3

class ConvLocalBlock(nn.Module):
    def __init__(self, dim: int, k: int = 5, dilation: int = 1, drop_path: float = 0.0):
        super().__init__()
        self.norm = RMSNorm2d(dim)
        self.dw = ADA(dim, k=k, dilation=dilation)
        self.pw = nn.Conv2d(dim, dim, 1, bias=False)
        self.act = nn.SiLU(inplace=True)
        self.drop = DropPath(drop_path)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s = x
        x = self.norm(x)
        x = self.dw(x)
        x = self.act(self.pw(x))
        return s + self.drop(x)


class WPE(nn.Module):
    """Learnable monotone warp over [0,1] via piecewise-linear spline with K knots.
    For each head h, learn segment lengths (positive, normalized) → cumulative knots y_k in [0,1].
    """
    def __init__(self, n_heads: int, K: int = 16):
        super().__init__()
        assert K >= 2
        self.nh = n_heads
        self.K = K
        # Learn K-1 positive segment lengths per head
        self.segments = nn.Parameter(torch.zeros(n_heads, K - 1))
        nn.init.constant_(self.segments, 0.0)
        # Optional temperature to control smoothness of warp usage in angles
        self.log_scale = nn.Parameter(torch.zeros(n_heads))  # scales the effective length multiplier

    def forward(self, L: int, device, dtype) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return:
        u: warped positions per head, shape (h, L) in [0,1]
        eff_len: per-head effective length scaling (>=0), shape (h,)
        """
        if L <= 1:
            u = torch.zeros(self.nh, max(L,1), device=device, dtype=dtype)
            eff_len = F.softplus(self.log_scale).to(dtype=dtype)
            return u, eff_len
        # base uniform x in [0,1]
        x = torch.linspace(0.0, 1.0, steps=L, device=device, dtype=dtype)  # (L,)
        # segment lengths → positive then normalize
        seg = F.softplus(self.segments) + 1e-6                              # (h, K-1)
        seg = seg / seg.sum(dim=1, keepdim=True)
        # cumulative knots y (h, K); y[:,0]=0, y[:,-1]=1
        y0 = torch.zeros(self.nh, 1, device=device, dtype=dtype)
        y = torch.cat([y0, torch.cumsum(seg, dim=1)], dim=1)               # (h, K)
        # interpolate u(x) piecewise-linearly between uniform x-knots
        t = x * (self.K - 1)
        j = torch.clamp(t.floor().long(), max=self.K - 2)                   # (L,)
        a = (t - j).to(dtype)                                              # (L,)
        j_exp = j.unsqueeze(0).expand(self.nh, -1)                          # (h, L)
        y_left  = torch.gather(y, 1, j_exp)                                 # (h, L)
        y_right = torch.gather(y, 1, j_exp + 1)                             # (h, L)
        u = (1.0 - a) * y_left + a * y_right                                # (h, L)
        eff_len = F.softplus(self.log_scale).to(dtype=dtype)                # (h,)
        return u, eff_len

# ------------------------------------------------------------
# Axial SDPA with WaRP-PE (no mixing)
# ------------------------------------------------------------
class WaRo(nn.Module):
    """
    Axial attention with WaRP-PE:
      - For each axis, per-head learn a monotone 1D warp u(i) over positions i.
      - Apply Rotary Positional Encoding using warped indices (L_eff * u(i)).
      - Add a learnable *non-linear distance penalty* in u-space: -[a*|Δu| + b*|Δu|^2].
      - No RoPE/ALiBi/rPE mixing; this is a single unified mechanism.
    """
    def __init__(
        self,
        dim: int,
        heads: int = 4,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        # per-axis configs
        use_warp_h: bool = True,
        use_warp_w: bool = True,
        knots_h: int = 16,
        knots_w: int = 16,
        # rotary base (shared), requires dh even
        rope_theta_h: float = 10000.0,
        rope_theta_w: float = 10000.0,
        # non-linear distance penalty in warped space
        use_bias_h: bool = True,
        use_bias_w: bool = True,
    ) -> None:
        super().__init__()
        assert dim % heads == 0, "dim must be divisible by heads"
        self.dim = dim
        self.h   = heads
        self.dh  = dim // heads
        if use_warp_h or use_warp_w:
            assert self.dh % 2 == 0, "Rotary requires even head dim"

        self.use_warp = {'h': use_warp_h, 'w': use_warp_w}
        self.use_bias = {'h': use_bias_h, 'w': use_bias_w}

        # qkv / proj per-axis
        self.qkv_h = nn.Linear(dim, dim * 3, bias=True)
        self.qkv_w = nn.Linear(dim, dim * 3, bias=True)
        self.proj_h = nn.Linear(dim, dim)
        self.proj_w = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        # warpers (per-axis)
        if self.use_warp['h']:
            self.warp_h = WPE(self.h, K=knots_h)
            self.register_buffer("log_theta_h", torch.tensor(math.log(rope_theta_h), dtype=torch.float32), persistent=False)
        if self.use_warp['w']:
            self.warp_w = WPE(self.h, K=knots_w)
            self.register_buffer("log_theta_w", torch.tensor(math.log(rope_theta_w), dtype=torch.float32), persistent=False)

        # distance penalty params per-axis, per-head: a>=0, b>=0
        if self.use_bias['h']:
            self.bias_a_h = nn.Parameter(torch.zeros(self.h))
            self.bias_b_h = nn.Parameter(torch.zeros(self.h))
        if self.use_bias['w']:
            self.bias_a_w = nn.Parameter(torch.zeros(self.h))
            self.bias_b_w = nn.Parameter(torch.zeros(self.h))

        self.attn_drop = attn_drop

    # ------ rotary cache in warped coords ------
    def _rope_cache_warp(self, u: torch.Tensor, eff_len: torch.Tensor, log_theta: torch.Tensor, device, dtype):
        """u: (h, L) in [0,1]; eff_len: (h,) >=0.
        Return cos,sin shaped (1,h,L,d/2).
        """
        inv = torch.exp(-torch.arange(0, self.dh, 2, device=device, dtype=dtype) / self.dh * log_theta.exp())
        # Above is not standard; match classic RoPE scaling: inv_freq = theta ** (-2i/d)
        inv = (torch.exp(log_theta) ** (-torch.arange(0, self.dh, 2, device=device, dtype=dtype) / self.dh))
        # Angles per head: (h,L,d/2) = (h,L,1) * (1,1,d/2) * L_eff
        L_eff = eff_len.view(-1, 1, 1) * (u.unsqueeze(-1)) * (u.size(1) - 1)
        angles = L_eff * inv.view(1, 1, -1)
        cos = torch.cos(angles).unsqueeze(0)
        sin = torch.sin(angles).unsqueeze(0)
        return cos, sin

    def _apply_rope_qk_warp(self, q: torch.Tensor, k: torch.Tensor, u: torch.Tensor, eff_len: torch.Tensor, log_theta: torch.Tensor):
        Bp, h, L, d = q.shape
        cos, sin = self._rope_cache_warp(u, eff_len, log_theta, q.device, q.dtype)
        def _rope(x):
            x = x.view(Bp, h, L, d // 2, 2)
            x1, x2 = x[..., 0], x[..., 1]
            xr = torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
            return xr.view(Bp, h, L, d)
        return _rope(q), _rope(k)

    # ------ distance penalty in warped space ------
    @staticmethod
    def _pairwise_dist_u(u: torch.Tensor) -> torch.Tensor:
        # u: (h,L) -> (h,L,L)
        return (u[:, None, :] - u[:, :, None]).abs()

    def _bias_from_u(self, u: torch.Tensor, axis: str, device, dtype):
        # returns (1,h,L,L) for broadcasting to (Bp,h,L,L)
        dist = self._pairwise_dist_u(u).to(dtype=dtype, device=device)
        if axis == 'h':
            a = F.softplus(self.bias_a_h).view(-1, 1, 1)
            b = F.softplus(self.bias_b_h).view(-1, 1, 1)
        else:
            a = F.softplus(self.bias_a_w).view(-1, 1, 1)
            b = F.softplus(self.bias_b_w).view(-1, 1, 1)
        bias = -(a * dist + b * dist * dist)  # penalize far positions (nonlinear generalization of ALiBi)
        return bias.unsqueeze(0)

    # ------ axial attention core ------
    def _attn_1d(self, x: torch.Tensor, along: str = 'h') -> torch.Tensor:
        B, C, H, W = x.shape
        if along == 'h':
            xt = x.permute(0, 3, 2, 1).contiguous()   # (B,W,H,C)
            Bp, L, Cdim = B * W, H, C
            tokens = xt.view(Bp, L, Cdim)
            qkv = self.qkv_h(tokens)
        else:
            xt = x.permute(0, 2, 3, 1).contiguous()   # (B,H,W,C)
            Bp, L, Cdim = B * H, W, C
            tokens = xt.view(Bp, L, Cdim)
            qkv = self.qkv_w(tokens)

        qkv = qkv.view(Bp, L, 3, self.h, self.dh)
        q, k, v = qkv.unbind(dim=2)  # (Bp,L,h,d)
        q = q.transpose(1, 2)        # (Bp,h,L,d)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Warped rotary + bias
        attn_bias = None
        if along == 'h' and self.use_warp['h']:
            u_h, eff_h = self.warp_h(L, q.device, q.dtype)    # (h,L), (h,)
            q, k = self._apply_rope_qk_warp(q, k, u_h, eff_h, self.log_theta_h)
            if self.use_bias['h']:
                attn_bias = self._bias_from_u(u_h, 'h', q.device, q.dtype)  # (1,h,L,L)
        elif along == 'w' and self.use_warp['w']:
            u_w, eff_w = self.warp_w(L, q.device, q.dtype)
            q, k = self._apply_rope_qk_warp(q, k, u_w, eff_w, self.log_theta_w)
            if self.use_bias['w']:
                attn_bias = self._bias_from_u(u_w, 'w', q.device, q.dtype)

        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_bias.expand(Bp, -1, -1, -1) if attn_bias is not None else None,
            dropout_p=self.attn_drop if self.training else 0.0,
            is_causal=False,
        )  # (Bp,h,L,d)
        out = out.transpose(1, 2).reshape(Bp, L, Cdim)

        if along == 'h':
            out = self.proj_h(out)
            out = out.view(B, W, H, C).permute(0, 3, 2, 1).contiguous()
        else:
            out = self.proj_w(out)
            out = out.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()
        return self.proj_drop(out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return 0.5 * (self._attn_1d(x, 'h') + self._attn_1d(x, 'w'))

# ------------------------------------------------------------
# gMLP
# ------------------------------------------------------------
class gMLP(nn.Module):
    def __init__(self, dim: int, expansion: int = 4, drop: float = 0.0):
        super().__init__()
        hidden = dim * expansion
        self.norm = RMSNorm2d(dim)
        self.fc1 = nn.Conv2d(dim, hidden, 1)
        self.act = nn.GELU()
        self.fc2 = nn.Conv2d(hidden, dim, 1)
        self.drop = DropPath(drop)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s = x
        x = self.norm(x)
        x = self.fc2(self.act(self.fc1(x)))
        return s + self.drop(x)


class WMAA(nn.Module):
    def __init__(self, dim: int, heads: int = 4, drop_path: float = 0.0, attn_drop: float = 0.0,
                 # WaRP-PE
                 use_warp_h: bool = True, use_warp_w: bool = True, knots_h: int = 16, knots_w: int = 16,
                 rope_theta_h: float = 10000.0, rope_theta_w: float = 10000.0,
                 use_bias_h: bool = True, use_bias_w: bool = True,
                 gmlp_expansion: int = 4):
        super().__init__()
        self.norm1 = RMSNorm2d(dim)
        self.axial = WaRo(
            dim, heads=heads, attn_drop=attn_drop, proj_drop=0.0,
            use_warp_h=use_warp_h, use_warp_w=use_warp_w,
            knots_h=knots_h, knots_w=knots_w,
            rope_theta_h=rope_theta_h, rope_theta_w=rope_theta_w,
            use_bias_h=use_bias_h, use_bias_w=use_bias_w,
        )
        self.dp1   = DropPath(drop_path)
        self.gmlp  = gMLP(dim, expansion=gmlp_expansion, drop=drop_path)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.dp1(self.axial(self.norm1(x)))
        x = self.gmlp(x)
        return x

# ------------------------------------------------------------
# Full Model
# ------------------------------------------------------------
class WaFormer(nn.Module):
    def __init__(self, in_ch: int = 1, num_classes: int = 12,
                 base: int = 48, heads: int = 4, depths: Tuple[int,int] = (1,1), k: int = 5, drop_path: float = 0.1,
                 # WaRP-PE defaults
                 use_warp_h: bool = True, use_warp_w: bool = True, knots_h: int = 16, knots_w: int = 16,
                 rope_theta_h: float = 10000.0, rope_theta_w: float = 10000.0,
                 use_bias_h: bool = True, use_bias_w: bool = True,
                 gmlp_expansion: int = 4):
        super().__init__()
        assert base % heads == 0, "Prefer base divisible by heads"
        if (use_warp_h or use_warp_w) and ((base // heads) % 2 != 0):
            raise ValueError("WaRP-PE uses rotary; require (base/heads) even.")

        # Stem /4
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, base, 3, 2, 1, bias=False),
            RMSNorm2d(base),
            nn.SiLU(inplace=True),
            nn.Conv2d(base, base, 3, 2, 1, bias=False),
            RMSNorm2d(base),
            nn.SiLU(inplace=True),
        )
        # Local stage
        self.local = nn.Sequential(
            ConvLocalBlock(base, k=k, dilation=1, drop_path=0.0),
            ConvLocalBlock(base, k=k, dilation=2, drop_path=drop_path * 0.5),
        )
     
        d1, d2 = depths
        self.axial1 = nn.Sequential(*[
            WMAA(
                base, heads=heads, drop_path=drop_path * (i + 1) / max(1, d1), attn_drop=0.0,
                use_warp_h=use_warp_h, use_warp_w=use_warp_w,
                knots_h=knots_h, knots_w=knots_w,
                rope_theta_h=rope_theta_h, rope_theta_w=rope_theta_w,
                use_bias_h=use_bias_h, use_bias_w=use_bias_w,
                gmlp_expansion=gmlp_expansion
            ) for i in range(d1)
        ])
        self.axial2 = nn.Sequential(*[
            WMAA(
                base, heads=heads, drop_path=drop_path, attn_drop=0.0,
                use_warp_h=use_warp_h, use_warp_w=use_warp_w,
                knots_h=knots_h, knots_w=knots_w,
                rope_theta_h=rope_theta_h, rope_theta_w=rope_theta_w,
                use_bias_h=use_bias_h, use_bias_w=use_bias_w,
                gmlp_expansion=gmlp_expansion
            ) for _ in range(1)
        ])
        # Head
        self.head_norm = RMSNorm2d(base)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc   = nn.Linear(base, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.local(x)
        x = self.axial1(x)
        x = self.axial2(x)
        x = self.head_norm(x)
        x = self.pool(x).flatten(1)
        return self.fc(x)


model = WaFormer(
        in_ch=3, num_classes=12,
        base=30, heads=3, depths=(2,2), k=5, drop_path=0.05,
        use_warp_h=True, use_warp_w=True, knots_h=16, knots_w=16,
        rope_theta_h=10000.0, rope_theta_w=10000.0,
        use_bias_h=True, use_bias_w=True,
        gmlp_expansion=4
    ).to(device)

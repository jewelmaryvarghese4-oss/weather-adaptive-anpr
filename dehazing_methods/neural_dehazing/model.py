"""
model.py - Transformer-based dehazing network
CNN encoder/decoder (U-Net) with window-attention transformer blocks
in the bottleneck. Predicts a residual correction added to the input
(standard practice in dehazing - converges faster than predicting the
clear image directly).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------
# Building blocks
# --------------------------------------------------------------------------

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
        )

    def forward(self, x):
        return self.net(x)


class Downsample(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.op = nn.Conv2d(ch, ch * 2, 3, stride=2, padding=1)

    def forward(self, x):
        return self.op(x)


class Upsample(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.op = nn.ConvTranspose2d(ch, ch // 2, 2, stride=2)

    def forward(self, x):
        return self.op(x)


class WindowAttention(nn.Module):
    """Non-shifted window self-attention (Swin-style)."""

    def __init__(self, dim, window_size=8, num_heads=4):
        super().__init__()
        self.window_size = window_size
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        B, C, H, W = x.shape
        ws = self.window_size
        pad_h = (ws - H % ws) % ws
        pad_w = (ws - W % ws) % ws
        x = F.pad(x, (0, pad_w, 0, pad_h))
        Hp, Wp = H + pad_h, W + pad_w

        x = x.permute(0, 2, 3, 1)
        x = x.view(B, Hp // ws, ws, Wp // ws, ws, C)
        x = x.permute(0, 1, 3, 2, 4, 5).reshape(-1, ws * ws, C)

        shortcut = x
        x = self.norm(x)
        qkv = self.qkv(x).reshape(x.shape[0], ws * ws, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(x.shape[0], ws * ws, C)
        out = self.proj(out) + shortcut

        out = out.view(B, Hp // ws, Wp // ws, ws, ws, C)
        out = out.permute(0, 1, 3, 2, 4, 5).reshape(B, Hp, Wp, C)
        out = out.permute(0, 3, 1, 2)
        return out[:, :, :H, :W]


class TransformerBottleneckBlock(nn.Module):
    def __init__(self, dim, window_size=8, num_heads=4, mlp_ratio=2.0):
        super().__init__()
        self.attn = WindowAttention(dim, window_size, num_heads)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Conv2d(dim, hidden, 1),
            nn.GELU(),
            nn.Conv2d(hidden, dim, 1),
        )
        self.norm = nn.GroupNorm(1, dim)

    def forward(self, x):
        x = x + self.attn(x)
        x = x + self.mlp(self.norm(x))
        return x


# --------------------------------------------------------------------------
# U-Net scaffold
# --------------------------------------------------------------------------

class TransformerDehazeNet(nn.Module):
    def __init__(self, base_ch=32, bottleneck_blocks=4, window_size=8, num_heads=4):
        super().__init__()
        self.stem = ConvBlock(3, base_ch)
        self.down1 = Downsample(base_ch)
        self.enc1 = ConvBlock(base_ch * 2, base_ch * 2)
        self.down2 = Downsample(base_ch * 2)
        self.enc2 = ConvBlock(base_ch * 4, base_ch * 4)

        self.bottleneck = nn.Sequential(
            *[TransformerBottleneckBlock(base_ch * 4, window_size, num_heads)
              for _ in range(bottleneck_blocks)]
        )

        self.up2 = Upsample(base_ch * 4)
        self.dec2 = ConvBlock(base_ch * 2 * 2, base_ch * 2)
        self.up1 = Upsample(base_ch * 2)
        self.dec1 = ConvBlock(base_ch * 2, base_ch)

        self.out_conv = nn.Conv2d(base_ch, 3, 3, padding=1)

    def forward(self, x):
        s0 = self.stem(x)
        s1 = self.enc1(self.down1(s0))
        s2 = self.enc2(self.down2(s1))

        b = self.bottleneck(s2)

        d2 = self.up2(b)
        d2 = self.dec2(torch.cat([d2, s1], dim=1))
        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, s0], dim=1))

        residual = self.out_conv(d1)
        out = x + residual
        return torch.clamp(out, 0.0, 1.0)


if __name__ == "__main__":
    x = torch.rand(2, 3, 256, 256)
    model = TransformerDehazeNet(base_ch=32, bottleneck_blocks=4)
    y = model(x)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Output shape: {y.shape}, params: {n_params/1e6:.2f}M")

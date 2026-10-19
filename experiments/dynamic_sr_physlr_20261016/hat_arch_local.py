"""HAT (Hybrid Attention Transformer, Chen et al., CVPR 2023) network for loading the official HAT-L x4 weights (Claude 4dsr-7c,
2026-10-18). The official source (XPixelGroup/HAT, hat/archs/hat_arch.py at commit 1638a9a, built on BasicSR) is not available on
this machine, so this file re-implements the inference/training graph of that module: same module tree and parameter names
(window attention + channel-attention conv branch scaled by conv_scale in each HAB, one overlapping cross-attention block per
residual group, pixel-shuffle reconstruction, RGB mean shift), relative-position indices and the shifted-window mask computed at
model level. Validation (hat_screen.py check): strict load of the official params_ema (all 1710 entries, names and shapes; the two
relative-position index buffers come from the checkpoint) and a functional bicubic x4 check against SwinIR-M on held-out frames.
It is a re-implementation, not the official file. A section-by-section comparison with the official source fetched on 2026-10-18
(ChannelAttention, CAB, WindowAttention, HAB, OCAB, the index / mask functions, forward_features, forward) found the inference graph
identical; the training graph differs in one point: no DropPath (the official HAT defaults to drop_path_rate=0.1).

License and changes: derived from XPixelGroup/HAT (Apache License 2.0, https://github.com/XPixelGroup/HAT/blob/main/LICENSE).
Changes relative to the upstream module: rewritten in condensed form; only the classical pixel-shuffle x4 path; no DropPath /
dropout; the OCA relative-position index is taken from the checkpoint buffer (the local formula is a shape placeholder); optional
gradient checkpointing per residual group.

Inference follows the official test pre-processing: reflect-pad the LR to a multiple of window_size, run, crop the output.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def window_partition(x, ws):
    b, h, w, c = x.shape
    x = x.view(b, h // ws, ws, w // ws, ws, c)
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, ws, ws, c)


def window_reverse(windows, ws, h, w):
    b = int(windows.shape[0] / (h * w / ws / ws))
    x = windows.view(b, h // ws, w // ws, ws, ws, -1)
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(b, h, w, -1)


class ChannelAttention(nn.Module):
    def __init__(self, num_feat, squeeze_factor):
        super().__init__()
        self.attention = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(num_feat, num_feat // squeeze_factor, 1, padding=0),
                                       nn.ReLU(inplace=True), nn.Conv2d(num_feat // squeeze_factor, num_feat, 1, padding=0), nn.Sigmoid())

    def forward(self, x):
        return x * self.attention(x)


class CAB(nn.Module):
    def __init__(self, num_feat, compress_ratio, squeeze_factor):
        super().__init__()
        self.cab = nn.Sequential(nn.Conv2d(num_feat, num_feat // compress_ratio, 3, 1, 1), nn.GELU(),
                                 nn.Conv2d(num_feat // compress_ratio, num_feat, 3, 1, 1), ChannelAttention(num_feat, squeeze_factor))

    def forward(self, x):
        return self.cab(x)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, in_features)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class WindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads):
        super().__init__()
        self.window_size, self.num_heads = window_size, num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.relative_position_bias_table = nn.Parameter(torch.zeros((2 * window_size - 1) ** 2, num_heads))
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)
        nn.init.trunc_normal_(self.relative_position_bias_table, std=.02)

    def forward(self, x, rpi, mask=None):
        b_, n, c = x.shape
        qkv = self.qkv(x).reshape(b_, n, 3, self.num_heads, c // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q * self.scale) @ k.transpose(-2, -1)
        n2 = self.window_size * self.window_size
        bias = self.relative_position_bias_table[rpi.view(-1)].view(n2, n2, -1).permute(2, 0, 1).contiguous()
        attn = attn + bias.unsqueeze(0)
        if mask is not None:
            nw = mask.shape[0]
            attn = attn.view(b_ // nw, nw, self.num_heads, n, n) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, n, n)
        attn = attn.softmax(dim=-1)
        return self.proj((attn @ v).transpose(1, 2).reshape(b_, n, c))


class HAB(nn.Module):
    """Hybrid attention block: (shifted) window self-attention plus a channel-attention conv branch scaled by conv_scale."""

    def __init__(self, dim, input_resolution, num_heads, window_size, shift_size, compress_ratio, squeeze_factor, conv_scale, mlp_ratio):
        super().__init__()
        self.window_size, self.shift_size = window_size, shift_size
        if min(input_resolution) <= self.window_size:
            self.shift_size, self.window_size = 0, min(input_resolution)
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, self.window_size, num_heads)
        self.conv_scale = conv_scale
        self.conv_block = CAB(dim, compress_ratio, squeeze_factor)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(dim, int(dim * mlp_ratio))

    def forward(self, x, x_size, rpi_sa, attn_mask):
        h, w = x_size
        b, _, c = x.shape
        shortcut = x
        x = self.norm1(x).view(b, h, w, c)
        conv_x = self.conv_block(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1).contiguous().view(b, h * w, c)
        if self.shift_size > 0:
            shifted = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            mask = attn_mask
        else:
            shifted, mask = x, None
        win = window_partition(shifted, self.window_size).view(-1, self.window_size * self.window_size, c)
        win = self.attn(win, rpi=rpi_sa, mask=mask).view(-1, self.window_size, self.window_size, c)
        shifted = window_reverse(win, self.window_size, h, w)
        attn_x = torch.roll(shifted, shifts=(self.shift_size, self.shift_size), dims=(1, 2)) if self.shift_size > 0 else shifted
        x = shortcut + attn_x.view(b, h * w, c) + conv_x * self.conv_scale
        return x + self.mlp(self.norm2(x))


class OCAB(nn.Module):
    """Overlapping cross-attention block: queries from non-overlapping windows, keys / values from enlarged overlapping windows."""

    def __init__(self, dim, window_size, overlap_ratio, num_heads, mlp_ratio):
        super().__init__()
        self.dim, self.window_size, self.num_heads = dim, window_size, num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.overlap_win_size = int(window_size * overlap_ratio) + window_size
        self.norm1 = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.unfold = nn.Unfold(kernel_size=(self.overlap_win_size, self.overlap_win_size), stride=window_size,
                                padding=(self.overlap_win_size - window_size) // 2)
        self.relative_position_bias_table = nn.Parameter(torch.zeros((window_size + self.overlap_win_size - 1) ** 2, num_heads))
        nn.init.trunc_normal_(self.relative_position_bias_table, std=.02)
        self.proj = nn.Linear(dim, dim)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(dim, int(dim * mlp_ratio))

    def forward(self, x, x_size, rpi):
        h, w = x_size
        b, _, c = x.shape
        ws, ows = self.window_size, self.overlap_win_size
        shortcut = x
        x = self.norm1(x).view(b, h, w, c)
        qkv = self.qkv(x).reshape(b, h, w, 3, c).permute(3, 0, 4, 1, 2)  # 3, b, c, h, w
        q = qkv[0].permute(0, 2, 3, 1)
        kv = torch.cat((qkv[1], qkv[2]), dim=1)  # b, 2c, h, w
        q_win = window_partition(q, ws).view(-1, ws * ws, c)
        kv_win = self.unfold(kv)  # b, 2c*ows*ows, nw  (channel-major, then kernel rows / columns)
        nw = kv_win.shape[-1]
        kv_win = kv_win.view(b, 2, c, ows * ows, nw).permute(1, 0, 4, 3, 2).reshape(2, b * nw, ows * ows, c)
        k_win, v_win = kv_win[0], kv_win[1]
        b_, nq, _ = q_win.shape
        n = k_win.shape[1]
        d = self.dim // self.num_heads
        q_ = q_win.reshape(b_, nq, self.num_heads, d).permute(0, 2, 1, 3)
        k_ = k_win.reshape(b_, n, self.num_heads, d).permute(0, 2, 1, 3)
        v_ = v_win.reshape(b_, n, self.num_heads, d).permute(0, 2, 1, 3)
        attn = (q_ * self.scale) @ k_.transpose(-2, -1)
        bias = self.relative_position_bias_table[rpi.view(-1)].view(ws * ws, ows * ows, -1).permute(2, 0, 1).contiguous()
        attn = (attn + bias.unsqueeze(0)).softmax(dim=-1)
        out = (attn @ v_).transpose(1, 2).reshape(b_, nq, self.dim).view(-1, ws, ws, self.dim)
        x = window_reverse(out, ws, h, w).view(b, h * w, self.dim)
        x = self.proj(x) + shortcut
        return x + self.mlp(self.norm2(x))


class AttenBlocks(nn.Module):
    def __init__(self, dim, input_resolution, depth, num_heads, window_size, compress_ratio, squeeze_factor, conv_scale,
                 overlap_ratio, mlp_ratio):
        super().__init__()
        self.blocks = nn.ModuleList([HAB(dim, input_resolution, num_heads, window_size, 0 if i % 2 == 0 else window_size // 2,
                                         compress_ratio, squeeze_factor, conv_scale, mlp_ratio) for i in range(depth)])
        self.overlap_attn = OCAB(dim, window_size, overlap_ratio, num_heads, mlp_ratio)

    def forward(self, x, x_size, params):
        for blk in self.blocks:
            x = blk(x, x_size, params["rpi_sa"], params["attn_mask"])
        return self.overlap_attn(x, x_size, params["rpi_oca"])


class RHAG(nn.Module):
    """Residual hybrid attention group: blocks, OCAB, 3x3 conv, residual."""

    def __init__(self, dim, input_resolution, depth, num_heads, window_size, compress_ratio, squeeze_factor, conv_scale,
                 overlap_ratio, mlp_ratio, use_checkpoint=False):
        super().__init__()
        self.residual_group = AttenBlocks(dim, input_resolution, depth, num_heads, window_size, compress_ratio, squeeze_factor,
                                          conv_scale, overlap_ratio, mlp_ratio)
        self.conv = nn.Conv2d(dim, dim, 3, 1, 1)
        self.dim, self.use_checkpoint = dim, use_checkpoint

    def _body(self, x, x_size, params):
        y = self.residual_group(x, x_size, params)
        y = y.transpose(1, 2).contiguous().view(x.shape[0], self.dim, x_size[0], x_size[1])
        return self.conv(y).flatten(2).transpose(1, 2)

    def forward(self, x, x_size, params):
        if self.use_checkpoint and self.training:
            from torch.utils.checkpoint import checkpoint
            return checkpoint(self._body, x, x_size, params, use_reentrant=False) + x
        return self._body(x, x_size, params) + x


class PatchEmbedNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        return self.norm(x.flatten(2).transpose(1, 2))


class Upsample(nn.Sequential):
    def __init__(self, scale, num_feat):
        m = []
        if scale & (scale - 1) != 0:
            raise ValueError("power-of-two scales only")
        for _ in range(int(math.log2(scale))):
            m += [nn.Conv2d(num_feat, 4 * num_feat, 3, 1, 1), nn.PixelShuffle(2)]
        super().__init__(*m)


class HAT(nn.Module):
    def __init__(self, upscale=4, in_chans=3, img_size=64, window_size=16, compress_ratio=3, squeeze_factor=30, conv_scale=0.01,
                 overlap_ratio=0.5, img_range=1., depths=(6,) * 12, embed_dim=180, num_heads=(6,) * 12, mlp_ratio=2,
                 use_checkpoint=False):
        super().__init__()
        self.window_size, self.shift_size, self.overlap_ratio = window_size, window_size // 2, overlap_ratio
        self.img_range, self.upscale = img_range, upscale
        self.mean = torch.Tensor((0.4488, 0.4371, 0.4040)).view(1, 3, 1, 1)
        self.register_buffer("relative_position_index_SA", self.calculate_rpi_sa())
        self.register_buffer("relative_position_index_OCA", self.calculate_rpi_oca())
        num_feat = 64
        self.conv_first = nn.Conv2d(in_chans, embed_dim, 3, 1, 1)
        self.patch_embed = PatchEmbedNorm(embed_dim)
        res = (img_size, img_size)
        self.layers = nn.ModuleList([RHAG(embed_dim, res, depths[i], num_heads[i], window_size, compress_ratio, squeeze_factor,
                                          conv_scale, overlap_ratio, mlp_ratio, use_checkpoint) for i in range(len(depths))])
        self.norm = nn.LayerNorm(embed_dim)
        self.conv_after_body = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)
        self.conv_before_upsample = nn.Sequential(nn.Conv2d(embed_dim, num_feat, 3, 1, 1), nn.LeakyReLU(inplace=True))
        self.upsample = Upsample(upscale, num_feat)
        self.conv_last = nn.Conv2d(num_feat, in_chans, 3, 1, 1)

    def calculate_rpi_sa(self):
        ws = self.window_size
        coords = torch.stack(torch.meshgrid(torch.arange(ws), torch.arange(ws), indexing="ij")).flatten(1)
        rel = (coords[:, :, None] - coords[:, None, :]).permute(1, 2, 0).contiguous()
        rel[:, :, 0] += ws - 1
        rel[:, :, 1] += ws - 1
        rel[:, :, 0] *= 2 * ws - 1
        return rel.sum(-1)

    def calculate_rpi_oca(self):
        # shape placeholder: the official index table is loaded from the checkpoint (strict load), so these values are overwritten
        ws, wse = self.window_size, self.window_size + int(self.overlap_ratio * self.window_size)
        ori = torch.stack(torch.meshgrid(torch.arange(ws), torch.arange(ws), indexing="ij")).flatten(1)
        ext = torch.stack(torch.meshgrid(torch.arange(wse), torch.arange(wse), indexing="ij")).flatten(1)
        rel = (ext[:, None, :] - ori[:, :, None]).permute(1, 2, 0).contiguous()
        rel[:, :, 0] += ws - 1
        rel[:, :, 1] += ws - 1
        rel[:, :, 0] *= ws + wse - 1
        return rel.sum(-1)

    def calculate_mask(self, x_size, device):
        h, w = x_size
        ws, ss = self.window_size, self.shift_size
        img_mask = torch.zeros((1, h, w, 1), device=device)
        cnt = 0
        for hs in (slice(0, -ws), slice(-ws, -ss), slice(-ss, None)):
            for wsl in (slice(0, -ws), slice(-ws, -ss), slice(-ss, None)):
                img_mask[:, hs, wsl, :] = cnt
                cnt += 1
        mw = window_partition(img_mask, ws).view(-1, ws * ws)
        m = mw.unsqueeze(1) - mw.unsqueeze(2)
        return m.masked_fill(m != 0, float(-100.0)).masked_fill(m == 0, float(0.0))

    def forward_features(self, x):
        x_size = (x.shape[2], x.shape[3])
        params = dict(attn_mask=self.calculate_mask(x_size, x.device), rpi_sa=self.relative_position_index_SA,
                      rpi_oca=self.relative_position_index_OCA)
        x = self.patch_embed(x)
        for layer in self.layers:
            x = layer(x, x_size, params)
        x = self.norm(x)
        return x.transpose(1, 2).contiguous().view(x.shape[0], -1, x_size[0], x_size[1])

    def forward(self, x):
        mean = self.mean.type_as(x)
        x = (x - mean) * self.img_range
        x = self.conv_first(x)
        x = self.conv_after_body(self.forward_features(x)) + x
        x = self.conv_last(self.upsample(self.conv_before_upsample(x)))
        return x / self.img_range + mean


def build_hat_l(use_checkpoint=False):
    return HAT(upscale=4, in_chans=3, img_size=64, window_size=16, compress_ratio=3, squeeze_factor=30, conv_scale=0.01,
               overlap_ratio=0.5, img_range=1., depths=(6,) * 12, embed_dim=180, num_heads=(6,) * 12, mlp_ratio=2,
               use_checkpoint=use_checkpoint)


def load_state(net, path, key="params_ema"):
    """Strict load; weights_only (no arbitrary unpickling)."""
    state = torch.load(path, map_location="cpu", weights_only=True)
    state = state[key] if key in state else state
    net.load_state_dict(state, strict=True)
    return net


def infer(net, lr, window_size=16, scale=4):
    """lr: (B,3,h,w) in [0,1]. Official HAT test pre-processing: reflect pad to a multiple of window_size, then crop."""
    _, _, h, w = lr.shape
    ph = (window_size - h % window_size) % window_size
    pw = (window_size - w % window_size) % window_size
    x = F.pad(lr, (0, pw, 0, ph), mode="reflect") if (ph or pw) else lr
    return net(x)[..., :h * scale, :w * scale]

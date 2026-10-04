# src/ibbi/models/_codino.py

"""
Pure-PyTorch inference implementation of Co-DINO with an EVA-02-L backbone (Zong, Song & Liu, ICCV 2023), as trained
for the IBBI arthropod detector.

Only the inference path is implemented: the EVA-02 ViT-L backbone (window + global attention with rotary position
embeddings), the simple feature pyramid (SFP), the two-stage DINO deformable transformer and the query head. The
auxiliary collaborative heads of Co-DETR are used only for training and are not needed. Module and parameter names
follow the original implementation, so the released state dict loads unchanged.

Adapted from Co-DETR (https://github.com/Sense-X/Co-DETR, MIT License), MMDetection 2.25 and MMCV 1.7
(https://github.com/open-mmlab, Apache License 2.0); multi-scale deformable attention uses MMCV's pure-PyTorch
formulation (bilinear `grid_sample`), so no compiled extension is required.
"""

import math
from functools import lru_cache

import torch
import torch.nn.functional as F
from torch import nn

# ----------------------------------------------------------------------------------------------------------------------
# EVA-02 ViT-L backbone


@lru_cache(maxsize=16)
def _rope_tables(h: int, w: int, dim: int = 32, pt_seq_len: int = 16, theta: float = 10000.0):
    """Rotary tables (cos, sin) [h*w, 2*dim] for an h x w token grid, computed on the CPU in float32 as in EVA-02."""
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: dim // 2].float() / dim))
    fh = torch.einsum("..., f -> ... f", torch.arange(h) / h * pt_seq_len, freqs).repeat_interleave(2, dim=-1)
    fw = torch.einsum("..., f -> ... f", torch.arange(w) / w * pt_seq_len, freqs).repeat_interleave(2, dim=-1)
    grid = torch.cat([fh[:, None, :].expand(h, w, -1), fw[None, :, :].expand(h, w, -1)], dim=-1)
    return grid.cos().reshape(h * w, -1), grid.sin().reshape(h * w, -1)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x = x.reshape(*x.shape[:-1], -1, 2)
    x1, x2 = x.unbind(-1)
    return torch.stack((-x2, x1), dim=-1).flatten(-2)


def _apply_rope(t: torch.Tensor, h: int, w: int) -> torch.Tensor:
    cos, sin = _rope_tables(h, w, dim=t.shape[-1] // 2)  # 32 for EVA-02-L (64-d heads), as in the original
    cos, sin = cos.to(t.device), sin.to(t.device)
    return t * cos + _rotate_half(t) * sin


class PatchEmbed(nn.Module):
    def __init__(self, patch_size: int = 16, in_chans: int = 3, embed_dim: int = 1024):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        return self.proj(x).permute(0, 2, 3, 1)  # B H W C


class SwiGLU(nn.Module):
    def __init__(self, dim: int, hidden: int):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden)
        self.w2 = nn.Linear(dim, hidden)
        self.act = nn.SiLU()
        self.ffn_ln = nn.LayerNorm(hidden, eps=1e-6)
        self.w3 = nn.Linear(hidden, dim)

    def forward(self, x):
        return self.w3(self.ffn_ln(self.act(self.w1(x)) * self.w2(x)))


class Attention(nn.Module):
    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.q_bias = nn.Parameter(torch.zeros(dim))
        self.v_bias = nn.Parameter(torch.zeros(dim))
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, H, W, C = x.shape
        x = x.reshape(B, H * W, C)
        q = F.linear(x, self.q_proj.weight, self.q_bias).reshape(B, H * W, self.num_heads, -1).transpose(1, 2)
        k = F.linear(x, self.k_proj.weight).reshape(B, H * W, self.num_heads, -1).transpose(1, 2)
        v = F.linear(x, self.v_proj.weight, self.v_bias).reshape(B, H * W, self.num_heads, -1).transpose(1, 2)
        q = _apply_rope(q, H, W).type_as(v)
        k = _apply_rope(k, H, W).type_as(v)
        # memory-efficient exact attention (same result as softmax(q k^T / sqrt(d)) v)
        x = F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(B, H * W, C)
        return self.proj(x).reshape(B, H, W, C)


def _window_partition(x, ws: int):
    B, H, W, C = x.shape
    ph, pw = (ws - H % ws) % ws, (ws - W % ws) % ws
    if ph or pw:
        x = F.pad(x, (0, 0, 0, pw, 0, ph))
    Hp, Wp = H + ph, W + pw
    x = x.view(B, Hp // ws, ws, Wp // ws, ws, C).permute(0, 1, 3, 2, 4, 5).reshape(-1, ws, ws, C)
    return x, (Hp, Wp)


def _window_unpartition(x, ws: int, pad_hw, hw):
    Hp, Wp = pad_hw
    H, W = hw
    B = x.shape[0] // (Hp * Wp // ws // ws)
    x = x.view(B, Hp // ws, Wp // ws, ws, ws, -1).permute(0, 1, 3, 2, 4, 5).reshape(B, Hp, Wp, -1)
    return x[:, :H, :W, :].contiguous()


class Block(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float, window_size: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = Attention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = SwiGLU(dim, int(dim * mlp_ratio))
        self.window_size = window_size

    def forward(self, x):
        shortcut = x
        x = self.norm1(x)
        if self.window_size > 0:
            H, W = x.shape[1], x.shape[2]
            x, pad_hw = _window_partition(x, self.window_size)
        x = self.attn(x)
        if self.window_size > 0:
            x = _window_unpartition(x, self.window_size, pad_hw, (H, W))
        x = shortcut + x
        return x + self.mlp(self.norm2(x))


class ViT(nn.Module):
    """EVA-02 ViT as used by Co-DETR (ViTDet-style: no cls token at inference, interpolated absolute position
    embedding, window attention in most blocks, rotary embeddings everywhere)."""

    def __init__(
        self,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4 * 2 / 3,
        patch_size=16,
        pretrain_img_size=512,
        window_size=24,
        window_block_indexes=(),
    ):
        super().__init__()
        self.patch_embed = PatchEmbed(patch_size, 3, embed_dim)
        n = (pretrain_img_size // patch_size) ** 2 + 1
        self.pos_embed = nn.Parameter(torch.zeros(1, n, embed_dim))
        self.blocks = nn.ModuleList(Block(embed_dim, num_heads, mlp_ratio, window_size if i in window_block_indexes else 0) for i in range(depth))
        self.out_norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.patch_embed(x)
        h, w = x.shape[1], x.shape[2]
        pos = self.pos_embed[:, 1:]
        size = int(math.sqrt(pos.shape[1]))
        if size != h or size != w:
            pos = F.interpolate(pos.reshape(1, size, size, -1).permute(0, 3, 1, 2), size=(h, w), mode="bicubic", align_corners=False)
            pos = pos.permute(0, 2, 3, 1)
        else:
            pos = pos.reshape(1, h, w, -1)
        x = x + pos
        for blk in self.blocks:
            x = blk(x)
        return self.out_norm(x).permute(0, 3, 1, 2).contiguous()


# ----------------------------------------------------------------------------------------------------------------------
# Simple feature pyramid


class LayerNorm2d(nn.Module):
    def __init__(self, channels: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = eps

    def forward(self, x):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        return self.weight[:, None, None] * x + self.bias[:, None, None]


def _conv(i, o, k=3, s=1):
    return nn.Conv2d(i, o, kernel_size=k, stride=s, padding=k // 2, bias=False)


class SFP(nn.Module):
    """Simple feature pyramid with p2: five levels at strides 4, 8, 16, 32, 64."""

    def __init__(self, c: int = 1024, out: int = 256):
        super().__init__()
        up = lambda: nn.Upsample(scale_factor=2)  # noqa: E731
        self.p2 = nn.Sequential(
            up(),
            _conv(c, c // 2),
            LayerNorm2d(c // 2),
            nn.GELU(),
            up(),
            _conv(c // 2, c // 4),
            LayerNorm2d(c // 4),
            nn.GELU(),
            _conv(c // 4, out, 1),
            LayerNorm2d(out),
            nn.GELU(),
            _conv(out, out),
            LayerNorm2d(out),
        )
        self.p3 = nn.Sequential(
            up(),
            _conv(c, c // 2),
            LayerNorm2d(c // 2),
            nn.GELU(),
            _conv(c // 2, out, 1),
            LayerNorm2d(out),
            nn.GELU(),
            _conv(out, out),
            LayerNorm2d(out),
        )
        self.p4 = nn.Sequential(_conv(c, out, 1), LayerNorm2d(out), nn.GELU(), _conv(out, out), LayerNorm2d(out))
        self.p5 = nn.Sequential(nn.MaxPool2d(3, 2, 1), _conv(c, out, 1), LayerNorm2d(out), nn.GELU(), _conv(out, out), LayerNorm2d(out))
        self.p6 = nn.Sequential(
            nn.MaxPool2d(3, 2, 1),
            _conv(c, c, 3, 2),
            LayerNorm2d(c),
            nn.GELU(),
            _conv(c, out, 1),
            LayerNorm2d(out),
            nn.GELU(),
            _conv(out, out),
            LayerNorm2d(out),
        )

    def forward(self, x):
        return [self.p2(x), self.p3(x), self.p4(x), self.p5(x), self.p6(x)]


# ----------------------------------------------------------------------------------------------------------------------
# DINO deformable transformer


def inverse_sigmoid(x, eps: float = 1e-5):
    x = x.clamp(min=0, max=1)
    return torch.log(x.clamp(min=eps) / (1 - x).clamp(min=eps))


def _msda_core(value, spatial_shapes, sampling_locations, attention_weights):
    """Multi-scale deformable attention, pure-PyTorch formulation of MMCV."""
    bs, _, num_heads, dims = value.shape
    _, nq, _, num_levels, num_points, _ = sampling_locations.shape
    shapes = [(int(h), int(w)) for h, w in spatial_shapes.tolist()]
    value_list = value.split([h * w for h, w in shapes], dim=1)
    grids = 2 * sampling_locations - 1
    sampled = []
    for lvl, (h, w) in enumerate(shapes):
        v = value_list[lvl].flatten(2).transpose(1, 2).reshape(bs * num_heads, dims, h, w)
        g = grids[:, :, :, lvl].transpose(1, 2).flatten(0, 1)
        sampled.append(F.grid_sample(v, g, mode="bilinear", padding_mode="zeros", align_corners=False))
    attention_weights = attention_weights.transpose(1, 2).reshape(bs * num_heads, 1, nq, num_levels * num_points)
    out = (torch.stack(sampled, dim=-2).flatten(-2) * attention_weights).sum(-1).view(bs, num_heads * dims, nq)
    return out.transpose(1, 2).contiguous()


class MultiScaleDeformableAttention(nn.Module):
    def __init__(self, embed_dims=256, num_heads=8, num_levels=5, num_points=4):
        super().__init__()
        self.num_heads, self.num_levels, self.num_points = num_heads, num_levels, num_points
        self.sampling_offsets = nn.Linear(embed_dims, num_heads * num_levels * num_points * 2)
        self.attention_weights = nn.Linear(embed_dims, num_heads * num_levels * num_points)
        self.value_proj = nn.Linear(embed_dims, embed_dims)
        self.output_proj = nn.Linear(embed_dims, embed_dims)

    def forward(self, query, value, query_pos, key_padding_mask, reference_points, spatial_shapes):
        # query, value: (num, bs, C) as in MMCV (batch_first=False)
        identity = query
        query = query + query_pos
        query = query.permute(1, 0, 2)
        value = value.permute(1, 0, 2)
        bs, nq, _ = query.shape
        nv = value.shape[1]
        value = self.value_proj(value)
        if key_padding_mask is not None:
            value = value.masked_fill(key_padding_mask[..., None], 0.0)
        value = value.view(bs, nv, self.num_heads, -1)
        off = self.sampling_offsets(query).view(bs, nq, self.num_heads, self.num_levels, self.num_points, 2)
        aw = self.attention_weights(query).view(bs, nq, self.num_heads, self.num_levels * self.num_points).softmax(-1)
        aw = aw.view(bs, nq, self.num_heads, self.num_levels, self.num_points)
        if reference_points.shape[-1] == 2:
            norm = torch.stack([spatial_shapes[..., 1], spatial_shapes[..., 0]], -1)
            loc = reference_points[:, :, None, :, None, :] + off / norm[None, None, None, :, None, :]
        else:
            loc = reference_points[:, :, None, :, None, :2] + off / self.num_points * reference_points[:, :, None, :, None, 2:] * 0.5
        out = self.output_proj(_msda_core(value, spatial_shapes, loc, aw))
        return out.permute(1, 0, 2) + identity


class _MHA(nn.Module):
    """MMCV MultiheadAttention wrapper (position added to query and key, identity shortcut)."""

    def __init__(self, embed_dims=256, num_heads=8):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dims, num_heads, dropout=0.0)

    def forward(self, query, query_pos):
        q = query + query_pos
        return query + self.attn(q, q, query)[0]


class _FFN(nn.Module):
    def __init__(self, embed_dims=256, hidden=2048):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Sequential(nn.Linear(embed_dims, hidden), nn.ReLU(inplace=True), nn.Dropout(0.0)), nn.Linear(hidden, embed_dims), nn.Dropout(0.0)
        )

    def forward(self, x):
        return x + self.layers(x)


class EncoderLayer(nn.Module):
    def __init__(self, d=256, levels=5, ffn=2048):
        super().__init__()
        self.attentions = nn.ModuleList([MultiScaleDeformableAttention(d, 8, levels, 4)])
        self.norms = nn.ModuleList([nn.LayerNorm(d), nn.LayerNorm(d)])
        self.ffns = nn.ModuleList([_FFN(d, ffn)])

    def forward(self, query, query_pos, mask, reference_points, spatial_shapes):
        query = self.attentions[0](query, query, query_pos, mask, reference_points, spatial_shapes)
        query = self.norms[0](query)
        return self.norms[1](self.ffns[0](query))


class DecoderLayer(nn.Module):
    def __init__(self, d=256, levels=5, ffn=2048):
        super().__init__()
        self.attentions = nn.ModuleList([_MHA(d, 8), MultiScaleDeformableAttention(d, 8, levels, 4)])
        self.norms = nn.ModuleList([nn.LayerNorm(d), nn.LayerNorm(d), nn.LayerNorm(d)])
        self.ffns = nn.ModuleList([_FFN(d, ffn)])

    def forward(self, query, memory, query_pos, mask, reference_points, spatial_shapes):
        query = self.norms[0](self.attentions[0](query, query_pos))
        query = self.norms[1](self.attentions[1](query, memory, query_pos, mask, reference_points, spatial_shapes))
        return self.norms[2](self.ffns[0](query))


class _Encoder(nn.Module):
    def __init__(self, n=6):
        super().__init__()
        self.layers = nn.ModuleList(EncoderLayer() for _ in range(n))


def _sine_embed(pos, num_feats: int):
    scale = 2 * math.pi
    dim_t = torch.arange(num_feats, dtype=torch.float32, device=pos.device)
    dim_t = 10000 ** (2 * (dim_t // 2) / num_feats)
    parts = []
    for i in (1, 0, 2, 3):  # y, x, w, h
        p = pos[:, :, i] * scale
        p = p[:, :, None] / dim_t
        parts.append(torch.stack((p[:, :, 0::2].sin(), p[:, :, 1::2].cos()), dim=3).flatten(2))
    return torch.cat(parts, dim=2)


class _Decoder(nn.Module):
    def __init__(self, n=6, d=256):
        super().__init__()
        self.layers = nn.ModuleList(DecoderLayer() for _ in range(n))
        self.ref_point_head = nn.Sequential(nn.Linear(2 * d, d), nn.ReLU(), nn.Linear(d, d))
        self.norm = nn.LayerNorm(d)

    def forward(self, query, memory, mask, reference_points, spatial_shapes, valid_ratios, reg_branches):
        intermediate, refs = [], [reference_points]
        for lid, layer in enumerate(self.layers):
            ref_in = reference_points[:, :, None] * torch.cat([valid_ratios, valid_ratios], -1)[:, None]
            query_pos = self.ref_point_head(_sine_embed(ref_in[:, :, 0, :], 128)).permute(1, 0, 2)
            query = layer(query, memory, query_pos, mask, ref_in, spatial_shapes)
            out = query.permute(1, 0, 2)
            new_ref = (reg_branches[lid](out) + inverse_sigmoid(reference_points, eps=1e-3)).sigmoid()
            reference_points = new_ref.detach()
            intermediate.append(self.norm(out))
            refs.append(new_ref)
        return torch.stack(intermediate), torch.stack(refs)


class CoDinoTransformer(nn.Module):
    def __init__(self, d=256, levels=5, num_queries=1500):
        super().__init__()
        self.level_embeds = nn.Parameter(torch.zeros(levels, d))
        self.enc_output = nn.Linear(d, d)
        self.enc_output_norm = nn.LayerNorm(d)
        self.query_embed = nn.Embedding(num_queries, d)
        self.encoder = _Encoder()
        self.decoder = _Decoder()
        self.num_queries = num_queries

    @staticmethod
    def _valid_ratio(mask):
        _, H, W = mask.shape
        vh = torch.sum(~mask[:, :, 0], 1).float() / H
        vw = torch.sum(~mask[:, 0, :], 1).float() / W
        return torch.stack([vw, vh], -1)

    @staticmethod
    def _reference_points(shapes, valid_ratios, device):
        refs = []
        for lvl, (H, W) in enumerate(shapes):
            ry, rx = torch.meshgrid(
                torch.linspace(0.5, H - 0.5, H, dtype=torch.float32, device=device),
                torch.linspace(0.5, W - 0.5, W, dtype=torch.float32, device=device),
                indexing="ij",
            )
            ry = ry.reshape(-1)[None] / (valid_ratios[:, None, lvl, 1] * H)
            rx = rx.reshape(-1)[None] / (valid_ratios[:, None, lvl, 0] * W)
            refs.append(torch.stack((rx, ry), -1))
        return torch.cat(refs, 1)[:, :, None] * valid_ratios[:, None]

    def _proposals(self, memory, mask, shapes):
        N = memory.shape[0]
        proposals, cur = [], 0
        for lvl, (H, W) in enumerate(shapes):
            m = mask[:, cur : cur + H * W].view(N, H, W, 1)
            vh = torch.sum(~m[:, :, 0, 0], 1)
            vw = torch.sum(~m[:, 0, :, 0], 1)
            gy, gx = torch.meshgrid(
                torch.linspace(0, H - 1, H, dtype=torch.float32, device=memory.device),
                torch.linspace(0, W - 1, W, dtype=torch.float32, device=memory.device),
                indexing="ij",
            )
            grid = torch.cat([gx.unsqueeze(-1), gy.unsqueeze(-1)], -1)
            scale = torch.cat([vw.unsqueeze(-1), vh.unsqueeze(-1)], 1).view(N, 1, 1, 2)
            grid = (grid.unsqueeze(0).expand(N, -1, -1, -1) + 0.5) / scale
            wh = torch.ones_like(grid) * 0.05 * (2.0**lvl)
            proposals.append(torch.cat((grid, wh), -1).view(N, -1, 4))
            cur += H * W
        props = torch.cat(proposals, 1)
        valid = ((props > 0.01) & (props < 0.99)).all(-1, keepdim=True)
        props = torch.log(props / (1 - props))
        props = props.masked_fill(mask.unsqueeze(-1), float("inf")).masked_fill(~valid, float("inf"))
        out = memory.masked_fill(mask.unsqueeze(-1), 0.0).masked_fill(~valid, 0.0)
        return self.enc_output_norm(self.enc_output(out)), props

    def forward(self, feats, masks, pos_embeds, cls_branches, reg_branches):
        feat_flat, mask_flat, pos_flat, shapes = [], [], [], []
        for lvl, (f, m, p) in enumerate(zip(feats, masks, pos_embeds)):
            bs, c, h, w = f.shape
            shapes.append((h, w))
            feat_flat.append(f.flatten(2).transpose(1, 2))
            mask_flat.append(m.flatten(1))
            pos_flat.append(p.flatten(2).transpose(1, 2) + self.level_embeds[lvl].view(1, 1, -1))
        feat_flat = torch.cat(feat_flat, 1)
        mask_flat = torch.cat(mask_flat, 1)
        pos_flat = torch.cat(pos_flat, 1)
        spatial_shapes = torch.as_tensor(shapes, dtype=torch.long, device=feat_flat.device)
        valid_ratios = torch.stack([self._valid_ratio(m) for m in masks], 1)
        ref = self._reference_points(shapes, valid_ratios, feat_flat.device)

        memory = feat_flat.permute(1, 0, 2)
        pos = pos_flat.permute(1, 0, 2)
        for layer in self.encoder.layers:
            memory = layer(memory, pos, mask_flat, ref, spatial_shapes)
        memory = memory.permute(1, 0, 2)
        bs = memory.shape[0]

        out_mem, props = self._proposals(memory, mask_flat, shapes)
        n = len(self.decoder.layers)
        enc_cls = cls_branches[n](out_mem)
        enc_coord = reg_branches[n](out_mem) + props
        topk = torch.topk(enc_cls.max(-1)[0], self.num_queries, dim=1)[1]
        ref_unact = torch.gather(enc_coord, 1, topk.unsqueeze(-1).repeat(1, 1, 4)).detach()

        query = self.query_embed.weight[:, None, :].repeat(1, bs, 1)  # (nq, bs, C)
        return self.decoder(query, memory.permute(1, 0, 2), mask_flat, ref_unact.sigmoid(), spatial_shapes, valid_ratios, reg_branches)


def _sine_position_encoding(mask, num_feats=128, temperature=20, scale=2 * math.pi, eps=1e-6):
    not_mask = 1 - mask.to(torch.int)
    y = not_mask.cumsum(1, dtype=torch.float32)
    x = not_mask.cumsum(2, dtype=torch.float32)
    y = y / (y[:, -1:, :] + eps) * scale
    x = x / (x[:, :, -1:] + eps) * scale
    dim_t = torch.arange(num_feats, dtype=torch.float32, device=mask.device)
    dim_t = temperature ** (2 * (dim_t // 2) / num_feats)
    px = x[:, :, :, None] / dim_t
    py = y[:, :, :, None] / dim_t
    B, H, W = mask.shape
    px = torch.stack((px[:, :, :, 0::2].sin(), px[:, :, :, 1::2].cos()), dim=4).view(B, H, W, -1)
    py = torch.stack((py[:, :, :, 0::2].sin(), py[:, :, :, 1::2].cos()), dim=4).view(B, H, W, -1)
    return torch.cat((py, px), dim=3).permute(0, 3, 1, 2)


def _reg_branch(d=256):
    return nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Linear(d, d), nn.ReLU(), nn.Linear(d, 4))


class QueryHead(nn.Module):
    def __init__(self, num_classes=1, d=256, num_queries=1500, num_dec_layers=6):
        super().__init__()
        self.transformer = CoDinoTransformer(d, 5, num_queries)
        self.cls_branches = nn.ModuleList(nn.Linear(d, num_classes) for _ in range(num_dec_layers + 1))
        self.reg_branches = nn.ModuleList(_reg_branch(d) for _ in range(num_dec_layers + 1))

    def forward(self, feats, img_mask):
        masks = [F.interpolate(img_mask[None].float(), size=f.shape[-2:]).to(torch.bool).squeeze(0) for f in feats]
        pos = [_sine_position_encoding(m) for m in masks]
        hs, refs = self.transformer(feats, masks, pos, self.cls_branches, self.reg_branches)
        last = hs.shape[0] - 1
        cls = self.cls_branches[last](hs[last])
        coord = (self.reg_branches[last](hs[last]) + inverse_sigmoid(refs[last], eps=1e-3)).sigmoid()
        return cls, coord  # [bs, nq, num_classes] logits, [bs, nq, 4] normalised cxcywh


class CoDINO(nn.Module):
    """Co-DINO EVA-02-L inference model: `forward(img, img_mask)` -> (class logits, normalised cxcywh boxes)."""

    def __init__(
        self, num_classes: int = 1, num_queries: int = 1500, window_block_indexes=(), embed_dim: int = 1024, depth: int = 24, num_heads: int = 16
    ):
        super().__init__()
        self.backbone = ViT(embed_dim=embed_dim, depth=depth, num_heads=num_heads, window_block_indexes=tuple(window_block_indexes))
        self.neck = SFP(c=embed_dim)
        self.query_head = QueryHead(num_classes, 256, num_queries)

    def forward(self, img, img_mask):
        return self.query_head(self.neck(self.backbone(img)), img_mask)


def soft_nms_linear(boxes, scores, iou_threshold: float = 0.8, min_score: float = 1e-3):
    """Linear soft-NMS as in MMCV (`soft_nms(..., method='linear')`): each box's score is multiplied by (1 - IoU) with
    every higher-scoring kept box it overlaps by more than `iou_threshold`; boxes below `min_score` are dropped.
    Returns (kept indices, updated scores), in the order of selection. numpy inputs."""
    import numpy as np

    boxes = boxes.astype(np.float64)
    scores = scores.astype(np.float64).copy()
    idx = np.arange(len(scores))
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    keep, kept_scores = [], []
    while len(idx):
        j = int(np.argmax(scores[idx]))
        i = idx[j]
        keep.append(i)
        kept_scores.append(scores[i])
        idx = np.delete(idx, j)
        if not len(idx):
            break
        xx1 = np.maximum(boxes[i, 0], boxes[idx, 0])
        yy1 = np.maximum(boxes[i, 1], boxes[idx, 1])
        xx2 = np.minimum(boxes[i, 2], boxes[idx, 2])
        yy2 = np.minimum(boxes[i, 3], boxes[idx, 3])
        inter = np.clip(xx2 - xx1, 0, None) * np.clip(yy2 - yy1, 0, None)
        iou = inter / np.maximum(areas[i] + areas[idx] - inter, 1e-12)
        scores[idx] *= np.where(iou > iou_threshold, 1.0 - iou, 1.0)
        idx = idx[scores[idx] >= min_score]
    return np.asarray(keep, dtype=int), np.asarray(kept_scores, dtype=np.float32)

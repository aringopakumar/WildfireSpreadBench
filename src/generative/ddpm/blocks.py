'''
Modified from https://github.com/tatakai1/classifier_free_ddim

Conditioning changed from additive feature injection to cross-attention:
  - Old: emb_cond = conv(cond_img) * mask  ->  h += emb_cond
  - New: CrossAttentionConditioner projects cond_img spatial features as
         keys/values, hidden state as queries. CFG mask zeroes the keys/values
         for unconditional passes (equivalent to attending to zeros).

This gives the model explicit spatial selectivity over the conditioning —
it can learn to attend to the fire boundary specifically rather than
receiving a uniform additive signal across all channels.

Memory note (approved change): the attention in CrossAttentionConditioner is
computed via F.scaled_dot_product_attention with q/k/v laid out as
[B, heads, N, head_dim] with head_dim contiguous-last. This lets PyTorch
dispatch to its fused flash / memory-efficient kernels, which compute the
same softmax(q k^T * scale) v in tiles WITHOUT materializing the [N, N]
score matrix (~64 GiB at 128x128). The math is identical to the original
explicit einsum + softmax; only the memory behavior changes.
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from abc import abstractmethod


def group_norm_layer(channels):
    return nn.GroupNorm(32, channels)


class TimestepBlock(nn.Module):
    @abstractmethod
    def forward(self, x, emb):
        pass


class TimestepSeqEmbedding(nn.Sequential, TimestepBlock):
    def forward(self, x, time_emb, cond_emb, mask):
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, time_emb, cond_emb, mask)
            else:
                x = layer(x)
        return x


class AttentionBlock(nn.Module):
    """Self-attention block (unchanged)."""
    def __init__(self, ch, heads=1):
        super(AttentionBlock, self).__init__()
        self.num_heads = heads
        assert ch % heads == 0

        self.norm = group_norm_layer(ch)
        self.proj = nn.Conv2d(ch, ch, kernel_size=1)
        self.qkv = nn.Conv2d(ch, ch * 3, kernel_size=1, bias=False)

    def forward(self, x):
        B, C, H, W = x.shape
        qkv = self.qkv(self.norm(x))
        q, k, v = qkv.reshape(B * self.num_heads, -1, H * W).chunk(3, dim=1)
        scale = 1. / math.sqrt(math.sqrt(C // self.num_heads))
        attention = torch.einsum("bct,bcs->bts", q * scale, k * scale)
        attention = attention.softmax(dim=-1)
        h = torch.einsum("bts,bcs->bct", attention, v).reshape(B, -1, H, W)
        return self.proj(h) + x


class CrossAttentionConditioner(nn.Module):
    """
    Cross-attention from hidden state (queries) to conditioning image (keys/values).

    The conditioning image is first projected to `inner_dim` spatial features,
    then flattened to a sequence. The hidden state at each spatial position
    attends over this sequence.

    CFG: when mask=0 for a sample, its keys and values are zeroed out,
    making the attention output a weighted sum of zeros — equivalent to
    no conditioning, matching the unconditional training pass.

    Args:
        hidden_ch:  number of channels in the hidden state (queries)
        cond_ch:    number of channels in the conditioning image
        n_heads:    number of attention heads
        head_dim:   dimension per head (inner_dim = n_heads * head_dim)
    """
    def __init__(self, hidden_ch, cond_ch, n_heads=4, head_dim=32):
        super().__init__()
        self.n_heads  = n_heads
        self.head_dim = head_dim
        inner_dim     = n_heads * head_dim
        self.scale    = head_dim ** -0.5

        # Project hidden state -> queries
        self.norm_q  = nn.GroupNorm(min(32, hidden_ch), hidden_ch)
        self.to_q    = nn.Conv2d(hidden_ch, inner_dim, 1, bias=False)

        # Project conditioning image -> keys and values
        # A small conv first to compress spatial context before projecting
        self.cond_proj = nn.Sequential(
            nn.Conv2d(cond_ch, inner_dim, 3, padding=1),
            nn.SiLU(),
        )
        self.to_k = nn.Conv2d(inner_dim, inner_dim, 1, bias=False)
        self.to_v = nn.Conv2d(inner_dim, inner_dim, 1, bias=False)

        # Project back to hidden_ch
        self.to_out = nn.Sequential(
            nn.Conv2d(inner_dim, hidden_ch, 1),
        )

    def forward(self, x, cond_img, mask):
        """
        x:        [B, hidden_ch, H, W]
        cond_img: [B, cond_ch,   H, W]  (already interpolated to match x by Unet)
        mask:     [B]  — 1 = conditional, 0 = unconditional (CFG)
        """
        B, C, H, W = x.shape
        Hh = self.n_heads
        D  = self.head_dim
        N  = H * W

        # Queries from hidden state
        q = self.to_q(self.norm_q(x))                    # [B, Hh*D, H, W]

        # Keys and values from conditioning image
        c  = self.cond_proj(cond_img)                    # [B, Hh*D, H, W]
        k  = self.to_k(c)                                # [B, Hh*D, H, W]
        v  = self.to_v(c)                                # [B, Hh*D, H, W]

        # Apply CFG mask: zero out keys/values for unconditional samples.
        # Cast the mask to k's dtype (instead of .float()) so that under AMP
        # autocast q/k/v keep one consistent dtype going into SDPA.
        cfg = mask.to(dtype=k.dtype).view(B, 1, 1, 1)
        k   = k * cfg
        v   = v * cfg

        # Reshape to [B, heads, N, head_dim] with head_dim as the CONTIGUOUS
        # LAST dimension. This exact layout is what allows PyTorch to dispatch
        # scaled_dot_product_attention to its fused flash / memory-efficient
        # kernels. (The previous permute left the last dim with stride N,
        # which silently forced the math backend — i.e. the full [N, N]
        # score matrix, ~64 GiB at 128x128.)
        q = q.reshape(B, Hh, D, N).transpose(2, 3).contiguous()   # [B, Hh, N, D]
        k = k.reshape(B, Hh, D, N).transpose(2, 3).contiguous()   # [B, Hh, N, D]
        v = v.reshape(B, Hh, D, N).transpose(2, 3).contiguous()   # [B, Hh, N, D]

        # Scaled dot-product attention — mathematically identical to
        #   attn = softmax(q @ k^T * self.scale); out = attn @ v
        # but computed in tiles, never materializing the [N, N] matrix.
        # scale=self.scale (= head_dim ** -0.5) matches the original explicit
        # scaling exactly (and is also SDPA's default for this head_dim).
        out = F.scaled_dot_product_attention(q, k, v, scale=self.scale)  # [B, Hh, N, D]

        # Back to image layout
        out = out.transpose(2, 3).reshape(B, Hh * D, H, W)        # [B, Hh*D, H, W]
        out = self.to_out(out)                                    # [B, hidden_ch, H, W]

        return out


class ResBlock(TimestepBlock):
    """
    ResBlock with cross-attention conditioning (replaces additive injection).

    The conditioning image features are attended to via CrossAttentionConditioner
    and added to the hidden state after the second conv, before the skip connection.
    The timestep embedding is still injected additively (standard practice).
    """
    def __init__(self, in_ch, out_ch, t_ch, cond_ch, dropout, n_heads=4, head_dim=32):
        super(ResBlock, self).__init__()

        self.conv_1 = nn.Sequential(
            group_norm_layer(in_ch),
            nn.SiLU(),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        )
        self.time_embedding = nn.Sequential(
            nn.SiLU(),
            nn.Linear(t_ch, out_ch)
        )
        # Cross-attention conditioning (replaces condition_conv + additive inject)
        self.cross_attn = CrossAttentionConditioner(
            hidden_ch=out_ch,
            cond_ch=cond_ch,
            n_heads=n_heads,
            head_dim=head_dim,
        )
        self.conv_2 = nn.Sequential(
            group_norm_layer(out_ch),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
        )
        self.skip_conn = nn.Conv2d(in_ch, out_ch, kernel_size=1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, t, cond_img, mask):
        h = self.conv_1(x)
        # Timestep: additive (standard)
        h = h + self.time_embedding(t)[:, :, None, None]
        h = self.conv_2(h)
        # Conditioning: cross-attention, CFG-masked
        h = h + self.cross_attn(h, cond_img, mask)
        return h + self.skip_conn(x)

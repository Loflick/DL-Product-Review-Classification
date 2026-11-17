"""Bi-LSTM for review classification."""

import math
import torch
import torch.nn as nn

BGE_DIM = 768
SENT_DIM = 5
EMBED_DIM = BGE_DIM + SENT_DIM
NUM_CLASSES = 5
NUM_STATS = 10


def reverse_valid(x, counts):
    """Reverse real reviews, keep padding."""
    B, L = x.shape[:2]
    t = torch.arange(L, device=x.device).unsqueeze(0).expand(B, L)
    c = counts.unsqueeze(1)
    idx = torch.where(t < c, c - 1 - t, t)
    return x.gather(1, idx.unsqueeze(-1).expand_as(x))


class ReviewLSTM(nn.Module):
    def __init__(self, hidden=128, d_in=256, num_layers=1, attn_heads=4, dropout=0.2):
        super().__init__()
        self.bge_proj = nn.Linear(BGE_DIM, d_in)
        self.bge_norm = nn.LayerNorm(d_in)
        self.sent_proj = nn.Linear(SENT_DIM + 1, d_in)
        self.sent_norm = nn.LayerNorm(d_in)
        self.in_drop = nn.Sequential(nn.GELU(), nn.Dropout(dropout))

        # Forward and backward LSTM
        lstm_drop = dropout if num_layers > 1 else 0.0
        self.fwd = nn.LSTM(d_in, hidden, num_layers, batch_first=True, dropout=lstm_drop)
        self.bwd = nn.LSTM(d_in, hidden, num_layers, batch_first=True, dropout=lstm_drop)

        feat = 2 * hidden
        self.attn = nn.MultiheadAttention(feat, attn_heads, dropout=dropout, batch_first=True)
        self.attn_norm = nn.LayerNorm(feat)
        self.pool_score = nn.Sequential(nn.Linear(feat, 128), nn.Tanh(), nn.Linear(128, 1))

        fused = 3 * feat + NUM_STATS
        self.head = nn.Sequential(
            nn.Linear(fused, 256), nn.LayerNorm(256), nn.GELU(), nn.Dropout(dropout),
        )
        self.cls_out = nn.Linear(256, NUM_CLASSES)
        self.reg_out = nn.Linear(256, 1)

    def forward(self, x, counts):
        """Class scores and rating."""
        B, L, _ = x.shape
        counts = counts.clamp(min=1)
        mask = torch.arange(L, device=x.device).unsqueeze(0) < counts.unsqueeze(1)
        mask_f = mask.unsqueeze(-1).float()
        n = mask_f.sum(1)

        bge, sent = x[..., :BGE_DIM], x[..., BGE_DIM:]
        stars = torch.arange(1, 6, device=x.device, dtype=x.dtype)
        star = (sent * stars).sum(-1, keepdim=True)

        # Sentiment summary
        mean_sent = (sent * mask_f).sum(1) / n
        mean_star = (star * mask_f).sum(1) / n
        std_star = torch.sqrt((((star - mean_star.unsqueeze(1)) ** 2) * mask_f).sum(1) / n + 1e-6)
        max_star = star.masked_fill(~mask.unsqueeze(-1), -1e4).max(1).values
        min_star = star.masked_fill(~mask.unsqueeze(-1), 1e4).min(1).values
        log_n = torch.log1p(n) / math.log1p(L)
        stats = torch.cat([mean_sent, (mean_star - 3) / 2, std_star / 2,
                           (min_star - 3) / 2, (max_star - 3) / 2, log_n], dim=-1)

        # Review inputs
        sent_feat = torch.cat([sent, (star - 3) / 2], dim=-1)
        h = self.in_drop(self.bge_norm(self.bge_proj(bge)) + self.sent_norm(self.sent_proj(sent_feat)))

        # Bi-LSTM
        out_f, _ = self.fwd(h)
        out_b, _ = self.bwd(reverse_valid(h, counts))
        h = torch.cat([out_f, reverse_valid(out_b, counts)], dim=-1) * mask_f

        # Self-attention across reviews
        a, _ = self.attn(h, h, h, key_padding_mask=~mask, need_weights=False)
        h = self.attn_norm(h + a) * mask_f

        # Condense to 1D
        mean_pool = h.sum(1) / n
        max_pool = h.masked_fill(~mask.unsqueeze(-1), -1e4).max(1).values
        w = torch.softmax(self.pool_score(h).masked_fill(~mask.unsqueeze(-1), -1e4), dim=1)
        attn_pool = (w * h).sum(1)

        z = self.head(torch.cat([mean_pool, max_pool, attn_pool, stats], dim=-1))
        logits = self.cls_out(z)
        rating = (mean_star + self.reg_out(z)).squeeze(-1)
        return logits, rating

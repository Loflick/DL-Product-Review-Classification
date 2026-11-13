import os
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from sklearn.metrics import cohen_kappa_score, f1_score
from sklearn.model_selection import ParameterGrid
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

try:
    import torch_directml
    device = torch_directml.device()
    IS_CUDA = False
except ImportError:
    IS_CUDA = torch.cuda.is_available()
    device = torch.device("cuda" if IS_CUDA else "cpu")

print(f"Using device: {device}")

ROOT = Path(__file__).resolve().parents[1]
MODEL_SAVE_PATH = ROOT / "transformer_bge_sent.pth"
NUM_WORKERS = 2 if (os.name != "nt" and IS_CUDA) else 0


class MemmapProductDataset(Dataset):
    def __init__(self, split_name):
        meta = np.load(ROOT / f"{split_name}_meta.npz")
        self.counts = meta["counts"]
        self.targets = meta["targets"]
        self.embed_dim = int(meta["embed_dim"])
        self.max_reviews = int(meta["max_reviews"])
        self.embeds = np.load(ROOT / f"{split_name}_embeds.npy", mmap_mode="r")

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        x = torch.from_numpy(np.array(self.embeds[idx], dtype=np.float32))
        c = torch.tensor(int(self.counts[idx]), dtype=torch.long)
        y = torch.tensor([float(self.targets[idx])], dtype=torch.float32)
        return x, c, y

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, x, mask=None):
        B, S, _ = x.size()
        Q = self.W_q(x).view(B, S, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(B, S, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(B, S, self.num_heads, self.d_k).transpose(1, 2)

        attn_output = F.scaled_dot_product_attention(Q, K, V, attn_mask=mask)
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, S, self.d_model)
        return self.W_o(attn_output)

class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.15):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = nn.Sequential(nn.Linear(d_model, d_ff),nn.GELU(),nn.Dropout(dropout),nn.Linear(d_ff, d_model),)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        x = x + self.dropout(self.self_attn(self.norm1(x), mask))
        x = x + self.dropout(self.feed_forward(self.norm2(x)))
        return x

class ReviewAttentionPooling(nn.Module):
    def __init__(self, d_model, hidden_dim=128):
        super().__init__()
        self.scorer = nn.Sequential(nn.Linear(d_model, hidden_dim),nn.Tanh(),nn.Linear(hidden_dim, 1),)

    def forward(self, review_vectors, review_mask):
        scores = self.scorer(review_vectors).masked_fill(~review_mask.unsqueeze(-1), -1e9)
        weights = torch.softmax(scores, dim=1)
        attn_pool = (weights * review_vectors).sum(dim=1)

        mask_f = review_mask.unsqueeze(-1).float()
        mean_pool = (review_vectors * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1.0)
        max_pool = review_vectors.masked_fill(~review_mask.unsqueeze(-1), -1e9).max(dim=1).values

        item_vector = torch.cat([attn_pool, mean_pool, max_pool], dim=-1)
        return item_vector, weights


class ProductReviewTransformer(nn.Module):
    def __init__(self, input_dim=773, d_model=256, num_heads=4, num_layers=2, d_ff=512, dropout=0.15):
        super().__init__()
        self.input_proj = nn.Sequential(nn.Linear(input_dim, d_model),nn.LayerNorm(d_model),nn.GELU(),nn.Dropout(dropout),)
        self.encoder_layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.pooling = ReviewAttentionPooling(d_model)
        self.fc = nn.Sequential(nn.Linear(d_model * 3, 256),nn.GELU(),nn.Dropout(dropout),nn.Linear(256, 1),)

    def forward(self, batch_reviews, counts):
        seq_len = batch_reviews.size(1)
        arange = torch.arange(seq_len, device=batch_reviews.device).unsqueeze(0)
        review_mask = arange < counts.unsqueeze(1)
        review_mask[:, 0] = True
        attn_mask = review_mask.unsqueeze(1).unsqueeze(2)

        x = self.input_proj(batch_reviews)
        for layer in self.encoder_layers:
            x = layer(x, attn_mask)

        item_vector, weights = self.pooling(x, review_mask)
        prediction = self.fc(item_vector)
        return prediction, weights


def compute_metrics(preds_np, targets_np, classes, cutoffs=None):
    mae = float(np.mean(np.abs(preds_np - targets_np)))
    disc_targets = np.round(targets_np).astype(int)

    if cutoffs is None:
        cum_probs = [np.mean(disc_targets <= c) for c in classes[:-1]]
        cutoffs = np.quantile(preds_np, cum_probs)

    sorted_cutoffs = np.sort(cutoffs)
    disc_preds = classes[np.digitize(preds_np, sorted_cutoffs)]

    off_by_one = float(np.mean(np.abs(disc_preds - disc_targets) <= 1))
    macro_f1 = float(f1_score(disc_targets, disc_preds, average="macro", zero_division=0))
    qwk = float(cohen_kappa_score(disc_targets, disc_preds, weights="quadratic"))

    return {
        "MAE": mae,
        "Off-by-one": off_by_one,
        "Macro F1": macro_f1,
        "QWK": qwk,
        "cutoffs": sorted_cutoffs,
        "discrete_preds": disc_preds,
        "discrete_targets": disc_targets,
    }


def collect_preds(model, loader):
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for bx, bc, by in loader:
            out, _ = model(bx.to(device), bc.to(device))
            preds.append(out.squeeze(-1).cpu().numpy())
            targets.append(by.squeeze(-1).numpy())
    return np.concatenate(preds), np.concatenate(targets)


def train_and_evaluate(params):
    train_ds = MemmapProductDataset("train")
    val_ds = MemmapProductDataset("validation")
    test_ds = MemmapProductDataset("test")

    classes = np.sort(np.unique(np.round(train_ds.targets).astype(int)))
    print(f"Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)} | Classes: {classes}")

    train_loader = DataLoader(train_ds, batch_size=params["batch_size"], shuffle=True, num_workers=NUM_WORKERS, pin_memory=IS_CUDA)
    val_loader = DataLoader(val_ds, batch_size=128, shuffle=False, num_workers=NUM_WORKERS, pin_memory=IS_CUDA)
    test_loader = DataLoader(test_ds, batch_size=128, shuffle=False, num_workers=NUM_WORKERS, pin_memory=IS_CUDA)

    model = ProductReviewTransformer(input_dim=train_ds.embed_dim,d_model=params["d_model"],num_heads=params["num_heads"],num_layers=params["num_layers"],d_ff=params["d_model"] * 2,dropout=params["dropout"],).to(device)

    epochs = params["epochs"]
    optimizer = torch.optim.AdamW(model.parameters(), lr=params["lr"], weight_decay=params["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.MSELoss()

    best_val_qwk = -1.0

    for epoch in range(epochs):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}", unit="batch", leave=False)
        for bx, bc, by in pbar:
            bx, bc, by = bx.to(device), bc.to(device), by.to(device)
            preds, _ = model(bx, bc)
            loss = criterion(preds, by)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        scheduler.step()

        vp, vt = collect_preds(model, val_loader)
        m = compute_metrics(vp, vt, classes)
        print(
            f"Epoch {epoch + 1:02d} | Val MAE: {m['MAE']:.4f} | "
            f"Val QWK: {m['QWK']:.4f} | Val Macro F1: {m['Macro F1']:.4f}"
        )

        if m["QWK"] > best_val_qwk:
            best_val_qwk = m["QWK"]
            torch.save(model.state_dict(), MODEL_SAVE_PATH)

    model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location="cpu", weights_only=True))
    model.to(device)

    val_preds, val_targets = collect_preds(model, val_loader)
    disc_val_targets = np.round(val_targets).astype(int)
    init_cutoffs = np.quantile(val_preds, [np.mean(disc_val_targets <= c) for c in classes[:-1]])

    def neg_qwk(c):
        dp = classes[np.digitize(val_preds, np.sort(c))]
        return -cohen_kappa_score(disc_val_targets, dp, weights="quadratic")

    opt_cutoffs = np.sort(minimize(neg_qwk, init_cutoffs, method="Nelder-Mead").x)
    print(f"\nOptimized Val Cutoffs: {[round(float(x), 4) for x in opt_cutoffs]}")

    test_preds, test_targets = collect_preds(model, test_loader)
    tm = compute_metrics(test_preds, test_targets, classes, cutoffs=opt_cutoffs)

    print("\n" + "=" * 40)
    print("           FINAL TEST RESULTS           ")
    print("Params:", params)
    print("=" * 40)
    print(f"Total Test Items:         {len(test_preds)}")
    print(f"Mean Absolute Error:      {tm['MAE']:.4f}")
    print(f"Off-by-1 Accuracy:        {tm['Off-by-one'] * 100:.2f}%")
    print(f"Macro-Averaged F1:        {tm['Macro F1']:.4f}")
    print(f"Quadratic Weighted Kappa: {tm['QWK']:.4f}")
    print("=" * 40)

    return f"{params} -> MAE: {tm['MAE']:.4f} | Macro F1: {tm['Macro F1']:.4f} | QWK: {tm['QWK']:.4f}"


if __name__ == "__main__":
    param_grid = {
        "lr": [3e-4],
        "weight_decay": [1e-2],
        "d_model": [256],
        "num_heads": [4],
        "num_layers": [2],
        "dropout": [0.15],
        "batch_size": [64],
        "epochs": [20],
    }

    results = []
    for params in ParameterGrid(param_grid):
        results.append(train_and_evaluate(params))

    print("\nSummary:")
    for r in results:
        print(r)
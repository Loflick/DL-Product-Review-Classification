import math
import os
import random
from pathlib import Path

import numpy as np
from sklearn.metrics import classification_report, cohen_kappa_score, f1_score
from sklearn.model_selection import ParameterGrid
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

IS_CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if IS_CUDA else "cpu")
print(f"Using device: {DEVICE}")

HERE = Path(__file__).resolve()
ROOT = next((p for p in HERE.parents if (p / "train_embeds.npy").exists()), HERE.parents[1])
MODEL_SAVE_PATH = (ROOT / "models" if (ROOT / "models").is_dir() else ROOT) / "transformer_bge_sent.pth"
print(f"Data ROOT: {ROOT}")

NUM_WORKERS = 2 if (os.name != "nt" and IS_CUDA) else 0
SEED = 42
CUTOFF_F1_WEIGHT = 0.5


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if IS_CUDA:
        torch.cuda.manual_seed_all(seed)


class MemmapProductDataset(Dataset):
    def __init__(self, split_name, weight_power=0.0, max_weight=5.0):
        meta = np.load(ROOT / f"{split_name}_meta.npz")
        self.counts = meta["counts"]
        self.targets = meta["targets"]
        self.embed_dim = int(meta["embed_dim"])
        self.embeds = np.load(ROOT / f"{split_name}_embeds.npy", mmap_mode="r")

        if weight_power > 0.0:
            disc = np.round(self.targets).astype(int)
            vals, cnts = np.unique(disc, return_counts=True)
            cmap = dict(zip(vals, cnts))
            w = np.array([(len(disc) / cmap[c]) ** weight_power for c in disc], dtype=np.float32)
            w = np.clip(w, 0.5, max_weight)
            self.weights = w / w.mean()
        else:
            self.weights = np.ones(len(self.targets), dtype=np.float32)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        x = torch.from_numpy(np.array(self.embeds[idx], dtype=np.float32))
        c = torch.tensor(int(self.counts[idx]), dtype=torch.long)
        y = torch.tensor([float(self.targets[idx])], dtype=torch.float32)
        w = torch.tensor([float(self.weights[idx])], dtype=torch.float32)
        return x, c, y, w


class ReviewAttentionPooling(nn.Module):
    def __init__(self, d_model, hidden_dim=128, num_pool_heads=4):
        super().__init__()
        self.scorer = nn.Sequential(nn.Linear(d_model, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, num_pool_heads))

    def forward(self, review_vectors, review_mask):
        scores = self.scorer(review_vectors).masked_fill(~review_mask.unsqueeze(-1), -1e9)
        weights = torch.softmax(scores, dim=1)

        attn_pool_flat = torch.bmm(weights.transpose(1, 2), review_vectors).reshape(review_vectors.size(0), -1)

        mask_f = review_mask.unsqueeze(-1).float()
        mean_pool = (review_vectors * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1.0)
        max_pool = review_vectors.masked_fill(~review_mask.unsqueeze(-1), -1e9).max(dim=1).values

        return torch.cat([attn_pool_flat, mean_pool, max_pool], dim=-1), weights


class ProductReviewTransformer(nn.Module):
    def __init__(self, input_dim=773, d_model=256, num_heads=4, num_layers=2, d_ff=512, dropout=0.15, num_pool_heads=4):
        super().__init__()
        self.bge_proj = nn.Linear(input_dim - 5, d_model)
        self.bge_norm = nn.LayerNorm(d_model)
        self.sent_proj = nn.Linear(6, d_model)
        self.sent_norm = nn.LayerNorm(d_model)
        self.in_act_drop = nn.Sequential(nn.GELU(), nn.Dropout(dropout))

        enc_layer = nn.TransformerEncoderLayer(d_model=d_model,nhead=num_heads,dim_feedforward=d_ff,dropout=dropout,activation="gelu",batch_first=True,norm_first=True,)
        self.encoder = nn.TransformerEncoder(enc_layer,num_layers=num_layers,norm=nn.LayerNorm(d_model),enable_nested_tensor=False,)
        self.pooling = ReviewAttentionPooling(d_model, hidden_dim=128, num_pool_heads=num_pool_heads)

        fused_dim = (num_pool_heads + 2) * d_model + 11 + num_pool_heads
        self.fc = nn.Sequential(nn.Linear(fused_dim, 256),nn.LayerNorm(256),nn.GELU(),nn.Dropout(dropout),nn.Linear(256, 1),)

    def forward(self, batch_reviews, counts):
        seq_len = batch_reviews.size(1)
        arange = torch.arange(seq_len, device=batch_reviews.device).unsqueeze(0)
        review_mask = arange < counts.unsqueeze(1)
        review_mask[:, 0] = True

        bge_x = batch_reviews[:, :, :-5]
        sent_x = batch_reviews[:, :, -5:]
        stars = torch.arange(1, 6, device=batch_reviews.device, dtype=batch_reviews.dtype)
        review_stars = (sent_x * stars).sum(dim=-1, keepdim=True)

        mask_f = review_mask.unsqueeze(-1).float()
        denom = mask_f.sum(dim=1).clamp(min=1.0)
        mean_sent_probs = (sent_x * mask_f).sum(dim=1) / denom
        mean_star = (review_stars * mask_f).sum(dim=1) / denom
        var_star = (((review_stars - mean_star.unsqueeze(1)) ** 2) * mask_f).sum(dim=1) / denom
        std_star = torch.sqrt(var_star + 1e-6)
        max_star = review_stars.masked_fill(~review_mask.unsqueeze(-1), -1e9).max(dim=1).values
        min_star = review_stars.masked_fill(~review_mask.unsqueeze(-1), 1e9).min(dim=1).values

        mean_bge = (bge_x * mask_f).sum(dim=1) / denom
        bge_consensus = torch.sqrt((mean_bge ** 2).sum(dim=-1, keepdim=True) + 1e-8)

        c_f = counts.float().unsqueeze(-1).clamp(min=1.0)
        log_c = torch.log1p(c_f) / math.log1p(seq_len)
        inv_sqrt_c = 1.0 / torch.sqrt(c_f)

        sent_feat = torch.cat([sent_x, (review_stars - 3.0) / 2.0], dim=-1)
        x = self.in_act_drop(self.bge_norm(self.bge_proj(bge_x)) + self.sent_norm(self.sent_proj(sent_feat)))
        x = self.encoder(x, src_key_padding_mask=~review_mask)

        item_vector, weights = self.pooling(x, review_mask)
        attn_stars = torch.bmm(weights.transpose(1, 2), review_stars).squeeze(-1)

        fused = torch.cat(
            [
                item_vector,
                mean_sent_probs,
                std_star / 2.0,
                (min_star - 3.0) / 2.0,
                (max_star - 3.0) / 2.0,
                (attn_stars - 3.0) / 2.0,
                bge_consensus,
                log_c,
                inv_sqrt_c,
            ],
            dim=-1,
        )
        return mean_star + self.fc(fused)


def cutoff_objective(disc_targets, disc_preds, classes, f1_weight=CUTOFF_F1_WEIGHT):
    qwk = cohen_kappa_score(disc_targets, disc_preds, weights="quadratic")
    if f1_weight <= 0.0:
        return qwk
    return qwk + f1_weight * f1_score(disc_targets, disc_preds, labels=classes, average="macro", zero_division=0)


def optimize_cutoffs(preds, disc_targets, classes, f1_weight=CUTOFF_F1_WEIGHT):
    if len(classes) < 2:
        return np.array([])
    cutoffs = np.quantile(preds, [np.mean(disc_targets <= c) for c in classes[:-1]])
    cutoffs = np.maximum.accumulate(cutoffs + np.arange(len(cutoffs)) * 1e-4)
    best_score = cutoff_objective(disc_targets, classes[np.digitize(preds, cutoffs)], classes, f1_weight)

    for _ in range(3):
        for i in range(len(cutoffs)):
            low = cutoffs[i - 1] + 1e-4 if i > 0 else float(preds.min())
            high = cutoffs[i + 1] - 1e-4 if i < len(cutoffs) - 1 else float(preds.max())
            if low >= high:
                continue
            for cand in np.linspace(low, high, 80):
                trial = cutoffs.copy()
                trial[i] = cand
                score = cutoff_objective(disc_targets, classes[np.digitize(preds, trial)], classes, f1_weight)
                if score > best_score:
                    best_score = score
                    cutoffs = trial
    return cutoffs


def compute_metrics(preds_np, targets_np, classes, cutoffs=None):
    mae = float(np.mean(np.abs(preds_np - targets_np)))
    disc_targets = np.round(targets_np).astype(int)

    if cutoffs is None:
        cutoffs = np.quantile(preds_np, [np.mean(disc_targets <= c) for c in classes[:-1]])

    disc_preds = classes[np.digitize(preds_np, np.sort(cutoffs))]
    return {
        "MAE": mae,
        "Off-by-one": float(np.mean(np.abs(disc_preds - disc_targets) <= 1)),
        "Macro F1": float(f1_score(disc_targets, disc_preds, labels=classes, average="macro", zero_division=0)),
        "QWK": float(cohen_kappa_score(disc_targets, disc_preds, weights="quadratic")),
        "discrete_preds": disc_preds,
        "discrete_targets": disc_targets,
    }


def collect_preds(model, loader, clip_min, clip_max):
    model.eval()
    preds = []
    with torch.no_grad():
        for bx, bc, _, _ in loader:
            preds.append(model(bx.to(DEVICE), bc.to(DEVICE)).squeeze(-1).cpu().numpy())
    return np.clip(np.concatenate(preds), clip_min, clip_max)


def average_state_dicts(states):
    return {
        k: torch.stack([s[k] for s in states], dim=0).mean(dim=0) if v.is_floating_point() else v
        for k, v in states[0].items()
    }


def train_and_evaluate(params, val_loader, test_loader, val_targets, test_targets, classes, target_min, target_max):
    set_seed(SEED)
    train_ds = MemmapProductDataset("train", weight_power=params["weight_power"], max_weight=params["max_weight"])
    persistent = NUM_WORKERS > 0
    train_loader = DataLoader(train_ds,batch_size=params["batch_size"],shuffle=True,num_workers=NUM_WORKERS,pin_memory=IS_CUDA,persistent_workers=persistent,)

    model = ProductReviewTransformer(input_dim=train_ds.embed_dim,d_model=params["d_model"],num_heads=params["num_heads"],num_layers=params["num_layers"],d_ff=params["d_model"] * 2,dropout=params["dropout"],num_pool_heads=params["num_pool_heads"],).to(DEVICE)

    epochs = params["epochs"]
    optimizer = torch.optim.AdamW(model.parameters(), lr=params["lr"], weight_decay=params["weight_decay"])
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[
            torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=2),
            torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs - 2), eta_min=1e-6),
        ],
        milestones=[2],
    )

    l1_weight = params["l1_weight"]
    top_ckpts = []

    for epoch in range(epochs):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}", unit="batch", leave=False)
        for bx, bc, by, bw in pbar:
            bx, bc, by, bw = bx.to(DEVICE), bc.to(DEVICE), by.to(DEVICE), bw.to(DEVICE)
            preds = model(bx, bc)
            loss = (bw * ((1.0 - l1_weight) * (preds - by) ** 2 + l1_weight * torch.abs(preds - by))).mean()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        scheduler.step()

        vp = collect_preds(model, val_loader, target_min, target_max)
        m = compute_metrics(vp, val_targets, classes)
        sel_score = m["QWK"] - m["MAE"]
        print(
            f"Epoch {epoch + 1:02d} | Val MAE: {m['MAE']:.4f} | "
            f"Val QWK: {m['QWK']:.4f} | Val Macro F1: {m['Macro F1']:.4f} | Score: {sel_score:.4f}"
        )

        cpu_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        top_ckpts.append((sel_score, cpu_state, vp))
        top_ckpts.sort(key=lambda item: item[0], reverse=True)
        top_ckpts = top_ckpts[:3]

    best_score, chosen_state, val_preds = top_ckpts[0]
    if len(top_ckpts) > 1:
        avg_state = average_state_dicts([s for _, s, _ in top_ckpts])
        model.load_state_dict(avg_state)
        model.to(DEVICE)
        vp_avg = collect_preds(model, val_loader, target_min, target_max)
        m_avg = compute_metrics(vp_avg, val_targets, classes)
        avg_score = m_avg["QWK"] - m_avg["MAE"]
        if avg_score >= best_score:
            chosen_state, val_preds = avg_state, vp_avg
            print(f"Using top-{len(top_ckpts)} averaged checkpoint (Val score: {avg_score:.4f} >= {best_score:.4f})")
        else:
            model.load_state_dict(chosen_state)
            model.to(DEVICE)
            print(f"Using single-best checkpoint (Val score: {best_score:.4f} > avg {avg_score:.4f})")

    opt_cutoffs = optimize_cutoffs(val_preds, np.round(val_targets).astype(int), classes)
    vm = compute_metrics(val_preds, val_targets, classes, cutoffs=opt_cutoffs)
    print(f"\nOptimized Val Cutoffs: {[round(float(x), 4) for x in opt_cutoffs]}")
    print(f"Val (Optimized) | MAE: {vm['MAE']:.4f} | Macro F1: {vm['Macro F1']:.4f} | QWK: {vm['QWK']:.4f}")

    test_preds = collect_preds(model, test_loader, target_min, target_max)
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
    print("\nTest per-class breakdown:")
    print(classification_report(tm["discrete_targets"], tm["discrete_preds"], labels=classes, zero_division=0, digits=4))

    return {
        "summary": f"{params} -> MAE: {tm['MAE']:.4f} | Macro F1: {tm['Macro F1']:.4f} | QWK: {tm['QWK']:.4f}",
        "val_preds": val_preds,
        "test_preds": test_preds,
        "val_score": vm["QWK"] - vm["MAE"],
        "state": chosen_state,
    }


if __name__ == "__main__":
    base_train = MemmapProductDataset("train")
    val_ds = MemmapProductDataset("validation")
    test_ds = MemmapProductDataset("test")

    disc_train = np.round(base_train.targets).astype(int)
    classes, cls_cnts = np.unique(disc_train, return_counts=True)
    print(f"Train: {len(base_train)} | Val: {len(val_ds)} | Test: {len(test_ds)} | Classes: {classes}")
    print(f"Train class counts: {dict(zip(classes.tolist(), cls_cnts.tolist()))}")

    persistent = NUM_WORKERS > 0
    val_loader = DataLoader(val_ds, batch_size=512, shuffle=False, num_workers=NUM_WORKERS, pin_memory=IS_CUDA, persistent_workers=persistent)
    test_loader = DataLoader(test_ds, batch_size=512, shuffle=False, num_workers=NUM_WORKERS, pin_memory=IS_CUDA, persistent_workers=persistent)

    val_targets = val_ds.targets.astype(np.float32)
    test_targets = test_ds.targets.astype(np.float32)
    target_min, target_max = float(base_train.targets.min()), float(base_train.targets.max())
    del base_train

    param_grid = {
        "lr": [3e-4],
        "weight_decay": [2e-2],
        "d_model": [256],
        "num_heads": [4],
        "num_layers": [2],
        "num_pool_heads": [4],
        "dropout": [0.15],
        "batch_size": [256],
        "epochs": [18],
        "weight_power": [0.0, 0.20],
        "max_weight": [5.0],
        "l1_weight": [0.25, 0.40],
    }

    run_outputs = [
        train_and_evaluate(p, val_loader, test_loader, val_targets, test_targets, classes, target_min, target_max)
        for p in ParameterGrid(param_grid)
    ]

    print("\nSummary:")
    for out in run_outputs:
        print(out["summary"])

    best_run = max(run_outputs, key=lambda r: r["val_score"])
    torch.save(best_run["state"], MODEL_SAVE_PATH)

    if len(run_outputs) > 1:
        blend_val_preds = np.mean([r["val_preds"] for r in run_outputs], axis=0)
        blend_test_preds = np.mean([r["test_preds"] for r in run_outputs], axis=0)
        blend_cutoffs = optimize_cutoffs(blend_val_preds, np.round(val_targets).astype(int), classes)
        bm = compute_metrics(blend_test_preds, test_targets, classes, cutoffs=blend_cutoffs)
        print("\n" + "=" * 40)
        print("            ENSEMBLE RESULTS            ")
        print("=" * 40)
        print(f"Mean Absolute Error:      {bm['MAE']:.4f}")
        print(f"Off-by-1 Accuracy:        {bm['Off-by-one'] * 100:.2f}%")
        print(f"Macro-Averaged F1:        {bm['Macro F1']:.4f}")
        print(f"Quadratic Weighted Kappa: {bm['QWK']:.4f}")
        print("=" * 40)
        print("\nEnsemble Test per-class breakdown:")
        print(classification_report(bm["discrete_targets"], bm["discrete_preds"], labels=classes, zero_division=0, digits=4))
import hashlib
import json
import math
import os
from pathlib import Path
import numpy as np
from sklearn.metrics import cohen_kappa_score, f1_score
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from sklearn.model_selection import ParameterGrid

try:
    import torch_directml
    device = torch_directml.device()
except ImportError:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {device}")

TRAIN_EMBED_PATH = "train_minilm_embeddings.pt"
VAL_EMBED_PATH = "validation_minilm_embeddings.pt"
TEST_EMBED_PATH = "test_minilm_embeddings.pt"

TRAIN_ITEMS_PATH = "trainItems.jsonl"
VAL_ITEMS_PATH = "validationItems.jsonl"
TEST_EMBED_PATH = "testItems.jsonl"

MODEL_SAVE_PATH = "transformer_minilm.pth"

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, ("d_model must be divisible by num_heads")
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def split_heads(self, x):
        batch_size, seq_len, _ = x.size()
        return x.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)

    def combine_heads(self, x):
        batch_size, _, seq_len, _ = x.size()
        return (x.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model))

    def forward(self, Q, K, V, mask=None):
        Q = self.split_heads(self.W_q(Q))
        K = self.split_heads(self.W_k(K))
        V = self.split_heads(self.W_v(V))

        attn_output = F.scaled_dot_product_attention(Q, K, V, attn_mask=mask)
        return self.W_o(self.combine_heads(attn_output))


class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.05):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = nn.Sequential(nn.Linear(d_model, d_ff), nn.ReLU(), nn.Linear(d_ff, d_model))
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        norm_x = self.norm1(x)
        attn_out = self.self_attn(norm_x, norm_x, norm_x, mask)
        x = x + self.dropout(attn_out)

        norm_x = self.norm2(x)
        ff_out = self.feed_forward(norm_x)
        x = x + self.dropout(ff_out)
        return x


class ReviewAttentionPooling(nn.Module):
    def __init__(self, d_model, hidden_dim=128):
        super().__init__()
        self.scorer = nn.Sequential(nn.Linear(d_model, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, 1))

    def forward(self, review_vectors, review_mask=None):
        scores = self.scorer(review_vectors)

        if review_mask is not None:
            scores = scores.masked_fill(~review_mask.unsqueeze(-1), -1e9)

        weights = torch.softmax(scores, dim=1) 
        item_vector = (weights * review_vectors).sum(dim=1)
        return item_vector, weights


class ProductReviewTransformer(nn.Module):
    def __init__(self,d_model=384,num_heads=6,num_layers=2,d_ff=512,dropout=0.05,output_dim=1):
        super().__init__()
        self.encoder_layers = nn.ModuleList(
            [
                EncoderLayer(d_model, num_heads, d_ff, dropout)
                for _ in range(num_layers)
            ]
        )
        self.pooling = ReviewAttentionPooling(d_model)
        self.fc = nn.Sequential(nn.Linear(d_model, 128),nn.ReLU(),nn.Dropout(dropout),nn.Linear(128, output_dim),)

    def forward(self, batch_reviews):
        review_mask = batch_reviews.abs().sum(dim=-1) != 0
        review_mask[:, 0] = True

        attn_mask = review_mask.unsqueeze(1).unsqueeze(2)

        x = batch_reviews
        for layer in self.encoder_layers:
            x = layer(x, attn_mask)

        item_vector, weights = self.pooling(x, review_mask=review_mask)
        prediction = self.fc(item_vector)
        return prediction, weights


class ProductDataset(Dataset):
    def __init__(self, embeddings_dict, labels_dict, max_reviews=10, embed_dim=384):
        self.asins = [
            asin for asin in labels_dict.keys() if asin in embeddings_dict
        ]
        self.embeddings = embeddings_dict
        self.labels = labels_dict
        self.max_reviews = max_reviews
        self.embed_dim = embed_dim

    def __len__(self):
        return len(self.asins)

    def __getitem__(self, idx):
        asin = self.asins[idx]
        rev_tensors = self.embeddings[asin]
        target = float(self.labels[asin])

        prod_tensor = torch.zeros((self.max_reviews, self.embed_dim), dtype=torch.float32)
        n = min(len(rev_tensors), self.max_reviews)
        if n > 0:
            prod_tensor[:n] = rev_tensors[:n]

        return prod_tensor, torch.tensor([target], dtype=torch.float32)


def load_labels(jsonl_path):
    labels = {}
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            asin = data.get("parent_asin") or data.get("asin")
            rating = data.get("rating")
            if asin and rating is not None:
                labels[asin] = float(rating)
    return labels


def compute_metrics(continuous_preds, true_targets):
    preds_np = continuous_preds.cpu().numpy()
    targets_np = true_targets.cpu().numpy()

    mae = np.mean(np.abs(preds_np - targets_np))

    cutoff_3 = np.quantile(preds_np, 0.062)
    cutoff_5 = np.quantile(preds_np, 0.684)

    discrete_preds = np.select(
        [
            preds_np < cutoff_3,
            (preds_np >= cutoff_3) & (preds_np < cutoff_5),
            preds_np >= cutoff_5,
        ],
        [3, 4, 5],
    )
    discrete_targets = np.round(targets_np).astype(int)

    off_by_one = np.mean(np.abs(discrete_preds - discrete_targets) <= 1)
    macro_f1 = f1_score(discrete_targets, discrete_preds, average="macro", zero_division=0)
    qwk = cohen_kappa_score(discrete_targets, discrete_preds, weights="quadratic")

    return {
        "MAE": mae,
        "Off-by-one": off_by_one,
        "Macro F1": macro_f1,
        "QWK": qwk,
        "cutoff_3": cutoff_3,
        "cutoff_5": cutoff_5,
        "discrete_preds": discrete_preds,
        "discrete_targets": discrete_targets,
    }

def train(lr, weight_d, layer_count, dropout_rate, batch_size):
    print("Loading datasets...")
    train_labels = load_labels(TRAIN_ITEMS_PATH)
    val_labels = load_labels(VAL_ITEMS_PATH)

    train_embeds = torch.load(TRAIN_EMBED_PATH, weights_only=False)
    val_embeds = torch.load(VAL_EMBED_PATH, weights_only=False)

    train_dataset = ProductDataset(train_embeds, train_labels, max_reviews=10)
    val_dataset = ProductDataset(val_embeds, val_labels, max_reviews=10)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

    print(f"Train samples: {len(train_dataset)}, Validation samples:" f" {len(val_dataset)}")

    model = ProductReviewTransformer(d_model=384,num_heads=6,num_layers=layer_count,d_ff=512,dropout=dropout_rate,output_dim=1).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_d)
    criterion = nn.SmoothL1Loss()

    num_epochs = 3
    best_val_mae = float("inf")

    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}", unit="batch")
        for batch_x, batch_y in pbar:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            preds, _ = model(batch_x)
            loss = criterion(preds, batch_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        val_preds, val_targets = [], []
        model.eval()
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(device)
                preds, _ = model(batch_x)

                val_preds.extend(preds.squeeze(-1).cpu().tolist())
                val_targets.extend(batch_y.squeeze(-1).tolist())

        val_preds_t = torch.tensor(val_preds)
        val_targets_t = torch.tensor(val_targets)
        metrics = compute_metrics(val_preds_t, val_targets_t)

        print(f"Epoch {epoch + 1} Done | Val MAE: {metrics['MAE']:.4f} | Val QWK:"f" {metrics['QWK']:.4f}")

        if metrics["MAE"] < best_val_mae:
            best_val_mae = metrics["MAE"]
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"--> Saved new best checkpoint to {MODEL_SAVE_PATH}")


def evaluate(lr, weight_d, layer_count, dropout_rate, batch_size):
    print("\nRunning Evaluation on Validation Set")

    useTest = False
    if not useTest:
        val_labels = load_labels(VAL_ITEMS_PATH)
        val_embeds = torch.load(VAL_EMBED_PATH, weights_only=False)

        val_dataset = ProductDataset(val_embeds, val_labels, max_reviews=10)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    else: 
        val_labels = load_labels(TRAIN_ITEMS_PATH)
        val_embeds = torch.load(TEST_EMBED_PATH, weights_only=False)

        val_dataset = ProductDataset(val_embeds, val_labels, max_reviews=10)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = ProductReviewTransformer(d_model=384,num_heads=6,num_layers=layer_count,d_ff=512,dropout=dropout_rate,output_dim=1).to(device)
    model.load_state_dict(torch.load(MODEL_SAVE_PATH, weights_only=False))
    model.eval()

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch_x, batch_y in tqdm(val_loader, desc="Evaluating"):
            batch_x = batch_x.to(device)
            preds, _ = model(batch_x)

            all_preds.extend(preds.squeeze(-1).cpu().tolist())
            all_targets.extend(batch_y.squeeze(-1).tolist())

    preds_t = torch.tensor(all_preds)
    targets_t = torch.tensor(all_targets)

    metrics = compute_metrics(preds_t, targets_t)

    print(f"\nCalculated Cutoffs -> Class 3: < {metrics['cutoff_3']:.3f}, Class 5: >= {metrics['cutoff_5']:.3f}")
    print("Target distribution:    ", np.bincount(metrics["discrete_targets"]))
    print("Prediction distribution:", np.bincount(metrics["discrete_preds"]))
    print(f"Continuous Preds -> Min: {preds_t.min():.2f}, Max: {preds_t.max():.2f}, "f"Mean: {preds_t.mean():.2f}, Std: {preds_t.std():.2f}")

    print("\n" + "=" * 35)
    print("        FINAL TEST RESULTS        ")
    print("For values ", lr, " ", weight_d, " ", layer_count, " ", dropout_rate, " ", batch_size)
    print("=" * 35)
    print(f"Total Test Items:        {len(preds_t)}")
    print(f"Mean Absolute Error:     {metrics['MAE']:.4f}")
    print(f"Off-by-1 Accuracy:       {metrics['Off-by-one'] * 100:.2f}%")
    print(f"Macro-Averaged F1:       {metrics['Macro F1']:.4f}")
    print(f"Quadratic Weighted Kappa:{metrics['QWK']:.4f}")
    print("=" * 35)

    return f"Total Test Items:        {len(preds_t)} Mean Absolute Error:     {metrics['MAE']:.4f} Off-by-1 Accuracy:       {metrics['Off-by-one'] * 100:.2f}% Macro-Averaged F1:       {metrics['Macro F1']:.4f} Quadratic Weighted Kappa:{metrics['QWK']:.4f}"  


if __name__ == "__main__":
    param_grid = {
        "lr": [0.0001, 0.0003, 0.0005, 0.0008],
        "weight_decay": [0.0, 0.0001],
        "num_layers": [1, 2],
        "dropout": [0.0, 0.05],
        "batch_size": [16, 32, 64]
    }
    param_grid = {
        "lr": [0.0001],
        "weight_decay": [0.0],
        "num_layers": [1],
        "dropout": [0.0],
        "batch_size": [32]
    }

    retString = []
    for params in ParameterGrid(param_grid):
        train(params["lr"], params["weight_decay"], params["num_layers"], params["dropout"], params["batch_size"])
        retString.append(evaluate(params["lr"], params["weight_decay"], params["num_layers"], params["dropout"], params["batch_size"]))

    for val in retString:
        print(val)
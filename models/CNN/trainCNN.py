"""Training CNN for Review Classification """
#AI Disclamier: AI was used in both the development and understandment of this code

import sys
import json
import copy
import h5py
import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import TensorDataset, DataLoader
from pathlib import Path
from collections import Counter
from models.CNN.CNN import TextCNN, HParams

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))
from preprocessing.gloveEmbedTest import (
    CATEGORIES,
    TRAIN_ITEMS_OUTPUT,
    VALIDATION_ITEMS_OUTPUT,
    TEST_ITEMS_OUTPUT,
    TRAIN_EMBED_OUTPUT,
    VALIDATION_EMBED_OUTPUT,
    TEST_EMBED_OUTPUT,
)

# data splits: (items file, embeddings file)
SPLIT = {
    "train": (ROOT / TRAIN_ITEMS_OUTPUT, ROOT / TRAIN_EMBED_OUTPUT),
    "validation": (ROOT / VALIDATION_ITEMS_OUTPUT, ROOT / VALIDATION_EMBED_OUTPUT),
    "test": (ROOT / TEST_ITEMS_OUTPUT, ROOT / TEST_EMBED_OUTPUT),
}

device = "cuda" if torch.cuda.is_available() else "cpu"
MAX_LEN = 100 #reviews should not be that long I believe 
NUM_CLASSES = 5          # labels 0-4 = 1-5 stars


# ----------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------
def pad_matrix(m, max_len=MAX_LEN):
    # float16 storage halves memory 
    out = np.zeros((max_len, 300), dtype=np.float16)
    n = min(len(m), max_len)
    out[:n] = m[:n]
    return out


def load_items(path):
    text = open(path, encoding="utf-8").read()
    dec, i, n, items = json.JSONDecoder(), 0, len(text), []
    while i < n:
        while i < n and text[i] in " \t\r\n, []":
            i += 1
        if i >= n:
            break
        obj, i = dec.raw_decode(text, i)
        items.append(obj)
    return items


def build_split(split, items_per_category, max_per_item, seed=0):
    """
    Samples `items_per_category` random items from each category and keeps up to
    `max_per_item` reviews of each. Every review carries its item's rating.
    Returns X (N, 100, 300) float16, y (N,) long, ids (N,) long = which item each review is from.
    """
    items_path, embed_path = SPLIT[split]
    lookup = {(it["category"], it["parent_asin"]): it for it in load_items(items_path)}
    rng = np.random.default_rng(seed) #ranndomize
    X, y, ids, next_id = [], [], [], 0

    with h5py.File(embed_path, "r") as f:
        for cat in CATEGORIES:
            asins = list(f[cat].keys())
            rng.shuffle(asins)                           # random items, not file order
            n_items = 0
            for asin in asins:
                item = lookup.get((cat, asin))
                if item is None:
                    continue
                grp = f[cat][asin]
                keys = sorted(grp.keys(), key=lambda k: int(k.split("_")[1]))[:max_per_item]
                for k in keys:
                    X.append(pad_matrix(grp[k][:]))
                    y.append(int(item["rating"]))        # already 0-4
                    ids.append(next_id)
                next_id += 1
                n_items += 1
                if n_items >= items_per_category:
                    break
            print(f"  [{split}] {cat}: {n_items} items")

    assert min(y) >= 0 and max(y) <= 4, f"bad labels: {sorted(set(y))}"
    item_labels = {i: l for i, l in zip(ids, y)}
    print(f"  [{split}] {len(y)} reviews, {next_id} items | "
          f"review labels {sorted(Counter(y).items())} | "
          f"item labels {sorted(Counter(item_labels.values()).items())}")
    return (torch.from_numpy(np.stack(X)),
            torch.tensor(y, dtype=torch.long),
            torch.tensor(ids, dtype=torch.long))


# ----------------------------------------------------------------------
# Prediction and metrics 
# ----------------------------------------------------------------------
@torch.no_grad()
def predict_logits(model, X, batch_size=256):
    model.eval()
    return torch.cat([model(X[i:i + batch_size].to(device).float()).cpu()
                      for i in range(0, len(X), batch_size)])


def item_predictions(logits, y, ids):
    """Average each item's review probabilities, predict one rating per item."""
    probs = logits.softmax(1)
    uniq, inv = torch.unique(ids, return_inverse=True)
    M = len(uniq)
    summed = torch.zeros(M, probs.shape[1]).index_add_(0, inv, probs)
    counts = torch.zeros(M).index_add_(0, inv, torch.ones(len(inv)))
    item_pred = (summed / counts[:, None]).argmax(1)
    item_y = torch.zeros(M, dtype=torch.long)
    item_y[inv] = y                                      # same label for every review of an item
    return item_pred, item_y


def compute_metrics(y_true, y_pred, K=NUM_CLASSES):
    y_true, y_pred = y_true.cpu(), y_pred.cpu()
    O = torch.bincount(y_true * K + y_pred, minlength=K * K).reshape(K, K).float()

    # Quadratic weighted kappa
    idx = torch.arange(K).float()
    W = (idx[:, None] - idx[None, :]) ** 2 / (K - 1) ** 2
    E = torch.outer(O.sum(1), O.sum(0)) / O.sum()
    denom = (W * E).sum()
    qwk = 1 - (W * O).sum() / denom if denom > 0 else torch.tensor(0.0)

    # Macro-F1
    tp = O.diag()
    fp = O.sum(0) - tp
    fn = O.sum(1) - tp
    f1 = torch.where(2 * tp + fp + fn > 0, 2 * tp / (2 * tp + fp + fn), torch.zeros(K))

    # MAE and off-by-1
    err = (y_true - y_pred).abs().float()
    return {"qwk": qwk.item(), "macro_f1": f1.mean().item(), "mae": err.mean().item(),
            "off_by_1": (err <= 1).float().mean().item(),
            "acc": (err == 0).float().mean().item(),
            "per_class_f1": [round(v, 3) for v in f1.tolist()]}


def evaluate_items(model, data):
    X, y, ids = data
    pred, item_y = item_predictions(predict_logits(model, X), y, ids)
    return compute_metrics(item_y, pred)


def show_metrics(name, m):
    print(f"{name:<24} QWK {m['qwk']:.3f} | macro-F1 {m['macro_f1']:.3f} | "
          f"MAE {m['mae']:.3f} | off-by-1 {m['off_by_1']:.3f} | acc {m['acc']:.3f}")
    print(f"{'':<24} per-class F1: {m['per_class_f1']}")


# ----------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------
def train(hp, train_data, val_data, class_weights=None, patience=3, verbose=True):
    X_tr, y_tr = train_data[0], train_data[1]
    model = TextCNN(NUM_CLASSES, hp).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=hp.lr, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss(
        weight=class_weights.to(device) if class_weights is not None else None)
    loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=hp.batch_size, shuffle=True)

    best_qwk, best_state, bad = -1.0, None, 0
    for epoch in range(hp.epochs):
        model.train()
        total = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(device).float(), yb.to(device)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
            total += loss.item() * len(xb)

        m = evaluate_items(model, val_data)              # item-level validation metrics
        if verbose:
            print(f"epoch {epoch + 1}/{hp.epochs} loss {total / len(X_tr):.4f} "
                  f"val QWK {m['qwk']:.3f} macro-F1 {m['macro_f1']:.3f} MAE {m['mae']:.3f}")

        if m["qwk"] > best_qwk:
            best_qwk, best_state, bad = m["qwk"], copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= patience:  #if bad is greater than what ever pateience is break the loop and use the current QWK
                break

    model.load_state_dict(best_state)
    return model, best_qwk


def sample_hp(rng):
    return HParams(
        lr=float(10 ** rng.uniform(-4, -2.5)),           
        batch_size=int(rng.choice([32, 64, 128])),
        epochs=int(rng.choice([5, 8, 12])),
        activation=str(rng.choice(["relu", "tanh", "gelu"])),
        n_filters=int(rng.choice([50, 100, 200])),
        n_neurons=int(rng.choice([64, 128, 256])),
        dropout=float(rng.uniform(0.3, 0.6)),
    )


if __name__ == "__main__":
    N_TRIALS = 12 #find the hyparameters with trials; once the best parameters are found run on the test data set once
    RUN_SEARCH = True

    torch.manual_seed(0)
    print("Loading data...")
    train_data = build_split("train", items_per_category=2500, max_per_item=2)
    val_data = build_split("validation", items_per_category=600, max_per_item=8)

    counts = torch.bincount(train_data[1], minlength=NUM_CLASSES).float()
    weights = (counts.sum() / counts) ** 0.5             # sqrt-inverse-frequency class weights
    weights = weights / weights.mean()
    print("class weights:", [round(w, 2) for w in weights.tolist()])

    # Baseline: always predict the most common training class
    val_item_y = item_predictions(torch.zeros(len(val_data[1]), NUM_CLASSES),
                                  val_data[1], val_data[2])[1]
    show_metrics("always-majority (val)",
                 compute_metrics(val_item_y, torch.full_like(val_item_y, counts.argmax().item())))

    # ---- random search ----
    if RUN_SEARCH:
        rng = np.random.default_rng(0)
        results = []
        for t in range(N_TRIALS):
            hp = sample_hp(rng)
            print(f"\n=== trial {t + 1}/{N_TRIALS}: {hp}")
            _, q = train(hp, train_data, val_data, weights, verbose=False)
            results.append((q, hp))
            print(f"==> best val QWK {q:.3f}")
        results.sort(key=lambda r: -r[0])
        print("\nTop 3:")
        for q, hp in results[:3]:
            print(f"{q:.3f}  {hp}")
        best_hp = results[0][1]
    else:
        best_hp = HParams(lr=3e-4, batch_size=32, epochs=10, activation="relu",
                          n_filters=100, n_neurons=64, dropout=0.5)

    # ---- final: 3 seeds with the best config ----
    print(f"\nFinal config: {best_hp}")
    models, qwks = [], []
    for seed in (0, 1, 2):
        torch.manual_seed(seed)
        model, q = train(best_hp, train_data, val_data, weights)
        models.append(model)
        qwks.append(q)
    print(f"val QWK {np.mean(qwks):.3f} +/- {np.std(qwks):.3f}")

    #test: run ONCE
    test_data = build_split("test", items_per_category=10**9, max_per_item=4)
    test_item_y = item_predictions(torch.zeros(len(test_data[1]), NUM_CLASSES),
                                   test_data[1], test_data[2])[1]
    print(f"\nTotal test items: {len(test_item_y)}")
    show_metrics("always-majority (test)",
                 compute_metrics(test_item_y, torch.full_like(test_item_y, counts.argmax().item())))
    all_m = []
    for s, mdl in enumerate(models):
        m = evaluate_items(mdl, test_data)
        all_m.append(m)
        show_metrics(f"CNN test seed {s}", m)
        
    print("\n===================================")
    print("        FINAL TEST RESULTS (CNN)")
    print("===================================")
    print(f"Total Test Items:         {len(test_item_y)}")
    print(f"Mean Absolute Error:      {np.mean([m['mae'] for m in all_m]):.4f}")
    print(f"Off-by-1 Accuracy:        {100 * np.mean([m['off_by_1'] for m in all_m]):.2f}%")
    print(f"Macro-Averaged F1:        {np.mean([m['macro_f1'] for m in all_m]):.4f}")
    print(f"Quadratic Weighted Kappa: {np.mean([m['qwk'] for m in all_m]):.4f}")
    print("===================================")

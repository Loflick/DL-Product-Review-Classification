"""Trains and evaluates the Bi-LSTM.

Run from anywhere:
    python models/LSTM/runLSTM.py                  # random search, then final 3-seed run + test
    python models/LSTM/runLSTM.py --trials 0       # skip search; final run with the best saved config
                                                   # (or the slide baseline if nothing is saved)

Reads {train,validation,test}_embeds.npy and *_meta.npz from the repo root.

Evaluation matches models/transformer/transformer.py so results are comparable:
MAE is on the continuous rating score vs the item's float rating, classes are
round(rating) in 1..5, and class cutoffs are tuned on validation (QWK + 0.8 * macro-F1)
and then applied to test. The test set is only used once, at the end.
"""

import argparse
import copy
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import classification_report, cohen_kappa_score, f1_score

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
from LSTM import NUM_CLASSES, ReviewLSTM  # noqa: E402

DEVICE = torch.device("cuda" if torch.cuda.is_available()
                      else "mps" if torch.backends.mps.is_available() else "cpu")
RESULTS_PATH = HERE / "search_results.json"
MODEL_SAVE_PATH = HERE / "lstm_bge_sent.pth"
CLASSES = np.arange(1, NUM_CLASSES + 1)
CUTOFF_F1_WEIGHT = 0.8

# Slide baseline: 1 Bi-LSTM layer, 128 hidden per direction (256 combined)
BASELINE = {
    "hidden": 128, "d_in": 256, "num_layers": 1, "attn_heads": 4, "dropout": 0.2,
    "lr": 1e-3, "weight_decay": 1e-2, "batch_size": 256, "epochs": 20,
    "reg_weight": 1.0, "soft_labels": True, "class_weight_power": 0.0, "shuffle_reviews": True,
}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ----------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------
def load_split(name):
    meta = np.load(ROOT / f"{name}_meta.npz")
    x = torch.from_numpy(np.load(ROOT / f"{name}_embeds.npy"))            # float16, (N, 10, 773)
    counts = torch.from_numpy(meta["counts"].astype(np.int64))
    ratings = meta["targets"].astype(np.float32)
    return x, counts, ratings


def hard_class(r):
    """Float rating -> class index 0..4 (round to nearest star, as the other models do)."""
    return torch.clamp(torch.round(r) - 1, 0, NUM_CLASSES - 1).long()


def soft_targets(r):
    """Spread probability between the two nearest stars, e.g. 4.3 -> 70% '4', 30% '5'."""
    r = r.clamp(1.0, 5.0)
    lo = torch.floor(r).long().clamp(max=NUM_CLASSES - 1)                  # 1..4
    frac = (r - lo.float()).unsqueeze(1)
    t = torch.zeros(len(r), NUM_CLASSES, device=r.device)
    t.scatter_(1, (lo - 1).unsqueeze(1), 1 - frac)
    t.scatter_add_(1, lo.unsqueeze(1), frac)
    return t


def shuffle_reviews(x, counts):
    """Random order of each item's real reviews; padding stays at the end."""
    B, L = x.shape[:2]
    t = torch.arange(L, device=x.device).unsqueeze(0).expand(B, L)
    keys = torch.where(t < counts.unsqueeze(1), torch.rand(B, L, device=x.device), 1.0 + t.float())
    idx = keys.argsort(dim=1)
    return x.gather(1, idx.unsqueeze(-1).expand_as(x))


# ----------------------------------------------------------------------
# Metrics (same definitions as models/transformer/transformer.py)
# ----------------------------------------------------------------------
def cutoff_objective(y, pred, f1_weight=CUTOFF_F1_WEIGHT):
    qwk = cohen_kappa_score(y, pred, weights="quadratic")
    return qwk + f1_weight * f1_score(y, pred, labels=CLASSES, average="macro", zero_division=0)


def optimize_cutoffs(scores, y):
    cutoffs = np.quantile(scores, [np.mean(y <= c) for c in CLASSES[:-1]])
    cutoffs = np.maximum.accumulate(cutoffs + np.arange(len(cutoffs)) * 1e-4)
    best = cutoff_objective(y, CLASSES[np.digitize(scores, cutoffs)])
    for _ in range(3):
        for i in range(len(cutoffs)):
            low = cutoffs[i - 1] + 1e-4 if i > 0 else float(scores.min())
            high = cutoffs[i + 1] - 1e-4 if i < len(cutoffs) - 1 else float(scores.max())
            if low >= high:
                continue
            for cand in np.linspace(low, high, 80):
                trial = cutoffs.copy()
                trial[i] = cand
                s = cutoff_objective(y, CLASSES[np.digitize(scores, trial)])
                if s > best:
                    best, cutoffs = s, trial
    return cutoffs


def compute_metrics(scores, ratings, cutoffs=None):
    y = np.round(ratings).astype(int)
    if cutoffs is None:
        cutoffs = np.quantile(scores, [np.mean(y <= c) for c in CLASSES[:-1]])
    pred = CLASSES[np.digitize(scores, np.sort(cutoffs))]
    return {
        "MAE": float(np.mean(np.abs(scores - ratings))),
        "Off-by-one": float(np.mean(np.abs(pred - y) <= 1)),
        "Macro F1": float(f1_score(y, pred, labels=CLASSES, average="macro", zero_division=0)),
        "QWK": float(cohen_kappa_score(y, pred, weights="quadratic")),
        "pred": pred, "y": y,
    }


def selection_score(m):
    return m["QWK"] - m["MAE"]


# ----------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------
@torch.no_grad()
def predict(model, data, batch_size=1024):
    """Returns (expected star from class probs, regression rating), both numpy (N,)."""
    model.eval()
    x, counts, _ = data
    ev, reg = [], []
    for s in range(0, len(x), batch_size):
        xb = x[s:s + batch_size].to(DEVICE).float()
        cb = counts[s:s + batch_size].to(DEVICE)
        logits, rating = model(xb, cb)
        probs = logits.float().softmax(-1)
        ev.append((probs * torch.arange(1, 6, device=DEVICE)).sum(-1).cpu())
        reg.append(rating.float().cpu())
    return torch.cat(ev).numpy(), torch.cat(reg).numpy()


def blend(ev, reg, alpha):
    return np.clip(alpha * reg + (1 - alpha) * ev, 1.0, 5.0)


def best_blend(ev, reg, ratings):
    """Pick how to mix the two outputs on validation."""
    best = None
    for alpha in (0.0, 0.25, 0.5, 0.75, 1.0):
        m = compute_metrics(blend(ev, reg, alpha), ratings)
        if best is None or selection_score(m) > best[0]:
            best = (selection_score(m), alpha, m)
    return best


def train(p, train_data, val_data, seed=0, patience=4, verbose=True):
    set_seed(seed)
    x_tr, c_tr, r_tr = train_data
    r_tr_t = torch.from_numpy(r_tr)

    # Optional per-item weights from class frequency
    cls = hard_class(r_tr_t)
    freq = torch.bincount(cls, minlength=NUM_CLASSES).float()
    class_w = ((freq.sum() / freq.clamp(min=1)) ** p["class_weight_power"]).clamp(max=20.0)
    item_w = class_w[cls]
    item_w = item_w / item_w.mean()

    model = ReviewLSTM(hidden=p["hidden"], d_in=p["d_in"], num_layers=p["num_layers"],
                       attn_heads=p["attn_heads"], dropout=p["dropout"]).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=p["lr"], weight_decay=p["weight_decay"])
    steps_per_epoch = math.ceil(len(x_tr) / p["batch_size"])
    total = steps_per_epoch * p["epochs"]
    warm = steps_per_epoch
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm
        else 0.02 + 0.98 * 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm))))

    best = (-1e9, None, 0.5, 0)
    bad = 0
    for epoch in range(p["epochs"]):
        model.train()
        t0 = time.time()
        order = torch.randperm(len(x_tr))
        run_loss = 0.0
        for s in range(0, len(order), p["batch_size"]):
            idx = order[s:s + p["batch_size"]]
            xb = x_tr[idx].to(DEVICE).float()
            cb = c_tr[idx].to(DEVICE)
            rb = r_tr_t[idx].to(DEVICE)
            wb = item_w[idx].to(DEVICE)
            if p["shuffle_reviews"]:
                xb = shuffle_reviews(xb, cb)

            logits, rating = model(xb, cb)
            if p["soft_labels"]:
                ce = -(soft_targets(rb) * F.log_softmax(logits, -1)).sum(-1)
            else:
                ce = F.cross_entropy(logits, hard_class(rb), reduction="none")
            diff = rating - rb
            reg = 0.7 * diff ** 2 + 0.3 * diff.abs()
            loss = (wb * (ce + p["reg_weight"] * reg)).mean()

            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            run_loss += loss.item() * len(idx)

        ev, reg_v = predict(model, val_data)
        score, alpha, m = best_blend(ev, reg_v, val_data[2])
        if verbose:
            print(f"  epoch {epoch + 1:02d} loss {run_loss / len(x_tr):.4f} | val MAE {m['MAE']:.4f} "
                  f"QWK {m['QWK']:.4f} F1 {m['Macro F1']:.4f} | score {score:.4f} (alpha {alpha}) "
                  f"| {time.time() - t0:.0f}s", flush=True)
        if score > best[0]:
            best = (score, copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()}),
                    alpha, epoch + 1)
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break

    model.load_state_dict(best[1])
    return model, {"val_score": best[0], "alpha": best[2], "best_epoch": best[3]}


# ----------------------------------------------------------------------
# Search and final evaluation
# ----------------------------------------------------------------------
def sample_params(rng):
    return {
        "hidden": int(rng.choice([128, 192, 256])),
        "d_in": int(rng.choice([256, 384])),
        "num_layers": int(rng.choice([1, 2])),
        "attn_heads": 4,
        "dropout": round(float(rng.uniform(0.1, 0.35)), 3),
        "lr": float(f"{10 ** rng.uniform(-3.7, -2.7):.2e}"),
        "weight_decay": float(f"{10 ** rng.uniform(-3, -1.3):.2e}"),
        "batch_size": int(rng.choice([128, 256, 512])),
        "epochs": 20,
        "reg_weight": float(rng.choice([0.5, 1.0, 2.0])),
        "soft_labels": bool(rng.choice([True, False])),
        "class_weight_power": float(rng.choice([0.0, 0.15, 0.3])),
        "shuffle_reviews": bool(rng.choice([True, False])),
    }


def load_results():
    return json.loads(RESULTS_PATH.read_text()) if RESULTS_PATH.exists() else []


def print_results(title, m, n):
    print("\n" + "=" * 40)
    print(f"{title:^40}")
    print("=" * 40)
    print(f"Total Test Items:         {n}")
    print(f"Mean Absolute Error:      {m['MAE']:.4f}")
    print(f"Off-by-1 Accuracy:        {m['Off-by-one'] * 100:.2f}%")
    print(f"Macro-Averaged F1:        {m['Macro F1']:.4f}")
    print(f"Quadratic Weighted Kappa: {m['QWK']:.4f}")
    print("=" * 40, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=12, help="random-search trials (baseline counts as one)")
    ap.add_argument("--seeds", type=int, default=3, help="seeds for the final run")
    args = ap.parse_args()

    print(f"Device: {DEVICE} | data: {ROOT}")
    train_data, val_data = load_split("train"), load_split("validation")
    print(f"Train {len(train_data[0])} | Val {len(val_data[0])} items", flush=True)

    # ---- random search (resumable) ----
    results = load_results()
    rng = np.random.default_rng(0)
    planned = [BASELINE] + [sample_params(rng) for _ in range(max(0, args.trials - 1))]
    for t, p in enumerate(planned[:args.trials]):
        if any(r["params"] == p for r in results):
            continue
        print(f"\n=== trial {t + 1}/{args.trials}: {p}", flush=True)
        t0 = time.time()
        _, info = train(p, train_data, val_data, seed=0)
        results.append({"params": p, **info, "minutes": round((time.time() - t0) / 60, 1)})
        RESULTS_PATH.write_text(json.dumps(results, indent=1))
        print(f"==> val score {info['val_score']:.4f} at epoch {info['best_epoch']}", flush=True)

    results.sort(key=lambda r: -r["val_score"])
    if results:
        print("\nTop configs (val QWK - MAE):")
        for r in results[:5]:
            print(f"  {r['val_score']:.4f}  {r['params']}")
    best_p = results[0]["params"] if results else BASELINE

    # ---- final: several seeds with the best config ----
    print(f"\nFinal config: {best_p}", flush=True)
    runs = []
    for seed in range(args.seeds):
        print(f"\n--- seed {seed}", flush=True)
        model, info = train(best_p, train_data, val_data, seed=seed)
        ev_v, reg_v = predict(model, val_data)
        runs.append({"model": model, "alpha": info["alpha"], "val": blend(ev_v, reg_v, info["alpha"]),
                     "val_score": info["val_score"]})

    best_run = max(runs, key=lambda r: r["val_score"])
    torch.save({"params": best_p, "alpha": best_run["alpha"], "state": best_run["model"].state_dict()},
               MODEL_SAVE_PATH)

    # ---- test, once ----
    del train_data
    test_data = load_split("test")
    y_val = np.round(val_data[2]).astype(int)
    seed_metrics, test_scores = [], []
    for r in runs:
        ev_t, reg_t = predict(r["model"], test_data)
        r["test"] = blend(ev_t, reg_t, r["alpha"])
        cut = optimize_cutoffs(r["val"], y_val)
        seed_metrics.append(compute_metrics(r["test"], test_data[2], cut))

    for i, m in enumerate(seed_metrics):
        print(f"seed {i}: MAE {m['MAE']:.4f} | off-by-1 {m['Off-by-one'] * 100:.2f}% | "
              f"macro-F1 {m['Macro F1']:.4f} | QWK {m['QWK']:.4f}")
    mean = {k: float(np.mean([m[k] for m in seed_metrics])) for k in ("MAE", "Off-by-one", "Macro F1", "QWK")}
    print_results(f"FINAL TEST RESULTS (LSTM, mean of {len(runs)} seeds)", mean, len(test_data[2]))

    ens_val = np.mean([r["val"] for r in runs], axis=0)
    ens_test = np.mean([r["test"] for r in runs], axis=0)
    ens = compute_metrics(ens_test, test_data[2], optimize_cutoffs(ens_val, y_val))
    print_results(f"ENSEMBLE TEST RESULTS (LSTM, {len(runs)} seeds)", ens, len(test_data[2]))
    print("\nEnsemble per-class breakdown:")
    print(classification_report(ens["y"], ens["pred"], labels=CLASSES, zero_division=0, digits=4))
    print(f"Params: {best_p}")


if __name__ == "__main__":
    main()

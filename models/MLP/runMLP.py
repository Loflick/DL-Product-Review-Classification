"""Runs MLP training and testing."""

import random
import numpy as np
import torch
from sklearn.metrics import f1_score
from scipy.optimize import minimize
from sklearn.model_selection import ParameterGrid
from MLP import CLASSES, MLP, NUM_CLASSES, compute_qwk


TRAIN_EMBED_PATH = "train_embeds.npy"
VAL_EMBED_PATH = "validation_embeds.npy"
TEST_EMBED_PATH = "test_embeds.npy"

TRAIN_META_PATH = "train_meta.npz"
VAL_META_PATH = "validation_meta.npz"
TEST_META_PATH = "test_meta.npz"

MODEL_SAVE_PATH = "mlp_bge_sent.pth"


def to_class(rating):
    """Rating to class."""
    # Round rating stars
    stars = round(float(rating))

    # 0-4 classes
    label = stars - 1

    # Set range bounds
    if label < 0:
        label = 0
    if label > NUM_CLASSES - 1:
        label = NUM_CLASSES - 1

    return label


def load_data(embed_path, meta_path):
    """Load reviews, classes, and ratings."""
    meta = np.load(meta_path)
    x = np.load(embed_path)

    targets = meta["targets"]
    y = np.zeros(len(targets), dtype=np.int64)

    # Load classes
    for i in range(len(targets)):
        y[i] = to_class(targets[i])

    return x, y, targets


def compute_metrics(y_true, y_pred):
    """Computes evaluation metrics."""
    # error = |y - y_hat|
    error = np.abs(y_true - y_pred)

    # Quadratic weighted kappa
    qwk = compute_qwk(y_true, y_pred)

    # Macro-averaged F1
    f1 = f1_score(y_true, y_pred, average="macro", labels=CLASSES, zero_division=0)

    # Off-by-1 = mean(error <= 1)
    off_by_1 = float((error <= 1).mean())

    return {
        "qwk": qwk,
        "macro_f1": f1,
        "off_by_1_accuracy": off_by_1,
    }


def build_model(params):
    """Builds the MLP."""
    sizes = (256, 128, 64)
    hidden = sizes[:params["num_layers"]]

    return MLP(hidden=hidden)


def train(params):
    """Trains the MLP."""
    x_train, y_train, _ = load_data(TRAIN_EMBED_PATH, TRAIN_META_PATH)
    x_val, y_val, _ = load_data(VAL_EMBED_PATH, VAL_META_PATH)

    model = build_model(params)

    model.fit(
        x_train, y_train,
        x_val, y_val,
        params["lr"], params["weight_decay"], params["batch_size"], params["epochs"],
        MODEL_SAVE_PATH,
    )


def scores_to_classes(scores, cutoffs):
    """Convert scores to classes."""
    cutoffs = np.sort(cutoffs)
    return np.digitize(scores, cutoffs)


def qwk_loss(cutoffs, scores, y):
    """Loss for cutoffs."""
    pred = scores_to_classes(scores, cutoffs)
    qwk = compute_qwk(y, pred)

    return 1 - qwk


def find_cutoffs(scores, y):
    """Find cutoffs with best QWK."""
    # Search through class shares to find start
    shares = []
    for c in range(NUM_CLASSES - 1):
        shares.append(np.mean(y <= c))

    start = np.quantile(scores, shares)

    # Search for lowest loss
    result = minimize(qwk_loss, start, args=(scores, y), method="Nelder-Mead")

    return np.sort(result.x)


def evaluate(params):
    """Print test results."""
    x_val, y_val, _ = load_data(VAL_EMBED_PATH, VAL_META_PATH)
    x_test, y_test, ratings = load_data(TEST_EMBED_PATH, TEST_META_PATH)

    model = build_model(params)
    model.load(MODEL_SAVE_PATH)

    # Tune on validation set
    val_scores = model.predict(x_val)
    cutoffs = find_cutoffs(val_scores, y_val)
    print("Optimized Val Cutoffs:", cutoffs.round(4).tolist())

    # Classes from test cutoffs
    test_scores = model.predict(x_test)
    y_pred = scores_to_classes(test_scores, cutoffs)
    metrics = compute_metrics(y_test, y_pred)

    # MAE
    rating_pred = test_scores + 1
    mae = np.abs(rating_pred - ratings).mean()

    f1_each = f1_score(y_test, y_pred, average=None, labels=CLASSES, zero_division=0)
    print("F1 per class:", f1_each.round(3))

    print("\n" + "=" * 40)
    print("           FINAL TEST RESULTS")
    print(f"Params: {params}")
    print("=" * 40)
    print(f"Total Test Items:         {len(y_pred)}")
    print(f"Mean Absolute Error:      {mae:.4f}")
    off_by_1 = metrics["off_by_1_accuracy"] * 100
    print(f"Off-by-1 Accuracy:        {off_by_1:.2f}%")
    print(f"Macro-Averaged F1:        {metrics['macro_f1']:.4f}")
    print(f"Quadratic Weighted Kappa: {metrics['qwk']:.4f}")
    print("=" * 40)


if __name__ == "__main__":
    # Fix random seed
    torch.manual_seed(42)

    # Search space
    param_grid = {
        "batch_size": [128, 256, 512],
        "epochs": [20],
        "lr": [0.0003, 0.001, 0.003],
        "num_layers": [1, 2, 3],
        "weight_decay": [0.0001, 0.001, 0.02],
    }

    # Single run
    param_grid = {
        "batch_size": [256],
        "epochs": [20],
        "lr": [0.0003],
        "num_layers": [2],
        "weight_decay": [0.02],
    }

    # Random search trials
    trials = None

    combos = list(ParameterGrid(param_grid))
    if trials:
        combos = random.Random(42).sample(combos, trials)

    for params in combos:
        train(params)
        evaluate(params)

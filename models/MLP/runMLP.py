"""Runs MLP training and testing."""

import json
import random
import numpy as np
import torch
from sklearn.metrics import cohen_kappa_score, f1_score
from sklearn.model_selection import ParameterGrid
from MLP import EMBED_DIM, MAX_REVIEWS, MLP, NUM_CLASSES


TRAIN_EMBED_PATH = "train_minilm_embeddings.pt"
VAL_EMBED_PATH = "validation_minilm_embeddings.pt"
TEST_EMBED_PATH = "test_minilm_embeddings.pt"

TRAIN_ITEMS_PATH = "trainItems.jsonl"
VAL_ITEMS_PATH = "validationItems.jsonl"
TEST_ITEMS_PATH = "testItems.jsonl"

MODEL_SAVE_PATH = "mlp_minilm.pth"


def to_class(rating):
    """Rating to class."""
    # Round rating stars
    stars = round(rating)

    # 0-4 classes
    label = stars - 1

    # Set range bounds
    if label < 0:
        label = 0
    if label > NUM_CLASSES - 1:
        label = NUM_CLASSES - 1

    return label


def load_labels(path):
    """Load item labels for classes."""
    labels = {}

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            item = json.loads(line)
            asin = item["parent_asin"]
            labels[asin] = to_class(item["rating"])

    return labels


def load_data(items_path, embed_path):
    """Load reviews per item."""
    labels = load_labels(items_path)
    embeds = torch.load(embed_path)

    # Zero-padded arrays
    x = np.zeros((len(labels), MAX_REVIEWS, EMBED_DIM), dtype=np.float32)
    y = np.zeros(len(labels), dtype=np.int64)
    row = 0

    # One row per item
    for asin, reviews in embeds.items():
        if asin not in labels or len(reviews) == 0:
            continue

        num = min(len(reviews), MAX_REVIEWS)
        x[row, :num] = reviews[:num].numpy()
        y[row] = labels[asin]
        row += 1

    # Drop unused rows
    return x[:row], y[:row]


def class_weights(y, power=0.5):
    """Adjusts class weighting."""
    # Count items per class
    counts = np.bincount(y, minlength=NUM_CLASSES)
    counts = counts.astype(np.float32)

    # w = (1 / n)^p
    weights = (1.0 / (counts + 1e-5)) ** power

    # Scale to sum C
    weights = weights / weights.sum() * NUM_CLASSES

    return weights.tolist()


def compute_metrics(y_true, y_pred):
    """Computes evaluation metrics."""
    classes = list(range(NUM_CLASSES))

    # error = |y - y_hat|
    error = np.abs(y_true - y_pred)

    # Quadratic weighted kappa
    qwk = cohen_kappa_score(y_true, y_pred, weights="quadratic", labels=classes)

    # Macro-averaged F1
    f1 = f1_score(y_true, y_pred, average="macro", labels=classes, zero_division=0)

    # MAE = mean(error)
    mae = float(error.mean())

    # Off-by-1 = mean(error <= 1)
    off_by_1 = float((error <= 1).mean())

    return {"qwk": qwk, "macro_f1": f1, "mae": mae, "off_by_1_accuracy": off_by_1}


def build_model(layer_count):
    """Builds the MLP."""
    sizes = (256, 128, 64)
    hidden = sizes[:layer_count]

    return MLP(hidden=hidden)


def train(lr, weight_d, layer_count, batch_size):
    """Trains the MLP."""
    x_train, y_train = load_data(TRAIN_ITEMS_PATH, TRAIN_EMBED_PATH)
    x_val, y_val = load_data(VAL_ITEMS_PATH, VAL_EMBED_PATH)

    model = build_model(layer_count)
    weights = class_weights(y_train)

    model.fit(
        x_train, y_train,
        x_val, y_val,
        lr, weight_d, batch_size,
        weights, MODEL_SAVE_PATH,
    )


def evaluate(lr, weight_d, layer_count, batch_size):
    """Print test results."""
    x, y = load_data(TEST_ITEMS_PATH, TEST_EMBED_PATH)

    model = build_model(layer_count)
    model.load(MODEL_SAVE_PATH)

    y_pred = model.predict(x)
    metrics = compute_metrics(y, y_pred)

    classes = list(range(NUM_CLASSES))
    f1_each = f1_score(y, y_pred, average=None, labels=classes, zero_division=0)
    print("F1 per class:", f1_each.round(3))

    print("\n" + "=" * 35)
    print("        FINAL TEST RESULTS        ")
    print("For values ", lr, " ", weight_d, " ", layer_count, " ", batch_size)
    print("=" * 35)
    print(f"Total Test Items:        {len(y_pred)}")
    print(f"Mean Absolute Error:     {metrics['mae']:.4f}")
    print(f"Off-by-1 Accuracy:       {metrics['off_by_1_accuracy'] * 100:.2f}%")
    print(f"Macro-Averaged F1:       {metrics['macro_f1']:.4f}")
    print(f"Quadratic Weighted Kappa:{metrics['qwk']:.4f}")
    print("=" * 35)


if __name__ == "__main__":
    # Fix random seed
    torch.manual_seed(42)
    np.random.seed(42)

    # Search space
    param_grid = {
        "lr": [0.0003, 0.001, 0.003],
        "weight_decay": [0.0, 0.0001],
        "num_layers": [1, 2, 3],
        "batch_size": [128, 256, 512],
    }

    # Single run
    param_grid = {
        "lr": [0.001],
        "weight_decay": [0.0001],
        "num_layers": [2],
        "batch_size": [256],
    }

    # Random search trials
    trials = None

    combos = list(ParameterGrid(param_grid))
    if trials:
        combos = random.Random(42).sample(combos, trials)

    for params in combos:
        lr = params["lr"]
        weight_d = params["weight_decay"]
        layer_count = params["num_layers"]
        batch_size = params["batch_size"]

        train(lr, weight_d, layer_count, batch_size)
        evaluate(lr, weight_d, layer_count, batch_size)

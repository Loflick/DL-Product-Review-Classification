"""MLP Model for Review Classification"""

import math

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import cohen_kappa_score

EMBED_DIM = 773
NUM_CLASSES = 5
CLASSES = list(range(NUM_CLASSES))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def compute_qwk(y_true, y_pred):
    """Quadratic weighted kappa."""
    return cohen_kappa_score(y_true, y_pred, weights="quadratic", labels=CLASSES)


class MLP(nn.Module):
    """MLP with self-attention."""

    def __init__(self, hidden=(256, 128)):
        super().__init__()

        # Hidden layers, per review
        layers = []
        in_size = EMBED_DIM

        for out_size in hidden:
            layers.append(nn.Linear(in_size, out_size))
            layers.append(nn.ReLU())
            in_size = out_size

        self.hidden = nn.Sequential(*layers)

        # Self-attention across reviews
        self.query = nn.Linear(in_size, in_size)
        self.key = nn.Linear(in_size, in_size)
        self.value = nn.Linear(in_size, in_size)
        self.norm = nn.LayerNorm(in_size)

        # Output layer
        self.output = nn.Linear(in_size, NUM_CLASSES)

        self.to(device)

    def forward(self, x):
        """Compute class scores."""
        x = x.float()
        x = self.hidden(x)

        # Attention = softmax(Q * K^T / sqrt(d)) * V
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        scores = q @ k.transpose(1, 2)
        scores = scores / math.sqrt(x.size(-1))

        weights = torch.softmax(scores, dim=-1)
        attn_out = weights @ v

        # Normalize
        x = x + attn_out
        x = self.norm(x)

        # Condense to 1D
        x = x.mean(dim=1)

        return self.output(x)

    def fit(self, x_train, y_train, x_val, y_val, lr, decay, batch_size, epochs, path):
        """Train with early stopping."""
        x_train = torch.from_numpy(x_train)
        y_train = torch.from_numpy(y_train)

        # Cross-entropy loss
        loss_fn = nn.CrossEntropyLoss()

        # Optimizer
        optimizer = torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=decay)

        best_qwk = -1.0
        bad_epochs = 0

        for epoch in range(epochs):
            self.train()

            # Shuffle each epoch
            order = torch.randperm(len(x_train))

            # Loop over batches
            for start in range(0, len(order), batch_size):
                end = start + batch_size
                rows = order[start:end]

                batch_x = x_train[rows].to(device)
                batch_y = y_train[rows].to(device)

                scores = self(batch_x)
                loss = loss_fn(scores, batch_y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # Check validation
            pred = np.rint(self.predict(x_val)).astype(int)
            qwk = compute_qwk(y_val, pred)
            print(f"Epoch {epoch + 1} | Val QWK: {qwk:.4f}")

            # Stopping
            if qwk > best_qwk:
                best_qwk = qwk
                bad_epochs = 0
                torch.save(self.state_dict(), path)
            else:
                bad_epochs += 1

            if bad_epochs >= 4:
                break

        # Restore best weights
        self.load(path)

    def predict(self, x):
        """Predict class value."""
        self.eval()
        all_values = []
        classes = torch.arange(NUM_CLASSES, device=device)

        with torch.no_grad():
            # Predict in chunks
            for start in range(0, len(x), 1024):
                end = start + 1024
                batch = torch.from_numpy(x[start:end])
                batch = batch.to(device)

                scores = self(batch)
                probs = torch.softmax(scores, dim=1)

                # value = sum(p * c)
                values = (probs * classes).sum(dim=1)
                all_values.append(values.cpu().numpy())

        return np.concatenate(all_values)

    def load(self, path):
        """Load weights."""
        weights = torch.load(path, map_location=device)
        self.load_state_dict(weights)

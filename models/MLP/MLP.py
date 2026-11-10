"""MLP Model for Review Classification"""

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import cohen_kappa_score

EMBED_DIM = 384
MAX_REVIEWS = 10
NUM_HEADS = 4
NUM_CLASSES = 5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
        self.attention = nn.MultiheadAttention(in_size, NUM_HEADS, batch_first=True)
        self.norm = nn.LayerNorm(in_size)

        # Output layer
        self.output = nn.Linear(in_size, NUM_CLASSES)

        self.to(device)

    def forward(self, x):
        """Compute class scores."""
        # Mask
        mask = x.abs().sum(dim=-1) != 0

        # Avoid empty items
        mask[:, 0] = True

        x = self.hidden(x)

        # Normalize
        attn_out, _ = self.attention(x, x, x, key_padding_mask=~mask)
        x = x + attn_out
        x = self.norm(x)

        # Condense to 1D
        # mean = sum(h) / R
        mask = mask.unsqueeze(-1).float()
        x = (x * mask).sum(dim=1)
        x = x / mask.sum(dim=1)

        return self.output(x)

    def fit(self, x_train, y_train, x_val, y_val, lr, decay, batch_size, weights, path):
        """Train with early stopping."""
        x_train = torch.from_numpy(x_train)
        y_train = torch.from_numpy(y_train)

        # Weighted cross-entropy loss
        weights = torch.tensor(weights, dtype=torch.float32, device=device)
        loss_fn = nn.CrossEntropyLoss(weight=weights)

        # Optimizer
        optimizer = torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=decay)

        classes = list(range(NUM_CLASSES))
        best_qwk = -1.0
        bad_epochs = 0

        for epoch in range(30):
            self.train()

            # Shuffle each epoch
            order = torch.randperm(len(x_train))

            # Loop over batches
            for start in range(0, len(order), batch_size):
                rows = order[start:start + batch_size]
                batch_x = x_train[rows].to(device)
                batch_y = y_train[rows].to(device)

                scores = self(batch_x)
                loss = loss_fn(scores, batch_y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # Check validation
            pred = self.predict(x_val)
            mae = np.abs(y_val - pred).mean()
            qwk = cohen_kappa_score(y_val, pred, weights="quadratic", labels=classes)
            print(f"Epoch {epoch + 1} | Val MAE: {mae:.4f} | Val QWK: {qwk:.4f}")

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
        """Predict classes."""
        self.eval()
        preds = []

        with torch.no_grad():
            # Predict in chunks
            for start in range(0, len(x), 1024):
                batch = torch.from_numpy(x[start:start + 1024])
                batch = batch.to(device)

                scores = self(batch)
                pred = scores.argmax(dim=1)
                preds.append(pred.cpu().numpy())

        return np.concatenate(preds)

    def load(self, path):
        """Load weights."""
        weights = torch.load(path, map_location=device)
        self.load_state_dict(weights)

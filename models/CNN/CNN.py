"""CNN Model for Review Classification """
#AI Disclamier: AI was used in the devolpment of this code
#Additionally it was used to further knowledge of a 1D CNN


import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass


@dataclass
class HParams:
    lr: float = 1e-3          # learning rate
    batch_size: int = 64      # reviews per weight update
    epochs: int = 10          # max passes over the training data (early stopping may stop sooner)
    activation: str = "relu"  # non-linearity used after the convs and in the projection layer
    n_filters: int = 100      # filters PER kernel size (3 sizes -> 3 * n_filters total)
    n_neurons: int = 128      # size of the projection layer
    dropout: float = 0.5


ACTIVATIONS = { #different activation functions; trainCNN can select the best one in it's trials 
    "relu": F.relu,
    "tanh": torch.tanh,
    "sigmoid": torch.sigmoid,
    "gelu": F.gelu,
}


class TextCNN(nn.Module):
    """
      GloVe vectors (B, L, 300)
        -> 3 parallel Conv1d branches, kernel sizes 3/4/5, n_filters each   (CNN block)
        -> self-attention across word positions        
        -> 1D global max-pooling over positions       
        -> projection layer (3*n_filters -> n_neurons)
        -> linear layer -> raw logits                   
    """

    def __init__(self, num_classes, hp, embed_dim=300, kernel_sizes=(3, 4, 5)):
        super().__init__()
        self.act = ACTIVATIONS[hp.activation] #set the activation function 
        self.dropout = hp.dropout #set the dropout 

        # padding="same" keeps the sequence length L, so the conv outputs can be
        # concatenated per word position and attention has a sequence to work on.
        self.convs = nn.ModuleList(
            [nn.Conv1d(embed_dim, hp.n_filters, k, padding="same") for k in kernel_sizes]
        )

        features = hp.n_filters * len(kernel_sizes)          # 300 with the defaults
        # 3 heads always divides features (features = 3 * n_filters), so any n_filters works.
        self.attn = nn.MultiheadAttention(features, num_heads=len(kernel_sizes), batch_first=True)
        #self attnetion allows the cnn model to view differeent relationships at once
        self.norm = nn.LayerNorm(features)
        #stablize and speed up neural network traning by normalizng the actigvaation 
        #of the neurons within a single data sample rather than across a batch

        self.proj = nn.Linear(features, hp.n_neurons)        # the 1 projection layer
        self.out = nn.Linear(hp.n_neurons, num_classes)  # raw logits

    def forward(self, x):                                # x: (B, L, 300) float
        # Padding rows were zero-filled in preprocessing, so all-zero rows = padding.
        pad = x.abs().sum(-1) == 0                       # (B, L), True where padding
        pad[pad.all(1), 0] = False                       # never mask a whole review

        h = x.transpose(1, 2)                            # (B, 300, L): Conv1d wants channels first
        h = torch.cat([self.act(conv(h)) for conv in self.convs], dim=1)  # (B, F, L)
        h = h.transpose(1, 2)                            # (B, L, F)

        a, _ = self.attn(h, h, h, key_padding_mask=pad, need_weights=False)
        h = self.norm(h + a)                             # residual connection + layer norm

        # Condense to 1D: strongest value per feature across real (non-pad) positions.
        h = h.masked_fill(pad.unsqueeze(-1), float("-inf")).max(dim=1).values  # (B, F)

        h = F.dropout(h, self.dropout, self.training)
        h = self.act(self.proj(h))                       # (B, n_neurons)
        return self.out(h)                               # (B, num_classes)

# Deep Learning & Product Review Classifcation
Deep Learning Approaches to Product Review Classification using MLP, CNN, LSTM, Transformers

Each model reads Amazon product reviews, embedded with MiniLM, and predicts the item's rating class (0-4, meaning 1-5 stars).

- Dataset: https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023
- Embeddings: https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2 (384 dimensions, one vector per review)

## Setup
You also need `sentence-transformers` (`pip install sentence-transformers`). Run everything from the **repo root**, because the scripts write their output files to the folder you run them from.

### 1. Download and split the data
```
python preprocessing/downloadData.py
```
This downloads items and reviews from Hugging Face. It empties its output files when it starts, so let it finish.

What it does:
1. Picks items from each of 5 categories (Electronics, Books, Clothing_Shoes_and_Jewelry, Beauty_and_Personal_Care, Home_and_Kitchen). Only items with at least 50 ratings are kept.
2. Splits the **items** 80% / 10% / 10% into train / validation / test with a fixed random seed. Every review belongs to the split of its item, so no item appears in more than one split.
3. Downloads reviews for those items, 10 text reviews per item (reviews under 3 words are skipped).

### 2. Embed the reviews
```
python preprocessing/updatedEmbed.py
```
This embeds each review with `all-MiniLM-L6-v2` and saves one file per split.

### Generated files
| Split | Items | Reviews | Embeddings |
|---|---|---|---|
| Train | `trainItems.jsonl` | `trainReviews.jsonl` | `train_minilm_embeddings.pt` |
| Validation | `validationItems.jsonl` | `validationReviews.jsonl` | `validation_minilm_embeddings.pt` |
| Test | `testItems.json` | `testReviews.jsonl` | `test_minilm_embeddings.pt` |

All of these are git-ignored. Each person generates them locally.

File formats:
- `*Items`: JSON lines (one object per line), with `title`, `category`, `rating`, `rating_number`, `parent_asin`. `rating` is the item's **average rating as a float from 1.0 to 5.0**, not a class. Convert it to a class 0-4 with `clamp(round(rating) - 1, 0, 4)`. (`testItems.json` has a `.json` extension but is the same JSON-lines format.)
- `*Reviews.jsonl`: JSON lines, with `title`, `text`, `category`, `parent_asin`. There is no per-review rating; look up the item's rating by `parent_asin`.
- `*_minilm_embeddings.pt`: a dict saved with `torch.save`, mapping `parent_asin` to a float32 tensor of shape `(n_reviews, 384)` with `n_reviews` up to 10. The vectors are L2-normalized. Items without usable reviews are not in the dict.

## Using the embedded data in a model
Models live in `models/<name>/`. Load a split's embeddings and its labels like this:
```python
import json
import torch

embeddings = torch.load("train_minilm_embeddings.pt")   # {parent_asin: (n_reviews, 384)}

labels = {}
with open("trainItems.jsonl", encoding="utf-8") as f:
    for line in f:
        item = json.loads(line)
        labels[item["parent_asin"]] = min(max(round(item["rating"]) - 1, 0), 4)
```

## Models
| Model | Folder |
|---|---|
| MLP | `models/MLP/` (`MLP.py` is the model, `runMLP.py` runs the pipeline: `python models/MLP/runMLP.py`) |
| Transformer | `models/transformer/` |
| CNN | `models/CNN/` (`CNN.py` is the model, `trainCNN.py` runs the pipeline and trains the model) |


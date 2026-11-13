# Deep Learning & Product Review Classifcation
Deep Learning Approaches to Product Review Classification using MLP, CNN, LSTM, Transformers

Each model reads Amazon product reviews, embedded with BGE plus a sentiment model, and predicts the item's rating class (0-4, meaning 1-5 stars).

- Dataset: https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023
- Embeddings: https://huggingface.co/BAAI/bge-base-en-v1.5 (768 dimensions) and https://huggingface.co/nlptown/bert-base-multilingual-uncased-sentiment (5 sentiment scores), 773 dimensions per review

## Setup
You also need `sentence-transformers` and `transformers` (`pip install sentence-transformers transformers`). Run everything from the **repo root**, because the scripts write their output files to the folder you run them from.

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
This embeds each review (title and text) with `bge-base-en-v1.5` and the sentiment model, and saves two files per split.

`downloadData.py` writes the test items as `testItems.json`, but `updatedEmbed.py` reads `testItems.jsonl`. Rename the file to `testItems.jsonl` before running step 2.

### Generated files
| Split | Items | Reviews | Embeddings | Metadata |
|---|---|---|---|---|
| Train | `trainItems.jsonl` | `trainReviews.jsonl` | `train_embeds.npy` | `train_meta.npz` |
| Validation | `validationItems.jsonl` | `validationReviews.jsonl` | `validation_embeds.npy` | `validation_meta.npz` |
| Test | `testItems.jsonl` | `testReviews.jsonl` | `test_embeds.npy` | `test_meta.npz` |

All of these are git-ignored. Each person generates them locally.

File formats:
- `*Items`: JSON lines (one object per line), with `title`, `category`, `rating`, `rating_number`, `parent_asin`. `rating` is the item's **average rating as a float from 1.0 to 5.0**, not a class. Convert it to a class 0-4 with `clamp(round(rating) - 1, 0, 4)`.
- `*Reviews.jsonl`: JSON lines, with `title`, `text`, `category`, `parent_asin`. There is no per-review rating; look up the item's rating by `parent_asin`.
- `*_embeds.npy`: a float16 array of shape `(n_items, 10, 773)`. Each item has up to 10 review vectors (768 BGE values followed by 5 sentiment scores), zero-padded if there are fewer.
- `*_meta.npz`: `asins` (the `parent_asin` of each row), `counts` (reviews per item), `targets` (the item's float average rating, 1.0 to 5.0), `embed_dim` and `max_reviews`. Row `i` of the `.npy` file belongs to `asins[i]`. Only items with at least one review are included.

## Using the embedded data in a model
Models live in `models/<name>/`. Load a split's embeddings and its labels like this:
```python
import numpy as np

embeds = np.load("train_embeds.npy")          # (n_items, 10, 773)
meta = np.load("train_meta.npz")
ratings = meta["targets"]                     # float rating per item
classes = np.clip(np.round(ratings) - 1, 0, 4).astype(int)
```

## Models
| Model | Folder |
|---|---|
| MLP | `models/MLP/` (`MLP.py` is the model, `runMLP.py` runs the pipeline and trains the model) |
| Transformer | `models/transformer/` |
| CNN | `models/CNN/` (`CNN.py` is the model, `trainCNN.py` runs the pipeline and trains the model) |


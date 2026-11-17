# Deep Learning & Product Review Classifcation
Deep Learning Approaches to Product Review Classification using MLP, CNN, LSTM, Transformers

- Dataset: https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023
- Embeddings: https://huggingface.co/BAAI/bge-base-en-v1.5 (768 dimensions) and https://huggingface.co/nlptown/bert-base-multilingual-uncased-sentiment (5 sentiment scores), 773 dimensions per review

## Setup
Run every command from the **repo root**.

### 1. Download and split the data
```
python preprocessing/downloadData.py
```
`downloadData.py` picks items from 5 categories (Electronics, Books, Clothing_Shoes_and_Jewelry, Beauty_and_Personal_Care, Home_and_Kitchen) with at least 50 ratings, splits the items 80/10/10 into train / validation / test, and downloads up to 10 reviews per item.

### 2. Embed the reviews
```
python preprocessing/updatedEmbed.py
```
`updatedEmbed.py` embeds each review (title and text) with BGE and the sentiment model.

### Generated files
| Split | Items | Reviews | Embeddings | Metadata |
|---|---|---|---|---|
| Train | `trainItems.jsonl` | `trainReviews.jsonl` | `train_embeds.npy` | `train_meta.npz` |
| Validation | `validationItems.jsonl` | `validationReviews.jsonl` | `validation_embeds.npy` | `validation_meta.npz` |
| Test | `testItems.json` | `testReviews.jsonl` | `test_embeds.npy` | `test_meta.npz` |

- `*Items`: one item per line, with `title`, `category`, `rating` (average rating, 1.0 to 5.0), `rating_number`, `parent_asin`.
- `*Reviews.jsonl`: one review per line, with `title`, `text`, `category`, `parent_asin`.
- `*_embeds.npy`: float16 array of shape `(n_items, 10, 773)`, zero-padded when an item has fewer than 10 reviews.
- `*_meta.npz`: `asins`, `counts` (reviews per item), `targets` (average rating), `embed_dim`, `max_reviews`. Row `i` of the `.npy` belongs to `asins[i]`.

## Models
Run each one from the repo root after steps 1 and 2.

| Model | Files | Run |
|---|---|---|
| MLP | `models/MLP/MLP.py` is the model, `models/MLP/runMLP.py` loads data, trains and tests | `python models/MLP/runMLP.py` |
| Transformer | `models/transformer/transformer.py` is the model, trainer and tester | `python models/transformer/transformer.py` |
| CNN | `models/CNN/CNN.py` is the model, `models/CNN/trainCNN.py` loads data, trains and tests | `python models/CNN/trainCNN.py` |
| LSTM | `models/LSTM/LSTM.py` is the model, `models/LSTM/runLSTM.py` loads data, trains and tests | `python models/LSTM/runLSTM.py` |


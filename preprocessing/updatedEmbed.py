import collections
import json
import os
from pathlib import Path

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]

SPLITS = [
    ("train", ROOT / "trainReviews.jsonl", ROOT / "trainItems.jsonl"),
    ("validation", ROOT / "validationReviews.jsonl", ROOT / "validationItems.jsonl"),
    ("test", ROOT / "testReviews.jsonl", ROOT / "testItems.json"),
]

BGE_MODEL_NAME = "BAAI/bge-base-en-v1.5"
SENT_MODEL_NAME = "nlptown/bert-base-multilingual-uncased-sentiment"

MAX_REVIEWS_PER_PRODUCT = 10
MAX_CHARS = 2000
MAX_SEQ_LEN = 256   
BATCH_SIZE = 256    
CHUNK_ITEMS = 5000  

ASIN_KEY = "parent_asin"
TEXT_KEY = "text"
TITLE_KEY = "title"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_labels(items_path):
    labels = {}
    with open(items_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            asin = data.get("parent_asin") or data.get("asin")
            rating = data.get("rating")
            if asin and rating is not None:
                labels[asin] = float(rating)
    return labels

def load_and_group_reviews(reviews_path, valid_asins, max_reviews=10):
    print(f"\nReading JSONL from: {reviews_path}")
    product_reviews = collections.defaultdict(list)

    with open(reviews_path, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Grouping reviews"):
            line = line.strip()
            if not line:
                continue

            data = json.loads(line)
            asin = data.get(ASIN_KEY) or data.get("asin")
            if not asin or asin not in valid_asins or len(product_reviews[asin]) >= max_reviews:
                continue

            title = (data.get(TITLE_KEY) or "").strip()
            body = (data.get(TEXT_KEY) or "").strip()
            text = f"{title}. {body}".strip(". ") if title and body else (body or title)

            if text:
                product_reviews[asin].append(text[:MAX_CHARS])

    print(f"Found {len(product_reviews)} unique products with reviews.")
    return product_reviews

def encode_and_save(split_name, product_reviews, labels, bge_model, sent_tok, sent_model):
    npy_path = ROOT / f"{split_name}_embeds.npy"
    meta_path = ROOT / f"{split_name}_meta.npz"

    asins = list(product_reviews.keys())
    num_items = len(asins)
    embed_dim = 768 + 5

    counts = np.array([len(product_reviews[a]) for a in asins], dtype=np.int32)
    targets = np.array([labels[a] for a in asins], dtype=np.float32)

    embeds_mm = np.lib.format.open_memmap(npy_path,mode="w+",dtype=np.float16,shape=(num_items, MAX_REVIEWS_PER_PRODUCT, embed_dim),)

    for start in tqdm(range(0, num_items, CHUNK_ITEMS), desc=f"Encoding {split_name} on {DEVICE}"):
        end = min(start + CHUNK_ITEMS, num_items)
        chunk_texts = []
        chunk_slices = []
        cursor = 0

        for idx in range(start, end):
            texts = product_reviews[asins[idx]]
            chunk_texts.extend(texts)
            chunk_slices.append((idx, cursor, cursor + len(texts)))
            cursor += len(texts)

        if not chunk_texts:
            continue

        order = np.argsort([len(t) for t in chunk_texts], kind="mergesort")
        sorted_texts = [chunk_texts[i] for i in order]

        with torch.inference_mode():
            bge_vecs_sorted = bge_model.encode(sorted_texts,batch_size=BATCH_SIZE,show_progress_bar=False,convert_to_numpy=True,normalize_embeddings=True,)

            sent_probs_list = []
            for s in range(0, len(sorted_texts), BATCH_SIZE):
                batch_t = sorted_texts[s : s + BATCH_SIZE]
                enc = sent_tok(batch_t,padding=True,truncation=True,max_length=MAX_SEQ_LEN,return_tensors="pt",).to(DEVICE)
                logits = sent_model(**enc).logits
                probs = torch.softmax(logits.float(), dim=-1)
                sent_probs_list.append(probs.cpu().numpy())

        sent_vecs_sorted = np.concatenate(sent_probs_list, axis=0)

        combined = np.empty((len(chunk_texts), embed_dim), dtype=np.float16)
        combined[order] = np.concatenate([bge_vecs_sorted, sent_vecs_sorted], axis=1).astype(np.float16)

        for item_idx, c_start, c_end in chunk_slices:
            n = c_end - c_start
            embeds_mm[item_idx, :n, :] = combined[c_start:c_end]
            if n < MAX_REVIEWS_PER_PRODUCT:
                embeds_mm[item_idx, n:, :] = 0.0

    embeds_mm.flush()
    np.savez(meta_path,asins=np.array(asins),counts=counts,targets=targets,embed_dim=embed_dim,max_reviews=MAX_REVIEWS_PER_PRODUCT,)

    size_mb = os.path.getsize(npy_path) / (1024**2)
    print(f"Saved {npy_path.name} ({size_mb:.2f} MB) | Items: {num_items} | Shape: ({num_items}, {MAX_REVIEWS_PER_PRODUCT}, {embed_dim})")


if __name__ == "__main__":
    print(f"Loading {BGE_MODEL_NAME} and {SENT_MODEL_NAME} on {DEVICE}...")
    bge_model = SentenceTransformer(BGE_MODEL_NAME, device=DEVICE)
    bge_model.max_seq_length = MAX_SEQ_LEN
    sent_tok = AutoTokenizer.from_pretrained(SENT_MODEL_NAME)
    sent_model = AutoModelForSequenceClassification.from_pretrained(SENT_MODEL_NAME).to(DEVICE).eval()

    if DEVICE == "cuda":
        bge_model.half()
        sent_model.half()

    for split_name, rev_path, items_path in SPLITS:
        print(f"\n--- PROCESSING {split_name.upper()} ---")
        labels = load_labels(items_path)
        reviews_by_product = load_and_group_reviews(rev_path, labels, max_reviews=MAX_REVIEWS_PER_PRODUCT)
        encode_and_save(split_name, reviews_by_product, labels, bge_model, sent_tok, sent_model)
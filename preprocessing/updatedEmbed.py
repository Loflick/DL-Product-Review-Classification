import collections
import json
import os
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
from pathlib import Path
from sentence_transformers import SentenceTransformer
import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]

TRAIN_INPUT = ROOT / "trainReviews.jsonl"
VALIDATION_INPUT = ROOT / "validationReviews.jsonl"
TEST_INPUT = ROOT / "testReviews.jsonl"

TRAIN_EMBED = ROOT / "train_bge_embeddings.pt"
VALIDATION_EMBED = ROOT / "validation_bge_embeddings.pt"
TEST_EMBED = ROOT / "test_bge_embeddings.pt"

MODEL_NAME = "BAAI/bge-base-en-v1.5"
MAX_REVIEWS_PER_PRODUCT = 10 
BATCH_SIZE = 16  
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

ASIN_KEY = "parent_asin"  
TEXT_KEY = "text" 
TITLE_KEY = "title" 


def load_and_group_reviews(jsonl_path, max_reviews=10, asin_key="parent_asin", text_key="text"):
    print(f"\nReading JSONL from: {jsonl_path}")
    product_reviews = collections.defaultdict(list)

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Reading reviews"):
            line = line.strip()
            if not line:
                continue

            data = json.loads(line)
            asin = data.get(asin_key)
            if not asin or len(product_reviews[asin]) >= max_reviews:
                continue

            text = data.get(text_key, "")
            if not text and TITLE_KEY in data:
                text = data.get(TITLE_KEY, "")

            text = text.strip()
            if text:
                product_reviews[asin].append(text[:600])

    print(f"Found {len(product_reviews)} unique products.")
    return product_reviews


def encode_and_save(product_reviews, output_path):
    print(f"Loading {MODEL_NAME}...")
    model = SentenceTransformer(MODEL_NAME, device=DEVICE)

    all_texts = []
    product_index_map = {}
    current_idx = 0

    for asin, texts in product_reviews.items():
        all_texts.extend(texts)
        product_index_map[asin] = (current_idx, current_idx + len(texts))
        current_idx += len(texts)

    print(f"Encoding {len(all_texts)} individual reviews on {DEVICE}...")
    
    embeddings = model.encode(all_texts, batch_size=BATCH_SIZE, show_progress_bar=True, convert_to_tensor=False, normalize_embeddings=True)

    embeddings = torch.from_numpy(embeddings).to(torch.float32)

    final_dict = {}
    for asin, (start, end) in product_index_map.items():
        final_dict[asin] = embeddings[start:end]

    torch.save(final_dict, output_path)
    print(f"Saved: {output_path} | Size: {os.path.getsize(output_path) / (1024**2):.2f} MB")


if __name__ == "__main__":
    for split_name, in_path, out_path in [
        ("TRAIN", TRAIN_INPUT, TRAIN_EMBED),
        ("VALIDATION", VALIDATION_INPUT, VALIDATION_EMBED),
        ("TEST", TEST_INPUT, TEST_EMBED)
    ]:
        print(f"\n--- DOING {split_name} ---")
        reviews_by_product = load_and_group_reviews(in_path, max_reviews=MAX_REVIEWS_PER_PRODUCT, asin_key=ASIN_KEY)
        encode_and_save(reviews_by_product, out_path)

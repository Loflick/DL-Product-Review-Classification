import pandas as pd
import numpy as np
import json
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import h5py
from collections import defaultdict
from tqdm import tqdm
import re
import requests
import hashlib
from sklearn.model_selection import train_test_split
import gc
from pathlib import Path
import os
import collections

CATEGORIES = {
    "All_Beauty": {"target_reviews": 200_000, "target_items": 25_000},
    "Amazon_Fashion": {"target_reviews": 200_000, "target_items": 25_000},
    "Appliances": {"target_reviews": 200_000, "target_items": 25_000},
    "Arts_Crafts_and_Sewing": {"target_reviews": 200_000, "target_items": 25_000},
    "Automotive": {"target_reviews": 200_000, "target_items": 25_000},
    "Baby_Products": {"target_reviews": 200_000, "target_items": 25_000},
    "Beauty_and_Personal_Care": {"target_reviews": 200_000, "target_items": 25_000},
    "Books": {"target_reviews": 200_000, "target_items": 25_000},
    "CDs_and_Vinyl": {"target_reviews": 200_000, "target_items": 25_000},
    "Cell_Phones_and_Accessories": {"target_reviews": 200_000, "target_items": 25_000},
    "Clothing_Shoes_and_Jewelry": {"target_reviews": 200_000, "target_items": 25_000},
    "Digital_Music": {"target_reviews": 200_000, "target_items": 25_000},
    "Electronics": {"target_reviews": 200_000, "target_items": 25_000},
    "Gift_Cards": {"target_reviews": 200_000, "target_items": 25_000},
    "Grocery_and_Gourmet_Food": {"target_reviews": 200_000, "target_items": 25_000},
    "Handmade_Products": {"target_reviews": 200_000, "target_items": 25_000},
    "Health_and_Household": {"target_reviews": 200_000, "target_items": 25_000},
    "Health_and_Personal_Care": {"target_reviews": 200_000, "target_items": 25_000},
    "Home_and_Kitchen": {"target_reviews": 200_000, "target_items": 25_000},
    "Industrial_and_Scientific": {"target_reviews": 200_000, "target_items": 25_000},
    "Kindle_Store": {"target_reviews": 200_000, "target_items": 25_000},
    "Magazine_Subscriptions": {"target_reviews": 200_000, "target_items": 25_000},
    "Movies_and_TV": {"target_reviews": 200_000, "target_items": 25_000},
    "Musical_Instruments": {"target_reviews": 200_000, "target_items": 25_000},
    "Office_Products": {"target_reviews": 200_000, "target_items": 25_000},
    "Patio_Lawn_and_Garden": {"target_reviews": 200_000, "target_items": 25_000},
    "Pet_Supplies": {"target_reviews": 200_000, "target_items": 25_000},
    "Software": {"target_reviews": 200_000, "target_items": 25_000},
    "Sports_and_Outdoors": {"target_reviews": 200_000, "target_items": 25_000},
    "Subscription_Boxes": {"target_reviews": 200_000, "target_items": 25_000},
    "Tools_and_Home_Improvement": {"target_reviews": 200_000, "target_items": 25_000},
    "Toys_and_Games": {"target_reviews": 200_000, "target_items": 25_000},
    "Video_Games": {"target_reviews": 200_000, "target_items": 25_000},
    "Unknown": {"target_reviews": 200_000, "target_items": 25_000},
}

ITEMS_OUTPUT = "asin_to_item.jsonl"
REVIEWS_OUTPUT = "amazon_reviews_1M_sample.jsonl"
MAX_REVIEWS_PER_ITEM = 30

TRAIN_ITEMS_OUTPUT = "trainItems.jsonl"
VALIDATION_ITEMS_OUTPUT = "validationItems.jsonl"
TEST_ITEMS_OUTPUT = "testItems.json"

TRAIN_REVIEW_OUTPUT = "trainReviews.jsonl"
VALIDATION_REVIEW_OUTPUT = "validationReviews.jsonl"
TEST_REVIEW_OUTPUT = "testReviews.jsonl"

TRAIN_EMBED_OUTPUT = "trainEmbed.jsonl"
VALIDATION_EMBED_OUTPUT = "validationEmbed.jsonl"
TEST_EMBED_OUTPUT = "testEmbed.jsonl"

class extractDataTest:
    @staticmethod
    def extractValidItemsForCategory(category, min_ratings=5, max_per_bucket=5000, max_lines_scan=300_000):
        url = f"https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/resolve/main/raw/meta_categories/meta_{category}.jsonl"
        response = requests.get(url, stream=True)
        response.raise_for_status()

        bucket_counts = collections.Counter()
        pbar = tqdm(total=max_per_bucket * 5, unit="items", desc=f"Meta: {category}")

        with open(ITEMS_OUTPUT, "a", encoding="utf-8") as out_r:
            for line_idx, line in enumerate(response.iter_lines(chunk_size=1024 * 1024), start=1):
                if not line:
                    continue

                item = json.loads(line.decode("utf-8"))
                asin = item.get("parent_asin")
                avg_rating = item.get("average_rating")
                rating_number = item.get("rating_number")

                if not asin or avg_rating is None or rating_number is None or rating_number < min_ratings:
                    continue

                bucket = min(5, max(1, round(float(avg_rating))))
                if bucket_counts[bucket] >= max_per_bucket:
                    continue

                item_to_save = {
                    "title": item.get("title", ""),
                    "category": category,
                    "rating": float(avg_rating),
                    "rating_number": int(rating_number),
                    "parent_asin": asin,
                }
                out_r.write(json.dumps(item_to_save) + "\n")
                bucket_counts[bucket] += 1
                pbar.update(1)

                if all(bucket_counts[b] >= max_per_bucket for b in range(1, 6)) or line_idx >= max_lines_scan:
                    break

        pbar.close()
        print(f"{category} item buckets: {dict(sorted(bucket_counts.items()))}")

    @staticmethod
    def extractReviewsFromValidItems(asinLookup, category, reviews_per_item=10, min_reviews_per_item=1, max_lines_scan=2_000_000):
        url = f"https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/resolve/main/raw/review_categories/{category}.jsonl"
        response = requests.get(url, stream=True)
        response.raise_for_status()

        item_review_buffer = collections.defaultdict(list)
        completed_asins = set()
        written_reviews = 0
        pbar = tqdm(total=max_lines_scan, unit="lines", desc=f"Reviews: {category}")

        with open(TRAIN_REVIEW_OUTPUT, "a", encoding="utf-8") as out_T, \
             open(VALIDATION_REVIEW_OUTPUT, "a", encoding="utf-8") as out_V, \
             open(TEST_REVIEW_OUTPUT, "a", encoding="utf-8") as out_Te:

            writers = {"train": out_T, "val": out_V, "test": out_Te}

            for line_idx, line in enumerate(response.iter_lines(chunk_size=1024 * 1024), start=1):
                pbar.update(1)
                if line_idx > max_lines_scan:
                    break
                if not line:
                    continue

                review = json.loads(line.decode("utf-8"))
                asin = review.get("parent_asin")
                if not asin or asin in completed_asins or asin not in asinLookup:
                    continue

                text = (review.get("text") or "").strip()
                if len(text.split()) < 3:
                    continue

                item_review_buffer[asin].append({
                    "title": review.get("title", ""),
                    "text": text,
                    "category": category,
                    "parent_asin": asin,
                })

                if len(item_review_buffer[asin]) == reviews_per_item:
                    target = writers[asinLookup[asin]]
                    for r in item_review_buffer.pop(asin):
                        target.write(json.dumps(r) + "\n")
                    completed_asins.add(asin)
                    written_reviews += reviews_per_item

            partial_items = 0
            for asin, revs in item_review_buffer.items():
                if len(revs) >= min_reviews_per_item and asin in asinLookup:
                    target = writers[asinLookup[asin]]
                    for r in revs:
                        target.write(json.dumps(r) + "\n")
                    written_reviews += len(revs)
                    partial_items += 1

        pbar.close()
        print(
            f"{category}: {len(completed_asins)} full ({reviews_per_item}-rev) + "
            f"{partial_items} partial items | {written_reviews} reviews written."
        )


def main():
    with open(ITEMS_OUTPUT, "w", encoding="utf-8") as out_f:
        pass
    with open(TRAIN_ITEMS_OUTPUT, "w", encoding="utf-8") as out_f:
        pass
    with open(VALIDATION_ITEMS_OUTPUT, "w", encoding="utf-8") as out_f:
        pass
    with open(TEST_ITEMS_OUTPUT, "w", encoding="utf-8") as out_f:
        pass
    with open(TRAIN_REVIEW_OUTPUT, "w", encoding="utf-8") as out_f:
        pass
    with open(VALIDATION_REVIEW_OUTPUT, "w", encoding="utf-8") as out_f:
        pass
    with open(TEST_REVIEW_OUTPUT, "w", encoding="utf-8") as out_f:
        pass

    downloadNewFiles = True

    if downloadNewFiles:
        for category, config in CATEGORIES.items():
            print(f"Starting pipeline for {category}")
            extractDataTest.extractValidItemsForCategory(category,min_ratings=5,max_per_bucket=5000,max_lines_scan=300_000,)

        with open(ITEMS_OUTPUT, "r") as out_f:
            items = [json.loads(line) for line in out_f if line.strip()]

        df = pd.DataFrame(items)

        buckets = df["rating"].round().clip(1, 5).astype(int)
        train_df, temp_df = train_test_split(df, test_size = 0.20, stratify=buckets, random_state=25269876)
        val_df, test_df = train_test_split(temp_df, test_size = 0.50, stratify=temp_df["rating"].round().clip(1, 5).astype(int), random_state=25269876)

        train_df.to_json(TRAIN_ITEMS_OUTPUT, orient="records", lines=True)
        val_df.to_json(VALIDATION_ITEMS_OUTPUT, orient="records", lines=True)
        test_df.to_json(TEST_ITEMS_OUTPUT, orient="records", lines=True)

        lookup = defaultdict(str)
        lookup = {asin: "train" for asin in train_df["parent_asin"]}
        lookup.update({asin: "val" for asin in val_df["parent_asin"]})
        lookup.update({asin: "test" for asin in test_df["parent_asin"]})

        del train_df
        del val_df
        del test_df

        gc.collect()

        for category, config in CATEGORIES.items():
            print("Downloading Reviews")
            extractDataTest.extractReviewsFromValidItems(lookup,category,reviews_per_item=10,min_reviews_per_item=1,max_lines_scan=1_000_000,)


    print("All finished")

if __name__ == "__main__":
    main()
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
    "Electronics": {"target_reviews": 200_000, "target_items": 25_000},
    "Books": {"target_reviews": 200_000, "target_items": 25_000},
    "Clothing_Shoes_and_Jewelry": {"target_reviews": 200_000, "target_items": 25_000,},
    "Beauty_and_Personal_Care": {"target_reviews": 200_000,"target_items": 25_000,},
    "Home_and_Kitchen": {"target_reviews": 200_000, "target_items": 25_000},
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

class extractData:
    def extractValidItemsForCategory(category):
        target_items = CATEGORIES[category]["target_items"]
        url = f"https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/resolve/main/raw/meta_categories/meta_{category}.jsonl"

        response = requests.get(url, stream=True)
        response.raise_for_status()

        itemCount = 0

        pbar = tqdm(total=target_items, unit="reviews")

        with open(ITEMS_OUTPUT, "a", encoding="utf-8") as out_r:
            for line in response.iter_lines(chunk_size=1024 * 1024):
                if not line: continue
                item = json.loads(line.decode("utf-8"))
                asin = item.get("parent_asin")

                pbar.update(1)

                avg_rating = item.get("average_rating")
                if avg_rating is None:
                    continue

                item_to_save = {
                    "title": item.get("title"),
                    "category": category,
                    "rating": avg_rating,
                    "parent_asin": asin,
                }
                out_r.write(json.dumps(item_to_save) + "\n")
                itemCount += 1


                if itemCount > CATEGORIES[category]["target_items"]:
                    break
        pbar.close()

    def extractReviewsFromValidItems(asinLookup, category):
        target_reviews = CATEGORIES[category]["target_reviews"]
        url = f"https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/resolve/main/raw/review_categories/{category}.jsonl"

        response = requests.get(url, stream=True)
        response.raise_for_status()

        pbar = tqdm(total=target_reviews, unit="reviews")

        count = 0

        with open(TRAIN_REVIEW_OUTPUT, "a", encoding="utf-8") as out_T, \
             open(VALIDATION_REVIEW_OUTPUT, "a", encoding="utf-8") as out_V, \
             open(TEST_REVIEW_OUTPUT, "a", encoding="utf-8") as out_Te:
                writers = {
                    "train": out_T,
                    "val": out_V,
                    "test": out_Te
                }
                for line in response.iter_lines(chunk_size=1024 * 1024):
                    if not line: continue

                    review = json.loads(line.decode("utf-8"))
                    asin = review.get("parent_asin")

                    split = asinLookup.get(asin)
                    target = writers.get(split)

                    if target:
                        review_to_save = {
                            "title": review.get("title"),
                            "text": review.get("text"),
                            "category": category,
                            "parent_asin": asin,
                        }
                        target.write(json.dumps(review_to_save) + "\n")
                        pbar.update(1)
                        count += 1

                    if count > target_reviews:
                        break

        pbar.close()

class extractDataTest:
    @staticmethod
    def extractValidItemsForCategory(category, min_ratings=40, oversample=1.5):
        target_items = int(CATEGORIES[category]["target_items"] * oversample)
        url = f"https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/resolve/main/raw/meta_categories/meta_{category}.jsonl"

        response = requests.get(url, stream=True)
        response.raise_for_status()

        itemCount = 0
        pbar = tqdm(total=target_items, unit="items", desc=f"Candidates: {category}")

        with open(ITEMS_OUTPUT, "a", encoding="utf-8") as out_r:
            for line in response.iter_lines(chunk_size=1024 * 1024):
                if not line:
                    continue

                item = json.loads(line.decode("utf-8"))
                asin = item.get("parent_asin")
                avg_rating = item.get("average_rating")
                rating_number = item.get("rating_number", 0)

                if (
                    not asin
                    or avg_rating is None
                    or rating_number is None
                    or rating_number < min_ratings
                ):
                    continue

                item_to_save = {
                    "title": item.get("title", ""),
                    "category": category,
                    "rating": float(avg_rating),
                    "rating_number": int(rating_number),
                    "parent_asin": asin,
                }
                out_r.write(json.dumps(item_to_save) + "\n")
                itemCount += 1
                pbar.update(1)

                if itemCount >= target_items:
                    break

        pbar.close()

    @staticmethod
    def extractReviewsFromValidItems(asinLookup, category, reviews_per_item=10):
        target_reviews = CATEGORIES[category]["target_reviews"]
        url = f"https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/resolve/main/raw/review_categories/{category}.jsonl"

        response = requests.get(url, stream=True)
        response.raise_for_status()

        pbar = tqdm(total=target_reviews, unit="reviews", desc=f"Reviews: {category}")

        item_review_buffer = collections.defaultdict(list)
        completed_asins = (set()) 
        count = 0

        with open(TRAIN_REVIEW_OUTPUT, "a", encoding="utf-8") as out_T,\
            open(VALIDATION_REVIEW_OUTPUT, "a", encoding="utf-8") as out_V,\
            open(TEST_REVIEW_OUTPUT, "a", encoding="utf-8") as out_Te :

            writers = {"train": out_T, "val": out_V, "test": out_Te}

            for line in response.iter_lines(chunk_size=1024 * 1024):
                if not line:
                    continue

                review = json.loads(line.decode("utf-8"))
                asin = review.get("parent_asin")

                if not asin or asin in completed_asins:
                    continue

                split = asinLookup.get(asin)
                target = writers.get(split)
                if not target:
                    continue

                text = review.get("text", "")
                if not text or len(text.strip().split()) < 3:
                    continue

                review_to_save = {
                    "title": review.get("title", ""),
                    "text": text.strip(),
                    "category": category,
                    "parent_asin": asin,
                }
                item_review_buffer[asin].append(review_to_save)

                if len(item_review_buffer[asin]) == reviews_per_item:
                    for r in item_review_buffer[asin]:
                        target.write(json.dumps(r) + "\n")

                    completed_asins.add(asin)
                    del item_review_buffer[asin]
                    count += reviews_per_item
                    pbar.update(reviews_per_item)

                    if count >= target_reviews:
                        break

        pbar.close()
        print(
            f"Successfully collected {len(completed_asins)} products with"
            f" exactly {reviews_per_item} text reviews each."
        )
        return completed_asins


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
            extractDataTest.extractValidItemsForCategory(category, 50, 1.5)

        with open(ITEMS_OUTPUT, "r") as out_f:
            items = [json.loads(line) for line in out_f if line.strip()]

        df = pd.DataFrame(items)


        train_df, temp_df = train_test_split(df, test_size = 0.20, random_state=25269876)
        val_df, test_df = train_test_split(temp_df, test_size = 0.50, random_state=25269876)

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
            extractDataTest.extractReviewsFromValidItems(lookup, category, 10)


    print("All finished")

if __name__ == "__main__":
    main()
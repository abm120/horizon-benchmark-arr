import pandas as pd
import os
import json
import multiprocessing
from tqdm import tqdm
import logging
from huggingface_hub import hf_hub_download

# Output Directory for Parquet files
output_dir = "amazon_parquet_data"
os.makedirs(output_dir, exist_ok=True)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

REPO_ID = "McAuley-Lab/Amazon-Reviews-2023"


def _download(filename):
    """Download a single file from the dataset repo directly via huggingface_hub.
    This avoids the datasets library entirely (no loading script, no offline-cache
    fallback traps, and hf_hub_download supports resumable downloads out of the box).
    """
    return hf_hub_download(repo_id=REPO_ID, repo_type="dataset", filename=filename)


# Function to load dataset (without shared memory)
def load_reviews_dataset(category):
    # If parquet file already exists, skip loading
    if os.path.exists(os.path.join(output_dir, f"{category}_index.parquet")):
        print(f"⚠️ {category} index already exists, skipping...")
        return

    print(f"Loading {category} dataset...")

    # Downloads raw/review_categories/{category}.jsonl directly - this is the
    # exact file the old "raw_review_{category}" config pointed the loading
    # script at. We fetch it ourselves and parse the jsonl line by line.
    review_path = _download(f"raw/review_categories/{category}.jsonl")

    user_index = {}
    with open(review_path, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc=f"Setting up {category} dataset user index", unit="item"):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue

            user_id = item["user_id"]
            timestamp = item["timestamp"]

            if user_id not in user_index:
                user_index[user_id] = {}
            user_index[user_id][timestamp] = f"{item['title']} \n{item['text']}"

    # Save as Parquet
    save_path = os.path.join(output_dir, f"{category}_index.parquet")
    df = pd.DataFrame(
        [(user, ts, review) for user, timestamps in user_index.items() for ts, review in timestamps.items()],
        columns=["user_id", "timestamp", "review"],
    )
    df.to_parquet(save_path, engine="pyarrow", compression="snappy")

    print(f"✅ Saved {category} index to {save_path}")


def _load_history_split(category, split):
    """Downloads and reads one split (train/valid/test) of the
    0core_timestamp_w_his benchmark files. Columns are:
    user_id,parent_asin,rating,timestamp,history
    """
    path = _download(f"benchmark/0core/timestamp_w_his/{category}.{split}.csv")
    return pd.read_csv(path)


# Function to process each category
def process_category(category):
    print(f"Processing {category}...")

    # If processed Parquet file already exists, skip processing
    if os.path.exists(os.path.join(output_dir, f"{category}.parquet")):
        print(f"⚠️ {category} already processed, skipping...")
        return

    # Load previously saved dataset from Parquet
    review_index_path = os.path.join(output_dir, f"{category}_index.parquet")
    if not os.path.exists(review_index_path):
        print(f"⚠️ Missing {review_index_path}, skipping category {category}")
        return

    review_index = pd.read_parquet(review_index_path).set_index(["user_id", "timestamp"])

    # Load the benchmark splits directly from the Hub (no datasets library needed)
    dataset_timestamp_his_train = _load_history_split(category, "train")
    dataset_timestamp_his_val = _load_history_split(category, "valid")
    dataset_timestamp_his_test = _load_history_split(category, "test")

    # Merge validation and test datasets
    dataset_timestamp_his = pd.concat([dataset_timestamp_his_val, dataset_timestamp_his_test], ignore_index=True)
    dataset_timestamp_his = pd.concat([dataset_timestamp_his_train, dataset_timestamp_his], ignore_index=True)

    user_data = {}
    for _, item in tqdm(dataset_timestamp_his.iterrows(), desc=f"Processing {category} dataset", unit="item"):
        user_id = item["user_id"]
        timestamp = item["timestamp"]
        user_id_str = str(user_id)
        timestamp = int(timestamp)

        if user_id_str not in user_data:
            user_data[user_id_str] = {
                "user_id": user_id_str,
                "ratings": [],
                "timestamps": [],
                "history": [],
                "reviews": []
            }

        user_data[user_id_str]["ratings"].append(item["rating"])
        user_data[user_id_str]["timestamps"].append(item["timestamp"])
        user_data[user_id_str]["history"].append(item["parent_asin"])

        # Lookup review text from pre-saved index
        review_text = "NA"
        try:
            review_text = review_index.loc[(user_id_str, timestamp), "review"]
        except KeyError:
            print(f"⚠️ Review not found for user {user_id_str} at timestamp {timestamp}")

        user_data[user_id_str]["reviews"].append(review_text)

    # Convert to DataFrame
    df = pd.DataFrame.from_dict(user_data, orient="index")

    # Save as Parquet
    parquet_filename = os.path.join(output_dir, f"{category}.parquet")
    df.to_parquet(parquet_filename, engine="pyarrow", compression="snappy")

    print(f"✅ Saved {category} to {parquet_filename}")
    logging.info(f"Saved {category} to {parquet_filename}")
    logging.info(f"Number of users in {category}: {len(df)}")


if __name__ == "__main__":
    # Read category names from file
    category_names = []
    with open("category_names.txt", "r") as file:
        for line in file:
            category_names.append(line.strip())
    print(category_names)

    # Parallel execution for dataset loading
    with multiprocessing.Pool(processes=10) as pool:
        list(tqdm(pool.imap(load_reviews_dataset, category_names), total=len(category_names)))

    print("✅ All category indexes saved!")

    # Process each category in parallel
    with multiprocessing.Pool(processes=10) as pool:
        list(tqdm(pool.imap(process_category, category_names), total=len(category_names)))

    print("✅ All categories processed and saved as Parquet!")

    # Clean up
    for category in category_names:
        review_index_path = os.path.join(output_dir, f"{category}_index.parquet")
        if os.path.exists(review_index_path):
            os.remove(review_index_path)
            print(f"🗑️ Removed {review_index_path}")

    print("✅ Cleanup complete!")

    # Final message
    print("All datasets have been processed and saved successfully!")
    logging.info("All datasets have been processed and saved successfully!")
    print("Check the 'amazon_parquet_data' directory for the output files.")
    logging.info("Check the 'amazon_parquet_data' directory for the output files.")
# End of script
# Note: Make sure to have huggingface_hub installed (pip/mamba install huggingface_hub).
# This script downloads files directly from the Hub, bypassing the datasets library's
# loading-script mechanism entirely - it doesn't need `datasets` to be importable at all.
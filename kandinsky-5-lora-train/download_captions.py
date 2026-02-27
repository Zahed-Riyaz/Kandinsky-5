import os
import sys
import io
import argparse
import requests
import textwrap
from datasets import load_dataset
from tqdm import tqdm
from PIL import Image


# --- DOWNLOAD LOGIC ---
def download_pairs(dataset_name, output_dir, limit, image_col, caption_col, split):
    os.makedirs(output_dir, exist_ok=True)

    print(f"Streaming dataset '{dataset_name}'...")
    try:
        ds = load_dataset(dataset_name, split=split, streaming=True)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    count = 0
    pbar = tqdm(total=limit, desc="Downloading pairs")
    headers = {"User-Agent": "Mozilla/5.0"}

    for item in ds:
        if count >= limit: break
        
        url = item.get(image_col) or item.get(image_col.upper()) or item.get(image_col.lower())
        caption = item.get(caption_col) or item.get(caption_col.upper()) or item.get(caption_col.lower())

        if not url or not caption: continue

        try:
            resp = requests.get(url, timeout=5, headers=headers)
            resp.raise_for_status()
            image = Image.open(io.BytesIO(resp.content)).convert("RGB")
            
            file_id = f"{count:06d}"
            image.save(os.path.join(output_dir, f"{file_id}.jpg"), "JPEG", quality=95)
            with open(os.path.join(output_dir, f"{file_id}.txt"), "w") as f:
                f.write(str(caption).strip())

            count += 1
            pbar.update(1)
        except:
            continue

    pbar.close()
    print(f"Done. Saved {count} pairs.")
    sys.stdout.flush()
    os._exit(0)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="laion/relaion2B-en-research-safe")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--image_col", type=str, default="url")
    parser.add_argument("--caption_col", type=str, default="caption")
    parser.add_argument("--split", type=str, default="train")
    args = parser.parse_args()
    download_pairs(args.dataset, args.output_dir, args.limit, args.image_col, args.caption_col, args.split)
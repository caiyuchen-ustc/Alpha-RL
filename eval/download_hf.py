"""
Download multiple Hugging Face models by index range.

Example:
    python download_hf_models.py \
        --start 0 \
        --end 27 \
        --base_url caiyuchen/DAPO-step \
        --save_dir ./dapomodels

"""

import os
import argparse
from huggingface_hub import snapshot_download

def main():
    parser = argparse.ArgumentParser(description="Batch download Hugging Face models by index.")
    parser.add_argument("--start", type=int, required=True, help="Start index (inclusive)")
    parser.add_argument("--end", type=int, required=True, help="End index (inclusive)")
    parser.add_argument("--base_url", type=str, required=True, help="Base repo name (e.g. caiyuchen/DAPO-step)")
    parser.add_argument("--save_dir", type=str, required=True, help="Local directory to save models")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    for i in range(args.start, args.end + 1):
        repo_id = f"{args.base_url}-{i}"
        local_dir = os.path.join(args.save_dir, f"DAPO-step-{i}")
        print(f"🔽 Downloading {repo_id} ...")

        try:
            snapshot_download(
                repo_id=repo_id,
                local_dir=local_dir,
                resume_download=True
            )
            print(f"✅ Downloaded and saved to: {local_dir}\n")
        except Exception as e:
            print(f"❌ Failed to download {repo_id}: {e}\n")

    print("All downloads completed")

if __name__ == "__main__":
    main()
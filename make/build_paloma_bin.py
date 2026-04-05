import os

import numpy as np
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer


def main() -> None:
    try:
        ds = load_dataset("allenai/paloma", "c4_100_domains", split="validation")
    except ValueError:
        ds = load_dataset("allenai/paloma", "c4_100_domains", split="val")

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    all_ids: list[int] = []
    for item in tqdm(ds, desc="Tokenizing Paloma"):
        all_ids.extend(tokenizer.encode(item["text"]) + [tokenizer.eos_token_id])

    out_path = "data/paloma/tokenized_paloma_c4_100_domains_validation.bin"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.array(all_ids, dtype=np.uint16).tofile(out_path)
    print(f"Saved {len(all_ids)} tokens to {out_path}")


if __name__ == "__main__":
    main()

import argparse
import numpy as np
import os
from transformers import AutoTokenizer
from tqdm import tqdm

LEADERBOARD_DIR = "data/leaderboard"
CLASSIFIER_DIR = os.path.join(LEADERBOARD_DIR, "classifier")

tokenizer = AutoTokenizer.from_pretrained("gpt2")


def main(leaderboard_dir: str = LEADERBOARD_DIR, classifier_dir: str = CLASSIFIER_DIR):
    data_path = os.path.join(leaderboard_dir, "tokenized_paloma_c4_100_domains_validation.bin")
    out_path = os.path.join(classifier_dir, "paloma_c4_100_domains_validation_text.txt")

    tokens = np.memmap(data_path, dtype=np.uint16, mode="r")  # shape: (N,)
    eos = tokenizer.eos_token_id
    boundaries = np.where(tokens == eos)[0]
    docs = np.split(tokens, boundaries + 1)

    longest_doc = max(len(doc) for doc in docs)
    print(f"Longest doc: {longest_doc}")

    shortest_doc = min(len(doc) for doc in docs)
    print(f"Shortest doc: {shortest_doc}")

    mean_doc_length = sum(len(doc) for doc in docs) / len(docs)
    print(f"Mean doc length: {mean_doc_length}")

    os.makedirs(classifier_dir, exist_ok=True)
    with open(out_path, "w") as f:
        for doc in tqdm(docs, desc="Writing docs"):
            if len(doc) > 0 and doc[-1] == eos:
                doc = doc[:-1]
            if len(doc) == 0:
                continue
            f.write(tokenizer.decode(doc))
            f.write("\n\n---END_OF_DOC---\n\n")

    print(f"Total docs: {len(docs)}")
    print(f"Total tokens: {len(tokens)}")
    print("Special tokens:", tokenizer.special_tokens_map)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--leaderboard-dir", type=str, default=LEADERBOARD_DIR)
    parser.add_argument("--classifier-dir", type=str, default=CLASSIFIER_DIR)
    args = parser.parse_args()
    main(leaderboard_dir=args.leaderboard_dir, classifier_dir=args.classifier_dir)

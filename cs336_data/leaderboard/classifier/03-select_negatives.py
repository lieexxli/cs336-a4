import argparse
import random
import os
from tqdm import tqdm

DATA_DIR = "data/03-deduped"
CLASSIFIER_DIR = "data/leaderboard/classifier"

NUM_TRAIN_EXAMPLES = 28000
NUM_VALID_EXAMPLES = 500


def main(
    num_train_examples: int = NUM_TRAIN_EXAMPLES,
    num_valid_examples: int = NUM_VALID_EXAMPLES,
    data_dir: str = DATA_DIR,
    classifier_dir: str = CLASSIFIER_DIR,
):
    out_path = os.path.join(classifier_dir, "negatives_train.txt")
    out_path_valid = os.path.join(classifier_dir, "negatives_valid.txt")

    data_filepaths = [
        os.path.join(data_dir, filepath)
        for filepath in os.listdir(data_dir)
        if filepath.endswith(".warc.wet.gz")
    ]

    if not data_filepaths:
        raise ValueError(f"No deduped WET files found in {data_dir}")

    examples_per_file = 0
    while examples_per_file * (examples_per_file * 10) < num_train_examples + num_valid_examples:
        examples_per_file += 1
    n_files = examples_per_file * 10

    random.seed(42)
    if n_files <= len(data_filepaths):
        chosen_paths = random.sample(data_filepaths, n_files)
    else:
        # Local experiments may only have a few files; allow reusing them.
        chosen_paths = random.choices(data_filepaths, k=n_files)
    examples = []

    for data_filepath in tqdm(chosen_paths, desc="Processing files"):
        with open(data_filepath) as f:
            docs = f.read().split("\n\n---END_OF_DOC---\n\n")
            docs = [doc for doc in docs if doc.strip()]
            if not docs:
                continue

            if len(docs) >= examples_per_file:
                chosen_docs = random.sample(docs, examples_per_file)
            else:
                chosen_docs = random.choices(docs, k=examples_per_file)
            examples.extend(chosen_docs)

    total_examples = num_train_examples + num_valid_examples
    if not examples:
        raise ValueError(f"No negative examples could be sampled from {data_dir}")

    if len(examples) < total_examples:
        examples.extend(random.choices(examples, k=total_examples - len(examples)))

    valid_examples = examples[:num_valid_examples]
    train_examples = examples[num_valid_examples:]

    if len(train_examples) > num_train_examples:
        train_examples = train_examples[:num_train_examples]

    os.makedirs(classifier_dir, exist_ok=True)

    with open(out_path, "w") as f:
        for example in train_examples:
            joined_text = example.replace("\n", " ")
            f.write(f"__label__negative {joined_text}\n")

    print(f"Wrote {len(train_examples)} negative examples to {out_path}")

    with open(out_path_valid, "w") as f:
        for example in valid_examples:
            joined_text = example.replace("\n", " ")
            f.write(f"__label__negative {joined_text}\n")

    print(f"Wrote {len(valid_examples)} negative examples to {out_path_valid}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-train-examples", type=int, default=NUM_TRAIN_EXAMPLES)
    parser.add_argument("--num-valid-examples", type=int, default=NUM_VALID_EXAMPLES)
    parser.add_argument("--data-dir", type=str, default=DATA_DIR)
    parser.add_argument("--classifier-dir", type=str, default=CLASSIFIER_DIR)
    args = parser.parse_args()
    main(
        num_train_examples=args.num_train_examples,
        num_valid_examples=args.num_valid_examples,
        data_dir=args.data_dir,
        classifier_dir=args.classifier_dir,
    )

import argparse
import os
import random

CLASSIFIER_DIR = "data/leaderboard/classifier"


def main(classifier_dir: str = CLASSIFIER_DIR):
    pos_train_path = os.path.join(classifier_dir, "positives_train.txt")
    pos_valid_path = os.path.join(classifier_dir, "positives_valid.txt")
    neg_train_path = os.path.join(classifier_dir, "negatives_train.txt")
    neg_valid_path = os.path.join(classifier_dir, "negatives_valid.txt")
    train_out_path = os.path.join(classifier_dir, "quality.train")
    valid_out_path = os.path.join(classifier_dir, "quality.valid")

    random.seed(42)

    with open(pos_train_path) as f:
        positive_lines = f.readlines()

    with open(neg_train_path) as f:
        negative_lines = f.readlines()

    print(f"Read {len(positive_lines)} positive examples")
    print(f"Read {len(negative_lines)} negative examples")

    print(f"Words per positive example: {sum(len(line.split()) for line in positive_lines) / len(positive_lines)}")
    print(f"Words per negative example: {sum(len(line.split()) for line in negative_lines) / len(negative_lines)}")

    lines = positive_lines + negative_lines
    random.shuffle(lines)

    with open(train_out_path, "w") as f:
        f.writelines(lines)

    print(f"Wrote {len(lines)} training examples to {train_out_path}")

    with open(pos_valid_path) as f:
        positive_valid_lines = f.readlines()

    with open(neg_valid_path) as f:
        negative_valid_lines = f.readlines()

    lines = positive_valid_lines + negative_valid_lines
    random.shuffle(lines)

    with open(valid_out_path, "w") as f:
        f.writelines(lines)

    print(f"Wrote {len(lines)} validation examples to {valid_out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--classifier-dir", type=str, default=CLASSIFIER_DIR)
    args = parser.parse_args()
    main(classifier_dir=args.classifier_dir)

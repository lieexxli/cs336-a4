import json
import os
import sys
from pathlib import Path


def count_lines(path: str) -> int:
    with open(path) as f:
        return sum(1 for _ in f)


def check_quality_a(urls_path: str, model_path: str, train_path: str, valid_path: str, expected_urls: int) -> int:
    required = [urls_path, model_path, train_path, valid_path]
    if not all(os.path.exists(path) for path in required):
        return 1

    if count_lines(urls_path) != expected_urls:
        return 1

    if count_lines(train_path) <= 0 or count_lines(valid_path) <= 0:
        return 1

    return 0


def check_step3(meta_path: str, expected_files: int) -> int:
    if not os.path.exists(meta_path):
        return 1

    with open(meta_path) as f:
        meta = json.load(f)

    return 0 if meta.get("total_files") == expected_files else 1


def check_quality_b(classifier_dir: str, train_examples: int, valid_examples: int) -> int:
    train_path = os.path.join(classifier_dir, "quality.train")
    valid_path = os.path.join(classifier_dir, "quality.valid")
    model_path = os.path.join(classifier_dir, "quality.bin")

    required = [train_path, valid_path, model_path]
    if not all(os.path.exists(path) for path in required):
        return 1

    if count_lines(train_path) != 2 * train_examples:
        return 1

    if count_lines(valid_path) != 2 * valid_examples:
        return 1

    return 0


def check_step5(classified_dir: str, tokens_path: str, expected_files: int) -> int:
    stats_path = os.path.join(classified_dir, "stats.json")
    if not (os.path.exists(classified_dir) and os.path.exists(tokens_path) and os.path.exists(stats_path)):
        return 1

    file_count = len([path for path in Path(classified_dir).iterdir() if path.name.endswith(".warc.wet.gz")])
    return 0 if file_count == expected_files else 1


def main() -> int:
    if len(sys.argv) < 2:
        raise SystemExit("usage: check_reuse.py <quality-a|step3|quality-b|step5> ...")

    mode = sys.argv[1]
    if mode == "quality-a":
        return check_quality_a(sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], int(sys.argv[6]))
    if mode == "step3":
        return check_step3(sys.argv[2], int(sys.argv[3]))
    if mode == "quality-b":
        return check_quality_b(sys.argv[2], int(sys.argv[3]), int(sys.argv[4]))
    if mode == "step5":
        return check_step5(sys.argv[2], sys.argv[3], int(sys.argv[4]))

    raise SystemExit(f"unknown mode: {mode}")


if __name__ == "__main__":
    raise SystemExit(main())

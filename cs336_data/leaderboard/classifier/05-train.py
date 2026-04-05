import argparse
import fasttext

CLASSIFIER_DIR = "data/leaderboard/classifier"


def main(classifier_dir: str = CLASSIFIER_DIR):
    train_path = f"{classifier_dir}/quality.train"
    valid_path = f"{classifier_dir}/quality.valid"
    model_path = f"{classifier_dir}/quality.bin"

    model = fasttext.train_supervised(
        input=train_path,
        epoch=5,
        lr=0.2,
    )
    model.save_model(model_path)
    print(model.test(valid_path, k=1))
    print(f"Model saved to {model_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--classifier-dir", type=str, default=CLASSIFIER_DIR)
    args = parser.parse_args()
    main(classifier_dir=args.classifier_dir)

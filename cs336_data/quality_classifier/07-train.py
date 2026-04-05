import argparse
import os
import fasttext

DEFAULT_TRAIN_PATH = "data/wiki/quality.train"
DEFAULT_VALID_PATH = "data/wiki/quality.valid"
DEFAULT_MODEL_PATH = "out/models/quality.bin"


def main(train_path: str = DEFAULT_TRAIN_PATH, valid_path: str = DEFAULT_VALID_PATH, model_path: str = DEFAULT_MODEL_PATH):
    model = fasttext.train_supervised(
        input=train_path,
        epoch=30,
        lr=0.2,
    )
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    model.save_model(model_path)
    print(model.test(valid_path, k=1))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-path", default=DEFAULT_TRAIN_PATH)
    parser.add_argument("--valid-path", default=DEFAULT_VALID_PATH)
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    args = parser.parse_args()
    main(train_path=args.train_path, valid_path=args.valid_path, model_path=args.model_path)

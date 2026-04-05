import os
import fasttext


def main():
    model = fasttext.train_supervised(
        input="data/wiki/quality.train",
        epoch=30,
        lr=0.2,
    )
    os.makedirs("out/models", exist_ok=True)
    model.save_model("out/models/quality.bin")
    print(model.test("data/wiki/quality.valid", k=1))


if __name__ == "__main__":
    main()

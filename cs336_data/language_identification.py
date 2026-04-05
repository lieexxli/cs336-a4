import os
import fasttext

def _find_model(filename: str) -> str:
    for path in ["/data/classifiers", "data/classifiers", "../classifiers"]:
        full = os.path.join(path, filename)
        if os.path.exists(full):
            return full
    raise FileNotFoundError(f"{filename} not found. Put it in data/classifiers/, /data/classifiers/, or ../classifiers/")

MODEL_PATH = _find_model("lid.176.bin")

model = fasttext.load_model(MODEL_PATH)


def identify_language(text: str) -> str:
    clean = text.replace("\n", " ")
    labels, probs = model.predict(clean, k=1)

    lang = labels[0].replace("__label__", "")
    return lang, probs[0]

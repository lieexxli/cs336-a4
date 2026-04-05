import os
import fasttext

_CANDIDATE_PATHS = [
    "/data/c-sniderb/a4-leaderboard/classifier/quality.bin",  # 集群路径
    "data/leaderboard/classifier/quality.bin",                 # 个人服务器推荐路径
]

def _find_model() -> str:
    for path in _CANDIDATE_PATHS:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(
        "Leaderboard quality classifier not found. "
        "Train it with leaderboard/classifier/05-train.py and place the output at "
        "data/leaderboard/classifier/quality.bin"
    )

quality_model = fasttext.load_model(_find_model())


def classify_c4_100(text: str) -> tuple[str, float]:
    """
    Classify a document as positive (resembling c4_100) or negative (resembling cc).
    Returns a tuple of (label, confidence)
    """
    clean = text.replace("\n", " ")
    labels, probs = quality_model.predict(clean, k=1)

    label = labels[0].replace("__label__", "")
    return label, probs[0]

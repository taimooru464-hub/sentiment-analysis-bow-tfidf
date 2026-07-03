"""
utils.py
--------
Small shared helper functions used across the project: loading the raw
NLTK Movie Reviews corpus into a pandas DataFrame, and a couple of
path/IO helpers so the other modules don't repeat this logic.
"""

import os
import random
import pandas as pd
import nltk
from nltk.corpus import movie_reviews

RANDOM_STATE = 42
random.seed(RANDOM_STATE)


def load_movie_reviews_df():
    """
    Loads the NLTK Movie Reviews corpus (2000 labeled reviews: 1000 positive,
    1000 negative) and returns it as a pandas DataFrame with two columns:
    'review' (raw text) and 'sentiment' ('pos' / 'neg').
    """
    # Make sure the corpus is available locally (no-op if already downloaded)
    try:
        movie_reviews.fileids()
    except LookupError:
        nltk.download("movie_reviews")

    records = []
    for fileid in movie_reviews.fileids():
        sentiment = movie_reviews.categories(fileid)[0]  # 'pos' or 'neg'
        text = movie_reviews.raw(fileid)
        records.append({"review": text, "sentiment": sentiment})

    df = pd.DataFrame(records)
    # Shuffle so pos/neg reviews are mixed rather than sitting in two blocks
    df = df.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
    return df


def ensure_dir(path):
    """Create a directory if it doesn't already exist."""
    os.makedirs(path, exist_ok=True)
    return path


def project_path(*parts):
    """Build a path relative to the project root (one level above /src)."""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(root, *parts)

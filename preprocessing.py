"""
preprocessing.py
-----------------
Full NLP preprocessing pipeline for raw movie review text:
lowercasing -> punctuation/special-char removal -> tokenization ->
stop-word removal -> lemmatization.

Each step is exposed as its own function so it can be demonstrated
step-by-step (see Task 3 in the notebook), and also chained together
via `preprocess_review()` / `preprocess_corpus()` for actual model use.
"""

import re
import string
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

STOP_WORDS = set(stopwords.words("english"))
LEMMATIZER = WordNetLemmatizer()


def to_lowercase(text: str) -> str:
    """Step 1: Lowercase everything so 'Great' and 'great' are treated the same."""
    return text.lower()


def remove_punctuation(text: str) -> str:
    """Step 2: Strip standard punctuation characters (.,!?'\"; etc.)."""
    return text.translate(str.maketrans("", "", string.punctuation))


def remove_special_characters(text: str) -> str:
    """Step 3: Remove anything that isn't a letter or whitespace (digits, symbols,
    stray HTML-ish artifacts sometimes present in review dumps)."""
    return re.sub(r"[^a-zA-Z\s]", "", text)


def tokenize(text: str):
    """Step 4: Split the cleaned text into a list of word tokens."""
    return word_tokenize(text)


def remove_stopwords(tokens):
    """Step 5: Drop common English stop-words ('the', 'is', 'and', ...) that
    carry little sentiment-discriminating information on their own."""
    return [tok for tok in tokens if tok not in STOP_WORDS and len(tok) > 1]


def lemmatize(tokens):
    """Step 6: Reduce words to their dictionary base form
    (e.g. 'movies' -> 'movie', 'running' -> 'running'->'run' with correct POS,
    here using the default noun-based lemmatizer which is standard for BoW/TF-IDF)."""
    return [LEMMATIZER.lemmatize(tok) for tok in tokens]


def preprocess_review(text: str) -> str:
    """
    Runs the complete pipeline on a single review and returns a single
    cleaned string (tokens joined back with spaces) ready to be fed into
    CountVectorizer / TfidfVectorizer.
    """
    text = to_lowercase(text)
    text = remove_punctuation(text)
    text = remove_special_characters(text)
    tokens = tokenize(text)
    tokens = remove_stopwords(tokens)
    tokens = lemmatize(tokens)
    return " ".join(tokens)


def preprocess_corpus(reviews):
    """Applies preprocess_review() to an iterable of raw review strings."""
    return [preprocess_review(r) for r in reviews]


def show_pipeline_steps(text: str) -> dict:
    """
    Utility used in the notebook (Task 3) to display the output of every
    single stage for one example review, so the effect of each step is visible.
    """
    steps = {}
    steps["0_original"] = text
    steps["1_lowercase"] = to_lowercase(text)
    steps["2_no_punctuation"] = remove_punctuation(steps["1_lowercase"])
    steps["3_no_special_chars"] = remove_special_characters(steps["2_no_punctuation"])
    tokens = tokenize(steps["3_no_special_chars"])
    steps["4_tokens"] = tokens
    tokens_no_stop = remove_stopwords(tokens)
    steps["5_no_stopwords"] = tokens_no_stop
    tokens_lemmatized = lemmatize(tokens_no_stop)
    steps["6_lemmatized"] = tokens_lemmatized
    steps["7_final_string"] = " ".join(tokens_lemmatized)
    return steps

"""
feature_extraction.py
----------------------
Wraps scikit-learn's CountVectorizer (Bag of Words) and TfidfVectorizer
(TF-IDF) so both feature-extraction approaches are built the same way
and can be compared fairly (same max_features, same min_df, etc.).
"""

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# Keeping vocabulary size capped so both models train quickly and so the
# comparison isn't skewed by one method having a much larger vocabulary.
MAX_FEATURES = 5000
MIN_DF = 2  # ignore terms that appear in fewer than 2 documents (likely noise/typos)


def build_bow_features(train_texts, test_texts):
    """
    Fits a CountVectorizer (Bag of Words) on the training texts and
    transforms both train and test sets. Returns (X_train, X_test, vectorizer).
    """
    vectorizer = CountVectorizer(max_features=MAX_FEATURES, min_df=MIN_DF)
    X_train = vectorizer.fit_transform(train_texts)
    X_test = vectorizer.transform(test_texts)
    return X_train, X_test, vectorizer


def build_tfidf_features(train_texts, test_texts):
    """
    Fits a TfidfVectorizer on the training texts and transforms both
    train and test sets. Returns (X_train, X_test, vectorizer).
    """
    vectorizer = TfidfVectorizer(max_features=MAX_FEATURES, min_df=MIN_DF)
    X_train = vectorizer.fit_transform(train_texts)
    X_test = vectorizer.transform(test_texts)
    return X_train, X_test, vectorizer


def vocabulary_summary(vectorizer, n_sample=15):
    """Returns (vocab_size, sample_of_terms) for quick inspection/printing."""
    vocab = vectorizer.get_feature_names_out()
    sample = list(vocab[:n_sample])
    return len(vocab), sample

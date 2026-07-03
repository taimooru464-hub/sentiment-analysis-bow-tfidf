"""
train_tfidf.py
----------------
Standalone script: same pipeline as train_bow.py, but using TF-IDF
features instead of raw Bag-of-Words counts. Kept as a separate script
(rather than a flag on train_bow.py) to match the assignment's required
project structure.

Run directly with:  python src/train_tfidf.py
"""

import os
import sys
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils import load_movie_reviews_df, project_path, ensure_dir
from preprocessing import preprocess_corpus
from feature_extraction import build_tfidf_features
from evaluation import evaluate_model

RANDOM_STATE = 42


def main():
    print("Loading data...")
    df = load_movie_reviews_df()

    print("Preprocessing reviews...")
    df["clean_review"] = preprocess_corpus(df["review"])

    X_train_text, X_test_text, y_train, y_test = train_test_split(
        df["clean_review"], df["sentiment"],
        test_size=0.2, random_state=RANDOM_STATE, stratify=df["sentiment"]
    )

    print("Building TF-IDF features...")
    X_train, X_test, vectorizer = build_tfidf_features(X_train_text, X_test_text)

    print("Training Logistic Regression on TF-IDF features...")
    model = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    metrics = evaluate_model(y_test, y_pred)
    print("TF-IDF model results:", metrics)

    results_dir = ensure_dir(project_path("results"))
    joblib.dump(model, os.path.join(results_dir, "tfidf_logreg_model.pkl"))
    joblib.dump(vectorizer, os.path.join(results_dir, "tfidf_vectorizer.pkl"))
    print(f"Saved model + vectorizer to {results_dir}")

    return metrics


if __name__ == "__main__":
    main()

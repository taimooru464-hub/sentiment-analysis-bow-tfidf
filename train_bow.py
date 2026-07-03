"""
train_bow.py
-------------
Standalone script: loads the preprocessed movie review data, builds
Bag-of-Words features, trains a Logistic Regression classifier on top
of them, evaluates it, and saves the trained model + vectorizer to
/results so they can be reloaded later (e.g. for custom predictions).

Run directly with:  python src/train_bow.py
"""

import os
import sys
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils import load_movie_reviews_df, project_path, ensure_dir
from preprocessing import preprocess_corpus
from feature_extraction import build_bow_features
from evaluation import evaluate_model

RANDOM_STATE = 42


def main():
    print("Loading data...")
    df = load_movie_reviews_df()

    print("Preprocessing reviews (this takes a bit for 2000 reviews)...")
    df["clean_review"] = preprocess_corpus(df["review"])

    X_train_text, X_test_text, y_train, y_test = train_test_split(
        df["clean_review"], df["sentiment"],
        test_size=0.2, random_state=RANDOM_STATE, stratify=df["sentiment"]
    )

    print("Building Bag-of-Words features...")
    X_train, X_test, vectorizer = build_bow_features(X_train_text, X_test_text)

    print("Training Logistic Regression on BoW features...")
    model = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    metrics = evaluate_model(y_test, y_pred)
    print("BoW model results:", metrics)

    results_dir = ensure_dir(project_path("results"))
    joblib.dump(model, os.path.join(results_dir, "bow_logreg_model.pkl"))
    joblib.dump(vectorizer, os.path.join(results_dir, "bow_vectorizer.pkl"))
    print(f"Saved model + vectorizer to {results_dir}")

    return metrics


if __name__ == "__main__":
    main()

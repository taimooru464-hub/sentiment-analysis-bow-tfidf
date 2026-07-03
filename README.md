# Comparative Sentiment Analysis using Bag of Words and TF-IDF

**Course:** Programming for AI (Assignment 4)
**Author:** Qamar Reyhan — Department of Artificial Intelligence, Riphah International University

## Project Overview

This project builds a complete sentiment analysis system that classifies movie reviews as
**positive** or **negative**, and compares two classic text feature-extraction techniques —
**Bag of Words (BoW)** and **TF-IDF** — feeding a Logistic Regression classifier. The goal is not
just to build one working model, but to isolate and measure how much the feature-extraction
method itself affects performance, using an identical preprocessing pipeline, train/test split,
and classifier for both.

## Dataset

**NLTK Movie Reviews Corpus** (`nltk.corpus.movie_reviews`) — 2,000 labeled movie reviews,
perfectly balanced with 1,000 positive and 1,000 negative reviews. The corpus is downloaded
automatically via `nltk.download("movie_reviews")` the first time the code runs, so no manual
dataset download is required.

## Project Structure

```
sentiment_analysis_project/
│
├── data/                     # (dataset is fetched live via NLTK; no static files needed)
├── src/
│   ├── preprocessing.py      # Full NLP cleaning pipeline
│   ├── feature_extraction.py # BoW + TF-IDF vectorizer builders
│   ├── train_bow.py          # Standalone script: trains + saves the BoW model
│   ├── train_tfidf.py        # Standalone script: trains + saves the TF-IDF model
│   ├── evaluation.py         # Shared metrics (accuracy, precision, recall, F1, confusion matrix)
│   └── utils.py               # Data loading + path helpers
│
├── notebooks/
│   └── sentiment_analysis.ipynb   # Full end-to-end walkthrough (Tasks 1–10)
│
├── results/                  # Saved models, vectorizers, plots, custom prediction outputs
├── README.md
├── requirements.txt
└── .gitignore
```

## Installation Instructions

```bash
# 1. Clone the repository
git clone <THIS_REPOSITORY_URL>
cd sentiment_analysis_project

# 2. (Recommended) create a virtual environment
python -m venv venv
source venv/bin/activate      # on Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download the required NLTK data (one-time)
python -c "import nltk; nltk.download('movie_reviews'); nltk.download('punkt'); nltk.download('punkt_tab'); nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('omw-1.4')"
```

## How to Run

**Option A — Run the full notebook (recommended, includes all analysis/plots/report):**
```bash
jupyter notebook notebooks/sentiment_analysis.ipynb
```

**Option B — Run the standalone training scripts directly:**
```bash
python src/train_bow.py
python src/train_tfidf.py
```
Each script prints its evaluation metrics to the console and saves the trained model +
vectorizer to `results/`.

## Methodology

1. **Preprocessing** — lowercasing → punctuation removal → special-character removal →
   tokenization → stop-word removal → lemmatization (`src/preprocessing.py`).
2. **Feature Extraction** — Bag of Words (`CountVectorizer`) and TF-IDF (`TfidfVectorizer`),
   both capped at `max_features=5000`, `min_df=2` so vocabulary size is matched and the
   comparison isolates the weighting scheme rather than vocabulary size.
3. **Modeling** — Logistic Regression (`max_iter=1000`) trained separately on the BoW matrix and
   the TF-IDF matrix, using the same 80/20 stratified train/test split for both.
4. **Evaluation** — Accuracy, Precision, Recall, F1-score, and confusion matrices for both
   models.
5. **Feature Importance** — Top 10 positive/negative words identified from the TF-IDF model's
   Logistic Regression coefficients.
6. **Custom Predictions** — Five hand-written reviews run through the TF-IDF model to sanity
   check generalization beyond the training corpus.

## Results Summary

| Metric    | BoW    | TF-IDF |
|-----------|--------|--------|
| Accuracy  | 0.8325 | 0.8675 |
| Precision | 0.8519 | 0.8808 |
| Recall    | 0.8050 | 0.8500 |
| F1 Score  | 0.8278 | 0.8651 |

**TF-IDF outperformed Bag of Words on every metric** in this experiment. Full discussion,
confusion matrices, feature-importance charts, and the 300–500 word comparative analysis report
are in `notebooks/sentiment_analysis.ipynb` (Tasks 6, 7, and 9).

## Screenshots

See `results/class_distribution.png`, `results/confusion_matrices.png`, and
`results/feature_importance.png` for the generated visualizations, also embedded directly in the
notebook.

## GitHub Repository

`<THIS_REPOSITORY_URL>`

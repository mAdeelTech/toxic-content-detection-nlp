"""Fit and save Adeel's documented count vectorizer configurations once."""

import os
from tempfile import NamedTemporaryFile

import joblib
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from assignment2_vectorize import COUNT_FILES, ROOT

TRAIN = ROOT / "data" / "processed" / "toxicchat_train_cleaned.csv"
OUTPUT = ROOT / "artifacts" / "assignment2"


def main():
    # Match Part A's exact read_csv and astype(str) behavior.
    documents = pd.read_csv(TRAIN)["cleaned_text"].astype(str)
    vectorizers = {
        "bow": CountVectorizer(),
        "tfidf": TfidfVectorizer(),
        "bigram": CountVectorizer(ngram_range=(1, 2)),
        "trigram": CountVectorizer(ngram_range=(1, 3)),
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for name, vectorizer in vectorizers.items():
        path = OUTPUT / COUNT_FILES[name]
        if path.exists():
            print(f"Kept existing {path.relative_to(ROOT)}")
            continue
        vectorizer.fit(documents)
        with NamedTemporaryFile(dir=OUTPUT, delete=False) as temp:
            staged = temp.name
        try:
            joblib.dump(vectorizer, staged, compress=3)
            reloaded = joblib.load(staged)
            if (reloaded.transform(documents.iloc[:3]) != vectorizer.transform(documents.iloc[:3])).nnz:
                raise RuntimeError(f"Round-trip verification failed for {name}")
            try:
                os.link(staged, path)
                print(f"Saved and verified {path.relative_to(ROOT)}")
            except FileExistsError:
                print(f"Kept existing {path.relative_to(ROOT)}")
        finally:
            os.unlink(staged)


if __name__ == "__main__":
    main()

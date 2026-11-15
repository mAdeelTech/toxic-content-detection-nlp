"""Run Parts C and D using the checked-in corpus and saved team artifacts."""

import argparse
import json
import sys

import numpy as np
import pandas as pd

from assignment2_save_vectorizers import TRAIN
from assignment2_vectorize import (
    COUNT_FILES,
    EMBEDDING_FILES,
    load_representation,
    vectorize_document,
)

QUERIES = ("hate", "abuse", "stupid", "help", "write")
PAIRS = {
    "expected_related": (("hate", "dislike"), ("abuse", "offensive")),
    "expected_unrelated": (("hate", "recipe"), ("stupid", "python")),
}
ANALOGY = ("woman", "man", "king", "queen")


def cosine(first, second):
    denominator = np.linalg.norm(first) * np.linalg.norm(second)
    return float(np.dot(first, second) / denominator) if denominator else None


def probe_words(vectors):
    vocab = vectors.key_to_index
    available_queries = [word for word in QUERIES if word in vocab]
    a, b, c, expected = ANALOGY
    missing = [word for word in (a, b, c) if word not in vocab]
    analogy = {"operation": f"{a} - {b} + {c}", "expected": expected, "missing_inputs": missing}
    if not missing:
        candidate = vectors[a] - vectors[b] + vectors[c]
        ranked = vectors.similar_by_vector(candidate, topn=8)
        results = [[word, float(score)] for word, score in ranked if word not in {a, b, c}][:5]
        analogy.update(results=results, top1_correct=bool(results and results[0][0] == expected),
                       expected_in_vocab=expected in vocab)

    pairs = {}
    for group, word_pairs in PAIRS.items():
        pairs[group] = [
            {"words": [left, right],
             "cosine": cosine(vectors[left], vectors[right]) if left in vocab and right in vocab else None,
             "missing": [word for word in (left, right) if word not in vocab]}
            for left, right in word_pairs
        ]
    return {
        "dimension": vectors.vector_size,
        "trained_vocabulary_size": len(vocab),
        "queries": {
            word: [[neighbor, float(score)] for neighbor, score in vectors.most_similar(word, topn=5)]
            for word in available_queries
        },
        "missing_queries": [word for word in QUERIES if word not in vocab],
        "cosine": pairs,
        "analogy": analogy,
    }


def evaluate(count_only=False):
    train = pd.read_csv(TRAIN)["cleaned_text"].astype(str)
    test = pd.read_csv(TRAIN.with_name("toxicchat_test_cleaned.csv"))
    examples = test.loc[test["cleaned_text"].notna()].head(3)
    representations = COUNT_FILES if count_only else COUNT_FILES | EMBEDDING_FILES
    oov_token = "unseenwordxyz"
    report = {"training_rows": len(train), "actual_test_rows": examples.index.tolist(),
              "representations": {}, "missing_artifacts": {}}
    for name in representations:
        try:
            artifact = load_representation(name)
        except FileNotFoundError as error:
            report["missing_artifacts"][name] = str(error)
            continue
        if name in COUNT_FILES:
            matrix = artifact.transform(train)
            result = {"dimension": matrix.shape[1], "training_rows": matrix.shape[0],
                      "zero_percent": 100 * (1 - matrix.nnz / np.prod(matrix.shape))}
        else:
            result = probe_words(artifact)
            nonzeros = sum(np.count_nonzero(vectorize_document(doc, name, already_cleaned=True)) for doc in train)
            result["zero_percent"] = 100 * (1 - nonzeros / (len(train) * artifact.vector_size))
        checks = []
        for row, example in examples.iterrows():
            stored = vectorize_document(example.cleaned_text, name, already_cleaned=True)
            raw = vectorize_document(example.original_text, name)
            expected_shape = (1, result["dimension"])
            assert raw.shape == stored.shape == expected_shape
            if name in COUNT_FILES:
                matching = (raw != stored).nnz == 0
                nonzeros = int(raw.nnz)
            else:
                matching = bool(np.allclose(raw, stored))
                nonzeros = int(np.count_nonzero(raw))
            checks.append({"row": row, "shape": list(raw.shape), "nonzero": nonzeros,
                           "matches_saved_cleaned_row": matching})
        empty = vectorize_document("", name, already_cleaned=True)
        oov = vectorize_document(oov_token, name, already_cleaned=True)
        empty_nonzero = int(empty.nnz if name in COUNT_FILES else np.count_nonzero(empty))
        oov_nonzero = int(oov.nnz if name in COUNT_FILES else np.count_nonzero(oov))
        assert all(check["matches_saved_cleaned_row"] for check in checks)
        assert empty_nonzero == 0
        assert (oov_nonzero > 0) == (name == "fasttext")
        result["sample_checks"] = checks
        result["oov_check"] = {"token": oov_token, "nonzero": oov_nonzero}
        report["representations"][name] = result
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count-only", action="store_true")
    args = parser.parse_args()
    results = evaluate(args.count_only)
    print(json.dumps(results, indent=2))
    if results["missing_artifacts"]:
        sys.exit(1)

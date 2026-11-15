"""Load Assignment 2 artifacts and transform raw or already-cleaned documents."""

from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np

ROOT = Path(__file__).resolve().parent
COUNT_FILES = {
    "bow": "bow.joblib",
    "tfidf": "tfidf.joblib",
    "bigram": "unigrams_bigrams.joblib",
    "trigram": "unigrams_bigrams_trigrams.joblib",
}
EMBEDDING_FILES = {
    "cbow": "word2vec_cbow.model",
    "skipgram": "word2vec_skipgram.model",
    "fasttext": "fasttext.model",
}


@lru_cache(maxsize=None)
def load_representation(representation):
    """Load a trusted team artifact once per process."""
    if representation in COUNT_FILES:
        path = ROOT / "artifacts" / "assignment2" / COUNT_FILES[representation]
        if not path.is_file():
            raise FileNotFoundError(f"{path} is missing. Run assignment2_save_vectorizers.py.")
        return joblib.load(path)
    if representation in EMBEDDING_FILES:
        path = ROOT / "models" / EMBEDDING_FILES[representation]
        if not path.is_file():
            raise FileNotFoundError(f"{path} is missing. Obtain Hadia's model and companion files.")
        from gensim.models import FastText, Word2Vec

        loader = FastText if representation == "fasttext" else Word2Vec
        return loader.load(str(path)).wv
    raise ValueError(f"Unknown representation {representation!r}. Use {list(COUNT_FILES | EMBEDDING_FILES)}.")


def vectorize_document(document, representation, *, already_cleaned=False):
    """Return a one-row CSR matrix (counts) or a one-row dense array (embeddings).

    Raw text goes through Hadia's Assignment 1 cleaner. An empty embedding
    input, or all-OOV input for Word2Vec, yields a zero vector. The fastText
    branch uses inferred subword vectors when available.
    """
    if not isinstance(document, str):
        raise TypeError("document must be a string")
    artifact = load_representation(representation)
    if not already_cleaned and document.strip():
        from scripts.clean_toxicchat import clean_text

        document = clean_text(document)
    if representation in COUNT_FILES:
        return artifact.transform([document]).tocsr()

    from gensim.utils import simple_preprocess

    tokens = simple_preprocess(document, deacc=False, min_len=1)
    vectors = []
    for token in tokens:
        try:
            vectors.append(artifact.get_vector(token))
        except KeyError:
            continue
    if not vectors:
        return np.zeros((1, artifact.vector_size), dtype=np.float32)
    return np.mean(vectors, axis=0, keepdims=True)

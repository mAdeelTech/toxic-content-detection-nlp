# Assignment 2 · Parts C and D

## Reproduce and load

Use Python 3.11 or 3.12. From the root of the evaluation branch, install
`requirements.txt` in a virtual environment. The version pins match the
serialization and evaluation run. Use the exact Assignment 1 CSVs from the
evaluation branch's base commit `cb75760`.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -c "from scripts.clean_toxicchat import ensure_nltk_resources; ensure_nltk_resources()"
.\.venv\Scripts\python.exe assignment2_save_vectorizers.py
.\.venv\Scripts\python.exe assignment2_probe.py --count-only
```

The save script reproduces the fitted configurations in Adeel's notebook at
`61a181a`. It keeps existing artifacts. The `bigram` and `trigram` selectors
use Adeel's cumulative ranges `(1, 2)` and `(1, 3)`.

For a normal handoff, copy Hadia's existing artifacts into `models/`, keeping
all companion files beside the main files. Expected model filenames are
`word2vec_cbow.model`, `word2vec_skipgram.model`, and `fasttext.model`. If the
team needs to regenerate the models, use Hadia's unchanged
`notebooks/05_word_embeddings.ipynb` at `3cab4af`. Start the notebook kernel
with `PYTHONHASHSEED=42` (PowerShell `$env:PYTHONHASHSEED = '42'`), then run all
cells against the same training CSV. This verification run used that notebook
with Gensim 4.3.3, seed 42, one worker, 20 epochs, and 100 dimensions. Its
fastText run also produced `fasttext.model.wv.vectors_ngrams.npy`.
Use the exact handed-off files when comparing scores across machines.

```powershell
.\.venv\Scripts\python.exe assignment2_probe.py
```

The JSON on stdout has full top-five lists, four cosine values per model,
analogy results, sparsity and test-row checks. A run with missing files
reports their paths and returns a nonzero exit code.

## Phase 3 interface

```python
from assignment2_vectorize import vectorize_document

x = vectorize_document("Can you help me write a short message?", "tfidf")
```

Selectors are `bow`, `tfidf`, `bigram`, `trigram`, `cbow`, `skipgram`,
and `fasttext`. The raw-input route calls the Assignment 1 cleaner. Use
`already_cleaned=True` for text from the processed CSV. Each result has shape
`(1, feature_dimension)`. The count family returns a CSR sparse matrix.
The embedding family returns a dense mean of token vectors. Empty embedding
input gives a zero row. Keep the scikit-learn version used to save the joblib
files, and load artifacts from trusted teammates.

## Contribution statement

Shahzaib Khan, L1F23BSCS1062 — Parts C and D. This contribution implements and
runs word-neighbour, cosine and analogy probes, compares representations on
the Assignment 1 corpus, saves and reloads Adeel's vectorizer configurations,
and verifies a shared raw-document interface on actual held-out documents.
Adeel's Part A settings and Hadia's Part B training notebook are credited
separately. Add the final commit/PR link and disclose AI assistance according
to the course's submission rules.

# Assignment 1 · Task 3

Open `notebooks/02_tokenization_regex.ipynb` in VS Code or Jupyter with the
project's Python 3.11 environment. The notebook uses pandas, NLTK, spaCy and,
when the team CSV is unavailable, Hugging Face datasets.

For the existing terminal environment, install from the team's requirements.
If this branch is checked out separately, install the notebook dependencies:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install datasets==3.6.0 pandas==2.2.3 nltk==3.9.1 spacy==3.8.7 ipykernel
```

Run all notebook cells. On an integrated checkout, the notebook can reuse the
`original_text` column of the team's existing Assignment 1 training CSV. On
this branch by itself it downloads the official ToxicChat training split.

## Verified output

5,082 training rows were read from the team CSV in the verification run.

| Training row | Prompt type | NLTK tokens | spaCy tokens | Exact match |
|---:|---|---:|---:|---|
| 432 | Short contraction | 9 | 9 | Yes |
| 5 | Long prompt | 55 | 73 | No |
| 699 | URL prompt | 16 | 12 | No |

The contraction on row 432 was split into `Ca` and `n't` by both methods.
On row 699 the isolated URL produced five Treebank tokens and one spaCy token.
Dataset-wide pattern detection matched 38 URL rows, 31 HTML rows and 623 rows
with repeated whitespace.

## Contribution statement

Task 3 — Shahzaib Khan, L1F23BSCS1062. The Task 3 submission extends Adeel's
starter notebook with an NLTK Treebank and spaCy comparison, three regexes
with corpus match counts, and evidence for contraction and URL handling.
Add the final commit permalink to the report after pushing.

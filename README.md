# toxic-content-detection-nlp
NLP Project — Toxic Content Detection using LMSYS ToxicChat


## Text cleaning contribution (Hadia)

The reproducible preprocessing notebook is [`notebooks/03_text_cleaning.ipynb`](notebooks/03_text_cleaning.ipynb). A standalone script is available at [`scripts/clean_toxicchat.py`](scripts/clean_toxicchat.py).

### Run the cleaning pipeline

Use Python 3.11+:

```bash
python -m venv .venv
# Windows PowerShell:
.venv\\Scripts\\Activate.ps1
# macOS/Linux:
source .venv/bin/activate
python -m pip install -r requirements.txt
python scripts/clean_toxicchat.py
```

The script downloads `lmsys/toxic-chat` configuration `toxicchat0124` and writes:
- `data/processed/toxicchat_train_cleaned.csv`
- `data/processed/toxicchat_test_cleaned.csv`

Do not commit raw dataset files. The generated CSVs are local outputs unless the team agrees on a separate storage policy. See [Hadia's cleaning report section](reports/hadia_cleaning_section.md) for method choices and the contribution statement.

"""Reproducible LMSYS ToxicChat cleaning pipeline.

Run:
    python scripts/clean_toxicchat.py

The script downloads the public dataset and writes cleaned train/test CSVs
under data/processed/. Do not commit raw dataset files.
"""
from pathlib import Path
import re
import unicodedata

import nltk
import pandas as pd
from datasets import load_dataset
from nltk.stem import WordNetLemmatizer

DATASET_NAME = "lmsys/toxic-chat"
DATASET_CONFIG = "toxicchat0124"
TEXT_COLUMN = "user_input"
LABEL_COLUMN = "toxicity"
OUTPUT_DIR = Path("data/processed")

TOKEN_PATTERN = re.compile(r"\\w+(?:['’]\\w+)*|[^\\w\\s]", flags=re.UNICODE)
LEMMATIZER = WordNetLemmatizer()


def ensure_nltk_resources():
    for resource in ("averaged_perceptron_tagger_eng", "wordnet", "omw-1.4"):
        nltk.download(resource, quiet=True)


def penn_to_wordnet(tag):
    if tag.startswith("J"):
        return "a"
    if tag.startswith("V"):
        return "v"
    if tag.startswith("N"):
        return "n"
    if tag.startswith("R"):
        return "r"
    return None


def restore_case(original, lemma):
    if original.isupper():
        return lemma.upper()
    if original[:1].isupper():
        return lemma[:1].upper() + lemma[1:]
    return lemma


def clean_text(text):
    """Apply NFC, conservative URL/mention replacement and POS-aware lemmatization."""
    if pd.isna(text):
        return ""

    text = unicodedata.normalize("NFC", str(text))
    text = re.sub(r"https?://\\S+|www\\.\\S+", " URL ", text, flags=re.IGNORECASE)
    text = re.sub(r"(?<!\\w)@\\w+", " USER ", text)

    tokens = TOKEN_PATTERN.findall(text)
    if not tokens:
        return ""

    tagged_tokens = nltk.pos_tag(tokens)
    cleaned_tokens = []
    for token, tag in tagged_tokens:
        wn_pos = penn_to_wordnet(tag)
        if wn_pos and re.search(r"\\w", token, flags=re.UNICODE):
            lemma = LEMMATIZER.lemmatize(token.lower(), pos=wn_pos)
            token = restore_case(token, lemma)
        cleaned_tokens.append(token)

    cleaned = " ".join(cleaned_tokens)
    cleaned = re.sub(r"\\s+([,.;:!?%])", r"\\1", cleaned)
    cleaned = re.sub(r"([([{])\\s+", r"\\1", cleaned)
    cleaned = re.sub(r"\\s+([)\\]}])", r"\\1", cleaned)
    return re.sub(r"\\s+", " ", cleaned).strip()


def main():
    ensure_nltk_resources()
    dataset = load_dataset(DATASET_NAME, DATASET_CONFIG)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for split_name in ("train", "test"):
        frame = dataset[split_name].to_pandas().copy()
        missing = {TEXT_COLUMN, LABEL_COLUMN}.difference(frame.columns)
        if missing:
            raise ValueError(f"{split_name} split is missing columns: {sorted(missing)}")

        frame["original_text"] = frame[TEXT_COLUMN]
        frame["cleaned_text"] = frame["original_text"].apply(clean_text)
        output = OUTPUT_DIR / f"toxicchat_{split_name}_cleaned.csv"
        frame[["original_text", "cleaned_text", LABEL_COLUMN]].to_csv(output, index=False)

        print(f"{split_name}: {len(frame)} rows -> {output}")
        print("Null cleaned values:", frame["cleaned_text"].isna().sum())
        print("Empty cleaned values:", (frame["cleaned_text"].str.strip() == "").sum())


if __name__ == "__main__":
    main()

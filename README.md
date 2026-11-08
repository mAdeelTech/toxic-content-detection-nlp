# Toxic Content Detection using LMSYS ToxicChat

## Project overview

This repository contains the team implementation for the Introduction to NLP semester project. The project investigates text preprocessing and cleaning for a toxic-content classification task using the LMSYS ToxicChat dataset.

**Current phase:** Assignment 1 — Data Acquisition, Text Processing & Cleaning.

## Topic and dataset

- **Topic:** Toxic Content Detection
- **Dataset:** LMSYS ToxicChat
- **Source:** https://huggingface.co/datasets/lmsys/toxic-chat
- **Planned input text:** `user_input`
- **Planned prediction target:** `toxicity`

The exact dataset configuration, row count, column names, label values, class distribution, and licence will be verified against the downloaded dataset and its official dataset card before being reported. Do not treat the planned column names above as verified until that inspection is complete.

## Repository structure

```text
data/
  raw/          Original downloaded data (kept out of Git)
  processed/    Reproducibly generated cleaned data
notebooks/      Exploration notebook
src/            Reusable Python scripts and NLP pipeline
reports/
  screenshots/  Evidence screenshots for the report
requirements.txt
README.md
```

## Environment

- Python 3.11 or newer
- A virtual environment (`.venv`)
- Dependencies listed in `requirements.txt`

Environment setup and exact run commands will be documented after the scripts are implemented and tested.

## Data policy

- Do not commit the original/raw dataset to this repository.
- Keep downloaded raw files under `data/raw/` locally.
- Generate processed data using the project pipeline; do not edit the output manually.
- Record the dataset licence and attribution requirements after checking the official source.

## Assignment 1 workflow

1. Verify the dataset schema, labels, licence, and size.
2. Load the dataset and perform exploratory statistics.
3. Compare at least two tokenization methods.
4. implement and explain at least three task-relevant regular expressions.
5. Build one reusable cleaning pipeline with justified normalization choices.
6. Save the cleaned text and labels through code.
7. Test reproducibility from a fresh environment.
8. Prepare the 4–6 page PDF report and contribution statement.

## Team

Add the approved team name, member names, and university roll numbers here after confirming them with the team. The instructor and team members are invited to the repository.

## Status

Project structure is being prepared. Dataset statistics, preprocessing decisions, and final run instructions will be added only after they have been verified by executing the project code.

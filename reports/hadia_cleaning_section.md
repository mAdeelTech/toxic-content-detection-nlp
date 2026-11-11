# Hadia — Text Cleaning and Preprocessing Report Section

## Dataset and objective

This work uses the LMSYS ToxicChat `toxicchat0124` configuration. The text input is `user_input`, and the binary prediction target is `toxicity` (0 = non-toxic, 1 = toxic). The dataset contains approximately 10,165 user prompts across its train and test splits. The dataset card identifies the license as CC-BY-NC-4.0; this is a non-commercial license, so the dataset terms must be followed.

Source: https://huggingface.co/datasets/lmsys/toxic-chat

## Cleaning pipeline

The cleaning function applies operations in a fixed, reusable order:

1. **Null handling:** missing text is converted to an empty string so the function always returns a string.
2. **Unicode normalization:** NFC is applied before tokenization to make canonically equivalent character sequences consistent while avoiding the broader compatibility folding of NFKC.
3. **URL handling:** URLs are replaced with the placeholder `URL` to avoid high-cardinality URL strings while retaining the fact that a link appeared.
4. **Mention handling:** user handles are replaced with `USER` to reduce identifier-specific vocabulary.
5. **Tokenization and lemmatization:** text is tokenized with a Unicode-aware regular expression, POS-tagged with NLTK, and lemmatized with WordNet using the mapped part of speech. Original capitalization is restored after lookup.
6. **Whitespace reconstruction:** spacing is normalized and punctuation is attached to the preceding token.

Capitalization, punctuation, numbers, contractions, hashtags, and stopwords are retained. In particular, negations such as `not`, `no`, and `never` are preserved because removing them can reverse sentence meaning. Stemming is demonstrated for comparison but is not used in the final pipeline. POS-aware lemmatization is selected because it generally produces interpretable dictionary forms, although slang and misspellings may remain unchanged.

## Validation and output

The notebook applies the same deterministic `clean_text(text)` function to training and test data. It preserves `original_text`, creates `cleaned_text`, checks null and empty outputs, compares before/after examples, and calculates token and vocabulary counts from the actual training data.

**Fill these values after running the notebook — do not guess:**

- Training rows: [copy runtime output]
- Testing rows: [copy runtime output]
- Vocabulary before cleaning: [copy runtime output]
- Vocabulary after cleaning: [copy runtime output]
- Empty cleaned training rows: [copy runtime output]
- Empty cleaned testing rows: [copy runtime output]

The notebook writes `data/processed/toxicchat_train_cleaned.csv` and `data/processed/toxicchat_test_cleaned.csv` locally. Raw dataset files are not committed to the repository.

## Limitations

WordNet is English-oriented and may not recognize slang, misspellings, or newly coined abusive terms. Replacing URLs and mentions also removes their literal content, though placeholders preserve their presence. The selected choices should be evaluated in the later modelling phase rather than assumed to improve classification.

## References

- LMSYS ToxicChat dataset card: https://huggingface.co/datasets/lmsys/toxic-chat
- Lin et al. (2023), *ToxicChat: Unveiling Hidden Challenges of Toxicity Detection in Real-World User-AI Conversation*, arXiv:2310.17389.

## Individual contribution statement (edit if needed)

Hadia was responsible for the text-cleaning and preprocessing component of the NLP project. Her assigned work includes inspecting raw ToxicChat prompts, comparing Unicode normalization and case-handling strategies, implementing conservative URL and mention handling, documenting stopword and negation decisions, comparing stemming with lemmatization, implementing and validating the reusable cleaning function, and generating cleaned train/test outputs for the next project phase. Group member list: Muhammad Adeel (1072), Shahzaib Khan (1062), and Hadia Manzoor (1139). This document describes Hadia's assigned contribution only; other members' individual contributions should be added by the group based on their actual work. Add the final commit link and screenshots after the notebook has been run successfully.

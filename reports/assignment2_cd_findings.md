# Assignment 2 · Measured results

Parts C and D use the same 5,082 training rows as Parts A and B. Each embedding model has 6,375 words in its training vocabulary. The shared function was tested on the first three held-out rows. Passing in the raw prompt gave the same vector as passing in its saved cleaned text for all seven representations.

## C1 · Nearest neighbours

Cosines and top-five words for each query are in [the complete output](assignment2_cd_results.json).

| Query | CBOW top match | Skip-gram top match | fastText top match |
|---|---|---|---|
| hate | therefore (0.8820) | racism (0.7859) | kate (0.8506) |
| abuse | outer (0.9520) | chastity (0.8254) | fuse (0.7966) |
| stupid | hornet (0.9470) | jew (0.8310) | stuff (0.7909) |
| help | teach (0.7337) | assist (0.6025) | helo (0.7568) |
| write | create (0.7915) | compose (0.6743) | overwrite (0.8073) |

The matches for `write` were easy to explain. Both `create` and `compose` could occur in similar writing prompts. For `help`, Skip-gram found `assist`, whereas CBOW found the more general association `teach`. With fastText, `hate` and `kate` share a spelling pattern, and `overwrite` contains `write` in full. Those matches suggest that character overlap influenced the ranking. Some high scores were harder to explain. CBOW gave `abuse` and `outer` a cosine of 0.9520 even though the words have no obvious semantic connection.

## C2 · Word-pair cosine

| Expected relationship | Pair | CBOW | Skip-gram | fastText |
|---|---|---:|---:|---:|
| related | hate / dislike | 0.7405 | 0.5271 | 0.3702 |
| related | abuse / offensive | 0.3490 | 0.3789 | 0.5267 |
| unrelated | hate / recipe | 0.5233 | 0.2306 | 0.2583 |
| unrelated | stupid / python | 0.2861 | 0.2619 | 0.3877 |

Skip-gram was the only model where both expected related pairs scored above both expected unrelated pairs. The clearest exception for CBOW was `hate / recipe` (0.5233), which scored above `abuse / offensive` (0.3490). In fastText, `stupid / python` also scored slightly above `hate / dislike`. Four pairs are too few to decide which model would perform best on toxicity classification.

## C3 · Analogy

Computed `woman − man + king` with raw vectors and ranked other vocabulary words by cosine. The expected word was `queen`, which is in all three vocabularies.

| Model | Top result | Cosine |
|---|---|---:|
| cbow | people | 0.7567 |
| skipgram | castle | 0.5280 |
| fasttext | working | 0.8260 |

The analogy failed in all three runs. Skip-gram's answer, `castle`, is related to `king` by topic, but the operation was meant to recover the relation between `king` and `queen`. The royal terms are uncommon in this corpus, so there are few examples from which to learn that relation.

## C4 · Document representation

| Representation | Features | Zero entries on training set | What it captures | OOV behavior |
|---|---:|---:|---|---|
| bow | 13,237 | 99.8529% | Unigram frequency | Ignores unseen features |
| tfidf | 13,237 | 99.8529% | Unigram counts weighted by corpus rarity | Ignores unseen features |
| bigram | 84,059 | 99.9473% | Unigrams and adjacent word pairs | Ignores unseen features |
| trigram | 185,597 | 99.9626% | Unigrams and adjacent sequences of up to three words | Ignores unseen features |
| cbow | 100 | 0.0590% | Context-to-target word vectors, averaged per document | Skips unknown tokens |
| skipgram | 100 | 0.0590% | Target-to-context word vectors, averaged per document | Skips unknown tokens |
| fasttext | 100 | 0.0197% | Word and character subword structure, averaged per document | Infers vectors from subwords |

Adding bigrams and trigrams increased the number of features from 13,237 to 185,597 and made the count matrix sparser. The extra features record short sequences of words. The embedding methods used just 100 features per document, but averaging the token vectors removed word order. For the made-up word `unseenwordxyz`, fastText returned 100 nonzero values and the other six methods returned zero rows. This check shows that fastText could build a vector from pieces of that word. It gives no evidence about how meaningful that vector is.

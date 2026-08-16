# Architecture

## Pipeline

```
Reviews.csv (Amazon Fine Food Reviews)
        │
        ▼
  Label from star rating
  (>=4 → positive, <=2 → negative, 3 dropped)
        │
        ▼
  Sample 8,000 positive / 2,000 negative
        │
        ▼
  Text cleaning
  (lowercase, expand contractions, strip URLs/punctuation,
   remove stopwords, lemmatize)
        │
        ├────────────────────────────┐
        ▼                            │
  Baseline fine-tune                 │
  bert-base-uncased, 2 epochs        │
  → 87.5% accuracy                   │
        │                            │
        ▼                            ▼
  WordNet synonym-replacement augmentation
  (3 synthetic negative reviews generated per
   real negative review, to rebalance classes)
        │
        ▼
  Final fine-tune on augmented dataset
  bert-base-uncased, 2 epochs
  → 95.1% accuracy, 0.95 macro F1
        │
        ▼
  Push model + tokenizer to Hugging Face Hub
        │
        ▼
  Streamlit app loads model from the Hub for inference
```

## Model

- **Base model:** `bert-base-uncased`
- **Task:** binary sequence classification (positive / negative)
- **Max sequence length:** 128 tokens
- **Training:** Hugging Face `Trainer`, 2 epochs, batch size 8

## Why WordNet synonym replacement instead of a GAN

A GAN-based augmenter was considered early on, but our mentor pointed out
that GAN-based text augmentation is a common (and often overused) approach,
and suggested exploring an alternative. We settled on WordNet synonym
replacement (via `nltk.corpus.wordnet`): each negative review has one word
swapped for a synonym, generating three synthetic variants per original
review. This is a lightweight, dependency-light way to expand the minority
class without needing to train a separate generative model.

## Class imbalance note

The source dataset skews roughly 4:1 positive to negative. The sampled
training set (8,000 positive / 2,000 negative) mirrors that imbalance
directly. Augmentation only partially closes the gap — it does not fully
balance the classes — which is worth keeping in mind when interpreting
per-class metrics.

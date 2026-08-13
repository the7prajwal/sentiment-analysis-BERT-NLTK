"""
train.py
Sentiment Analysis - BERT fine-tuning with GAN-style (WordNet synonym) augmentation.

Pipeline:
1. Load and clean Amazon Fine Food Reviews (Reviews.csv)
2. Convert star ratings into binary sentiment labels
3. Clean/preprocess review text
4. Fine-tune bert-base-uncased on the imbalanced dataset (baseline)
5. Augment the minority (negative) class using WordNet synonym replacement
6. Re-train on the augmented, more balanced dataset (final model)
7. Save and push the final model + tokenizer to the Hugging Face Hub

Originally developed in Google Colab; this script is the cleaned, de-duplicated
version of that notebook for reproducibility outside Colab.
"""

import random

import numpy as np
import pandas as pd
import contractions
import nltk
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

from datasets import Dataset
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    Trainer,
    TrainingArguments,
)

import re

# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------
DATA_PATH = "Reviews.csv"  # Amazon Fine Food Reviews dataset
BASE_MODEL = "bert-base-uncased"
HUB_MODEL_ID = "sentiment-analysis-bert-model"

N_POSITIVE_SAMPLES = 8000
N_NEGATIVE_SAMPLES = 2000
AUGMENTATIONS_PER_NEGATIVE_REVIEW = 3  # synthetic copies generated per negative review

RANDOM_STATE = 42
MAX_LENGTH = 128
TEST_SIZE = 0.2

TRAINING_ARGS = TrainingArguments(
    output_dir="./results",
    num_train_epochs=2,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    logging_dir="./logs",
    logging_steps=50,
)


# --------------------------------------------------------------------------
# Step 1: Load data
# --------------------------------------------------------------------------
def load_data(path: str) -> pd.DataFrame:
    """Load the Amazon Fine Food Reviews CSV, tolerating malformed rows."""
    df = pd.read_csv(path, engine="python", on_bad_lines="skip")
    return df


# --------------------------------------------------------------------------
# Step 2: Label sentiment from star rating
# --------------------------------------------------------------------------
def sentiment_label(score: float):
    """4-5 stars -> positive (1), 1-2 stars -> negative (0), 3 stars -> dropped."""
    if score >= 4:
        return 1
    elif score <= 2:
        return 0
    return np.nan


def build_labeled_dataset(df: pd.DataFrame) -> pd.DataFrame:
    df = df[["Text", "Score"]].copy()
    df["Sentiment"] = df["Score"].apply(sentiment_label)
    df = df.dropna()

    positive = df[df["Sentiment"] == 1].sample(N_POSITIVE_SAMPLES, random_state=RANDOM_STATE)
    negative = df[df["Sentiment"] == 0].sample(N_NEGATIVE_SAMPLES, random_state=RANDOM_STATE)
    return pd.concat([positive, negative])


# --------------------------------------------------------------------------
# Step 3: Text cleaning
# --------------------------------------------------------------------------
def clean_text(text: str, stop_words: set, lemmatizer: WordNetLemmatizer) -> str:
    text = text.lower()
    text = contractions.fix(text)
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)

    words = text.split()
    words = [w for w in words if w not in stop_words]
    words = [lemmatizer.lemmatize(w) for w in words]
    return " ".join(words)


def preprocess_text(df: pd.DataFrame) -> pd.DataFrame:
    nltk.download("stopwords")
    nltk.download("wordnet")

    stop_words = set(stopwords.words("english"))
    lemmatizer = WordNetLemmatizer()

    df["Cleaned_Text"] = df["Text"].apply(lambda t: clean_text(t, stop_words, lemmatizer))
    return df[["Cleaned_Text", "Sentiment"]]


# --------------------------------------------------------------------------
# Step 4: Tokenize + build HF Datasets
# --------------------------------------------------------------------------
def tokenize_split(df: pd.DataFrame, tokenizer: BertTokenizer):
    train_texts, test_texts, train_labels, test_labels = train_test_split(
        df["Cleaned_Text"].tolist(),
        df["Sentiment"].tolist(),
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )
    train_labels = [int(l) for l in train_labels]
    test_labels = [int(l) for l in test_labels]

    train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=MAX_LENGTH)
    test_encodings = tokenizer(test_texts, truncation=True, padding=True, max_length=MAX_LENGTH)

    train_dataset = Dataset.from_dict({
        "input_ids": train_encodings["input_ids"],
        "attention_mask": train_encodings["attention_mask"],
        "labels": train_labels,
    })
    test_dataset = Dataset.from_dict({
        "input_ids": test_encodings["input_ids"],
        "attention_mask": test_encodings["attention_mask"],
        "labels": test_labels,
    })
    return train_dataset, test_dataset, test_labels


# --------------------------------------------------------------------------
# Step 5: Train + evaluate
# --------------------------------------------------------------------------
def train_and_evaluate(train_dataset, test_dataset, test_labels, label: str):
    model = BertForSequenceClassification.from_pretrained(BASE_MODEL, num_labels=2)

    trainer = Trainer(
        model=model,
        args=TRAINING_ARGS,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
    )
    trainer.train()

    predictions = trainer.predict(test_dataset)
    preds = np.argmax(predictions.predictions, axis=1)

    print(f"\n=== {label} ===")
    print("Accuracy:", accuracy_score(test_labels, preds))
    print(classification_report(test_labels, preds))

    return model, trainer


# --------------------------------------------------------------------------
# Step 6: WordNet synonym-replacement augmentation (minority class: negative)
# --------------------------------------------------------------------------
def synonym_replacement(sentence: str) -> str:
    """Replace the first word in the sentence that has a WordNet synonym."""
    words = sentence.split()
    new_words = words.copy()

    for i, word in enumerate(words):
        synonyms = wordnet.synsets(word)
        if not synonyms:
            continue

        synonym_words = list({lemma.name() for syn in synonyms for lemma in syn.lemmas()})
        if synonym_words:
            replacement = random.choice(synonym_words)
            if replacement != word:
                new_words[i] = replacement
                break

    return " ".join(new_words)


def augment_minority_class(df: pd.DataFrame) -> pd.DataFrame:
    negative_reviews = df[df["Sentiment"] == 0]["Cleaned_Text"].tolist()

    synthetic_reviews = []
    for review in negative_reviews:
        try:
            for _ in range(AUGMENTATIONS_PER_NEGATIVE_REVIEW):
                synthetic_reviews.append(synonym_replacement(review))
        except Exception:
            continue

    synthetic_df = pd.DataFrame({"Cleaned_Text": synthetic_reviews, "Sentiment": 0})
    return pd.concat([df, synthetic_df])


# --------------------------------------------------------------------------
# Step 7: Save + push to Hugging Face Hub
# --------------------------------------------------------------------------
def save_and_push(model, tokenizer, local_dir: str = "sentiment_model"):
    model.save_pretrained(local_dir)
    tokenizer.save_pretrained(local_dir)

    # Requires `huggingface_hub` login (`huggingface-cli login` or `login()`)
    # with a token that has write access, run once before calling this.
    model.push_to_hub(HUB_MODEL_ID)
    tokenizer.push_to_hub(HUB_MODEL_ID)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main():
    df = load_data(DATA_PATH)
    df = build_labeled_dataset(df)
    df = preprocess_text(df)

    tokenizer = BertTokenizer.from_pretrained(BASE_MODEL)

    # Baseline: train on the raw, imbalanced (8000/2000) dataset
    train_dataset, test_dataset, test_labels = tokenize_split(df, tokenizer)
    train_and_evaluate(train_dataset, test_dataset, test_labels, label="Baseline (pre-augmentation)")

    # Augment the negative class, then retrain on the more balanced dataset
    augmented_df = augment_minority_class(df)
    train_dataset, test_dataset, test_labels = tokenize_split(augmented_df, tokenizer)
    final_model, _ = train_and_evaluate(
        train_dataset, test_dataset, test_labels, label="Final (post-augmentation)"
    )

    save_and_push(final_model, tokenizer)


if __name__ == "__main__":
    main()

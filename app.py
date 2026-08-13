import streamlit as st
import torch
from transformers import BertTokenizer, BertForSequenceClassification

# Load model and tokenizer
model = BertForSequenceClassification.from_pretrained(
    "Sasaki2801/sentiment-analysis-bert-model"
)

tokenizer = BertTokenizer.from_pretrained(
    "Sasaki2801/sentiment-analysis-bert-model"
)

model.eval()

# Title
st.title("Sentiment Analysis using GAN-Augmented BERT")

st.write("Enter a product review below:")

# User input
review = st.text_area("Review Text")

# Prediction
if st.button("Predict Sentiment"):

    inputs = tokenizer(
        review,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )

    with torch.no_grad():
        outputs = model(**inputs)

    prediction = torch.argmax(outputs.logits, dim=1).item()

    confidence = torch.softmax(outputs.logits, dim=1)[0][prediction].item()

    if prediction == 1:
        st.success("Positive Review 😊")
    else:
        st.error("Negative Review 😠")

    st.write(f"Confidence Score: {confidence:.2f}")
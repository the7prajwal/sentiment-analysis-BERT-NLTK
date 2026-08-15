<img width="1920" height="1200" alt="Screenshot 2026-08-15 181232" src="https://github.com/user-attachments/assets/153ca597-190a-4322-8167-5ee0b7cf05b3" />A sentiment classifier fine-tuned on the Amazon Fine Food Reviews dataset, using
WordNet-based synthetic augmentation to address class imbalance, deployed as a
live Streamlit app backed by a model hosted on Hugging Face.


## Links

| | |
|---|---|
| Dataset Used | https://www.kaggle.com/datasets/dongrelaxman/amazon-reviews-dataset |
| Google Colab | https://colab.research.google.com/drive/1GCxkUvlMCRDJFK5HqDZ1JDKssXL_fD6L |
| Live Demo | https://sentiment-analysis-bert-6qf92de5ynrg2r9gknjbvv.streamlit.app/ |
| Model | https://huggingface.co/Sasaki2801/sentiment-analysis-bert-model |

## Web Application

The trained BERT model is deployed through a Streamlit web application that
accepts product reviews and predicts their sentiment along with a confidence
score.

### Positive Sentiment

![Positive sentiment prediction]:<img width="1920" height="1200" alt="positive" src="https://github.com/user-attachments/assets/34ae245c-4555-4167-b025-8129a17bc356" />



### Negative Sentiment

![Negative sentiment prediction]:<img width="1920" height="1200" alt="negative" src="https://github.com/user-attachments/assets/d3825a95-222a-4b58-ac7a-340f5b1d38a3" />



## Overview

The dataset is naturally imbalanced (roughly 4:1 positive-to-negative reviews).
Training directly on that imbalance biases the model toward predicting
"positive." To address this, synthetic negative reviews are generated via
WordNet synonym replacement and added to the training set before a final
fine-tuning pass. See [`ARCHITECTURE.md`](ARCHITECTURE.md) for the full
pipeline breakdown.

## Model Performance

| Metric | Baseline | With Data Augmentation | Improvement |
|---|---:|---:|---:|
| Accuracy | 87.45% | **95.13%** | **+7.68 pp** |
| Macro Precision | 0.81 | **0.95** | **+0.14** |
| Macro Recall | 0.76 | **0.95** | **+0.19** |
| Macro F1-Score | 0.78 | **0.95** | **+0.17** |
| Weighted F1-Score | 0.87 | **0.95** | **+0.08** |

### Per-Class Performance

| Class | Metric | Baseline | With Data Augmentation |
|---|---|---:|---:|
| 0 | Precision | 0.72 | **0.94** |
| 0 | Recall | 0.58 | **0.96** |
| 0 | F1-Score | 0.64 | **0.95** |
| 1 | Precision | 0.90 | **0.96** |
| 1 | Recall | 0.94 | **0.94** |
| 1 | F1-Score | 0.92 | **0.95** |


## Repo structure

```
.
├── app.py              # Streamlit inference app
├── train.py            # Full training pipeline (preprocessing, BERT fine-tuning, augmentation)
├── requirements.txt     # Python dependencies
├── ARCHITECTURE.md     # Pipeline / model design details
├── DEPLOYMENT.md       # How the app is deployed, and how to redeploy
└── LICENSE
```

## Quickstart

```bash
git clone https://github.com/Sasaki-28/sentiment-analysis-bert.git
cd sentiment-analysis-bert
pip install -r requirements.txt
streamlit run app.py
```

The app loads the fine-tuned model directly from the Hugging Face Hub
(`Sasaki2801/sentiment-analysis-bert-model`), so no local model files are
needed to run inference.

## Retraining the model

`train.py` reproduces the full pipeline against a local `Reviews.csv`
(Amazon Fine Food Reviews). See [`ARCHITECTURE.md`](ARCHITECTURE.md) for
details on each stage, and [`DEPLOYMENT.md`](DEPLOYMENT.md) for how a newly
trained model gets pushed to the Hub and picked up by the live app.

```bash
python train.py
```

## Contributors

- [Sasaki-28](https://github.com/Sasaki-28)
- [Prajwal](https://github.com/the7prajwal)

## License

MIT — see [`LICENSE`](LICENSE).

# Sentiment Analysis — GAN-Augmented BERT

A sentiment classifier fine-tuned on the Amazon Fine Food Reviews dataset, using
WordNet-based synthetic augmentation to address class imbalance, deployed as a
live Streamlit app backed by a model hosted on Hugging Face.

**Live demo:** https://sentiment-analysis-bert-6qf92de5ynrg2r9gknjbvv.streamlit.app/
**Model:** https://huggingface.co/Sasaki2801/sentiment-analysis-bert-model

## Overview

The dataset is naturally imbalanced (roughly 4:1 positive-to-negative reviews).
Training directly on that imbalance biases the model toward predicting
"positive." To address this, synthetic negative reviews are generated via
WordNet synonym replacement and added to the training set before a final
fine-tuning pass. See [`ARCHITECTURE.md`](ARCHITECTURE.md) for the full
pipeline breakdown.

| | Baseline (pre-augmentation) | Final (post-augmentation) |
|---|---|---|
| Accuracy | 87.5% | 95.1% |
| Macro F1 | — | 0.95 |

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
## Links

| | |
|---|---|
| Dataset Used | https://www.kaggle.com/datasets/dongrelaxman/amazon-reviews-dataset |
| Google Colab | https://colab.research.google.com/drive/1GCxkUvlMCRDJFK5HqDZ1JDKssXL_fD6L |
| Live Demo | https://sentiment-analysis-bert-6qf92de5ynrg2r9gknjbvv.streamlit.app/ |
| Model | https://huggingface.co/Sasaki2801/sentiment-analysis-bert-model |

## Contributors

- [Sasaki-28](https://github.com/Sasaki-28)
- [Prajwal](https://github.com/the7prajwal)

## License

MIT — see [`LICENSE`](LICENSE).

# Deployment

## Current setup

- **Frontend:** [Streamlit Community Cloud](https://streamlit.io/cloud), deployed from this repo's `main` branch, running `app.py`.
- **Model hosting:** [Hugging Face Hub](https://huggingface.co/Sasaki2801/sentiment-analysis-bert-model) — `app.py` loads the model and tokenizer directly via `from_pretrained("Sasaki2801/sentiment-analysis-bert-model")` at app startup, so the model itself is never bundled into this repo.
- **Code hosting:** this GitHub repo.

```
GitHub repo ──(Streamlit Cloud watches this branch)──▶ Streamlit Community Cloud
                                                                │
                                                                ▼
                                                     loads model at runtime from
                                                                │
                                                                ▼
                                                     Hugging Face Hub
                                                     (Sasaki2801/sentiment-analysis-bert-model)
```

## Redeploying after a code change

Streamlit Community Cloud auto-redeploys on every push to the connected
branch (`main`). Just:

```bash
git add .
git commit -m "your change"
git push origin main
```

No manual restart is needed unless dependencies changed significantly, in
which case restart the app from the Streamlit Cloud dashboard to force a
clean rebuild.

## Pushing an updated model

If `train.py` is re-run and produces a new model:

1. Authenticate once locally: `huggingface-cli login` (needs a Hugging Face token with **write** access).
2. `train.py` calls `model.push_to_hub(...)` and `tokenizer.push_to_hub(...)`, which uploads the new weights to the same `Sasaki2801/sentiment-analysis-bert-model` repo.
3. The Streamlit app doesn't need any code change to pick this up — it always pulls the latest version from the Hub on startup. Just restart the app from the Streamlit Cloud dashboard so it reloads with the new weights.

## Managing access

- **GitHub:** add teammates as collaborators under repo **Settings → Collaborators**.
- **Hugging Face model:** add teammates as collaborators under the model repo's **Settings → Collaborators**, or move the model into a shared HF Organization.
- **Streamlit Cloud app:** invite teammates as collaborators from the app's settings on [share.streamlit.io](https://share.streamlit.io).

## Environment / secrets

The app currently requires no secrets or API keys to run inference (the
model is public). If the Hugging Face model repo is ever made private, a
read-access token will need to be added as a Streamlit secret
(`HF_TOKEN` in **App settings → Secrets**) and referenced in `app.py`.

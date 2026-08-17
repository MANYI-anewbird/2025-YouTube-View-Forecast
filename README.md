# YouTube View Forecast

Predict how a video will perform **before you hit publish** — from the thumbnail and the metadata, not from hindsight.

BA865 Team 7 (May 2025). Multi-input neural net: a **CNN on the thumbnail** plus an **MLP on tabular features** (title symbols, timing, category, region, channel stats). Trained on **11,755 videos / 11,734 thumbnails** scraped from the YouTube Data API across US, Canada, India, UK, and Australia (Jan–Apr 2025). After dropping failed image downloads and missing values: **9,900 rows**.

Deck: [`docs/presentation.pdf`](docs/presentation.pdf) · [`docs/presentation.pptx`](docs/presentation.pptx)

---

## Why it exists

Creators spend hours on title, thumbnail, and timing, then find out too late whether the video will land. The two goals:

1. **Estimate views before publishing** from thumbnail + metadata.
2. **Show which knobs matter** (SHAP) so the next thumbnail or category choice is informed.

---

## What improved the model

Validation R² on log views, unless noted:

| Step | Val R² |
|---|---|
| Best tabular feature group | 0.595 |
| Multi-input baseline (CNN + MLP) | 0.638 |
| Thumbnail OCR / face features | 0.656 |
| **Log-view target transform** | **0.794** (raw views 0.763) |
| Model II (shallower convs, better for small n) | **0.809** (raw 0.775) |
| Bayesian hyperparameter search | 0.811 log / 0.769 raw |

The jump that actually mattered was **target transformation**, then a slightly smaller CNN (Model II) that fit the 9.9k-row set better than a deeper stack. Bayesian search was a small extra, not the whole story.

**SHAP (tabular):** likes dominate; Australia as a region; Sports category is a plus. **Thumbnail:** close-up faces help personality-driven content; otherwise bold readable text or a strong focal object.

---

## Repo layout

```
docs/presentation.pdf       # Team 7 slide deck (PDF)
docs/presentation.pptx      # same deck, editable
docs/processing-map.pdf     # collection → train set
notebooks/scrape_youtube_api.ipynb
notebooks/download_thumbnails.ipynb
notebooks/modeling/         # multi-input Keras experiments
```

Thumbnails and the full CSV are **not** in this repo (too large). Point the notebooks at a local data folder.

Set `YOUTUBE_API_KEY` in the environment before running the scraper. Do not hard-code keys.

---

## Setup

```sh
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## Known limits (from the deck)

- Scraper yield was uneven across regions; API quota is 10,000 units/day.
- Image-level SHAP was unreliable, so thumbnail advice is directional, not pixel-attribution.

---

## License

MIT. See [LICENSE](LICENSE).

# 2025 YouTube View Forecast

**The question:** If a creator has already chosen a thumbnail, title style, category, region, and publish window — will the video get views, or is it about to waste a slot?

This is a BA865 (Team 7, May 2025) demand-forecast study. It is not a growth hack list. It is a pre-publish **go / no-go** model: estimate performance from what you can still change, before the video is live.

Slides: [`docs/presentation.pdf`](docs/presentation.pdf)

---

## Decision this supports

Creators lock title, thumbnail, and timing, then discover the outcome too late. Re-cutting those after publish usually costs the first-day window.

Two decisions the work is built for:

1. **Ship or restyle** — is this package (thumbnail + metadata) in a range that has historically cleared, or is it a likely miss?
2. **What to change first** — of the levers still open, which ones actually moved predicted views in the data?

---

## What we found

Worked on **11,755** videos across the US, Canada, India, UK, and Australia (Jan–Apr 2025 YouTube Data API). After dropped thumbnails and missing fields: **9,900** training rows.

**The model can rank packages before publish.** On log views, validation R² moved from **0.656** (after thumbnail text/face features) to **0.809** once we forecast log views instead of raw counts, then used a smaller CNN that fit this sample. Bayesian tuning added almost nothing (0.811). The useful jump was the **target**, not a deeper network.

**What the data says to do (and not do):**

- **Likes** are the strongest tabular signal — they track popularity, they are not a pre-publish lever. Do not treat them as a thumbnail tip.
- **Sports** as a category, and **Australia** as a market, associate with higher predicted views in this sample (audience size / engagement mix — not “always film sports”).
- **Thumbnails:** a close, readable face helps personality-led or reaction content. If there is no face, the alternative that shows up is **bold text or one hard focal object** — not a busier collage.
- Image-level SHAP on pixels was **unstable**. Thumbnail advice is directional, not “move this pixel.”

---

## How we got there (short)

| Step | Why it mattered | Val R² (log views) |
|---|---|---|
| Tabular features only | Timing, category, region, title symbols | 0.595 |
| + thumbnail CNN | Package, not metadata alone | 0.638 |
| + face / text on the image | What the viewer actually sees | 0.656 |
| Predict log views | Raw counts blew up; this is the decision-useful scale | 0.794 |
| Smaller CNN (Model II) | Less overfit on 9.9k rows | **0.809** |
| Bayesian search | Costly; little extra | 0.811 |

Limits that change how you should use it: scrape volume was **skewed by region** (Canada over-weighted); API cap is 10k units/day; **likes leak post-publish information**, so a true pre-publish tool would need a likes-free variant.

---

## What’s in the repo

| Path | |
|---|---|
| [`docs/presentation.pdf`](docs/presentation.pdf) | Decision deck |
| [`docs/processing-map.pdf`](docs/processing-map.pdf) | Collection → train set |
| `notebooks/scrape_youtube_api.ipynb` | API pull (`YOUTUBE_API_KEY` in the environment) |
| `notebooks/download_thumbnails.ipynb` | Image download |
| `notebooks/modeling/` | Multi-input experiments |

Thumbnails and the full CSV are not checked in. Point the notebooks at a local data folder.

---

## Setup

```sh
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## License

MIT. See [LICENSE](LICENSE).

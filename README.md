# P4AI-DS EDA Dashboard

Exploratory Data Analysis dashboard for **Assignment 1 — P4AI-DS (CO3135)**. Covers three data modalities — tabular, text, and image — each with a dedicated interactive report built as a static web app.

**Live demo:** https://khenm.github.io/p4ai-ds/

---

## Datasets

| Modality | Dataset | Source |
|---|---|---|
| Tabular | Job Salary Prediction Dataset | [Kaggle](https://www.kaggle.com/datasets/nalisha/job-salary-prediction-dataset) |
| Text | News Category Dataset v3 | [Kaggle](https://www.kaggle.com/datasets/rmisra/news-category-dataset) |
| Image | PetFinder Adoption Prediction | [Kaggle](https://www.kaggle.com/competitions/petfinder-adoption-prediction/data) |

---

## Quick Start

**Requirements:** Python 3.10+, [`uv`](https://docs.astral.sh/uv/)

```bash
# Install dependencies
uv sync

# Run full pipeline and launch dashboard (default port 8081)
./run.sh
```

Open `http://localhost:8081` in your browser.

### Run individual stages

```bash
uv run python scripts/eda_text.py       # Text EDA  → ui/assets/data/
uv run python scripts/eda_salary.py     # Salary EDA → ui/assets/data/jobsalary/
uv run python scripts/eda_image.py      # PetFinder EDA (4 phases) → ui/assets/data/ + ui/assets/samples/
uv run python scripts/gallery_export.py # Breed gallery → ui/assets/data/image_gallery.json

# Start web server only (after assets are generated)
cd ui && python3 -m http.server 8081
```

---

## Project Structure

```
petfinder-analysis/
├── configs/
│   ├── eda.yaml               # EDA parameters (seed, top_labels_k)
│   ├── style.yaml             # Color palette config
│   └── datasets/              # Dataset-specific metadata
├── data/                      # Raw datasets (not tracked in git)
│   ├── petfinder/             # train.csv, train_images/, BreedLabels.csv, ColorLabels.csv
│   ├── newscategory/          # News_Category_Dataset_v3.json
│   └── jobsalary/             # job_salary_prediction_dataset.csv
├── scripts/
│   ├── eda_image.py           # PetFinder EDA entry point (4 phases)
│   ├── eda_text.py            # News Category EDA entry point
│   ├── eda_salary.py          # Salary EDA entry point
│   ├── gallery_export.py      # Breed gallery image export
│   └── preprocess.py          # Image preprocessing & COCO annotations
├── src/
│   ├── preprocess/            # Image preprocessing modules
│   └── eda/
│       ├── tabular_context.py # PetFinder tabular analysis (Phase 1)
│       ├── image_metadata.py  # Image dimensions & resolution (Phase 2)
│       ├── image_quality.py   # Quality metrics on 3K-image sample (Phase 3)
│       ├── image_advanced.py  # PCA, t-SNE, dominant colors (Phase 4)
│       ├── text_context.py    # News Category text analysis
│       ├── salary_eda.py      # Salary EDA analysis
│       └── theme.py           # Shared pastel color palette
├── ui/                        # Static web dashboard (vanilla HTML/CSS/JS)
│   ├── index.html             # Landing page
│   ├── salary_dashboard.html  # Salary EDA report
│   ├── salary_dashboard.js    # Salary chart rendering
│   ├── image.html             # PetFinder EDA report
│   ├── image.js               # PetFinder chart rendering & gallery
│   ├── text.html              # News Category report
│   ├── text.js                # News Category chart rendering
│   ├── style.css              # Warm pastel theme
│   ├── script.js              # Shared utilities & Plotly config
│   └── assets/                # Generated JSON + sample images (not tracked)
├── run.sh                     # One-command pipeline + server
├── pyproject.toml
└── uv.lock
```

---

## Dashboard Reports

### Tabular — Job Salary Prediction Dataset
- Dataset schema, missing values, and feature validation
- Multivariate interactions between Education Level and Company Size
- Category frequency tracking and Class Imbalance mapping for Job Titles
- Cramér's V categorical redundancy and Pearson correlation matrices
- Actionable insights for Machine Learning feature engineering

### Text — News Category
- Dataset overview, schema, and sample records
- Category imbalance and smallest class analysis
- Headline and combined text length distributions
- Timeline drift across publication years (2012–2022)
- Missing values, duplicates, top terms, and top authors

### Image — PetFinder Adoption Prediction
- Dataset overview: 14,993 listings, 58,313 images, 5-class `AdoptionSpeed` target
- Feature distributions, correlation heatmap, health and vaccination patterns
- Image dimensions, resolution, photo count impact on adoption speed
- Quality metrics: brightness, contrast, blur, saturation, colorfulness (3K-image sample)
- Composite quality scores and interaction heatmap
- Interactive breed gallery with Dog/Cat tabs
- Dominant color palettes, PCA variance, t-SNE projections, cross-modality analysis

---

## Dependencies

Key packages: `pandas`, `numpy`, `scikit-learn`, `opencv-python`, `Pillow`, `torch`, `torchvision`, `matplotlib`, `seaborn`.
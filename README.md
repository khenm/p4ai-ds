# P4AI-DS Project

This repository contains the coursework project for **P4AI-DS / CO3135**, covering exploratory data analysis, classical machine learning, deep learning, and interactive reporting across three independent datasets:

- **Tabular:** Job salary prediction and regression modeling
- **Text:** News category exploration and multi-class classification
- **Image:** PetFinder adoption prediction, image-quality analysis, CNN/ML baselines, Grad-CAM, and visual galleries

**Dashboard:** https://khenm.github.io/p4ai-ds/

## Datasets

| Modality | Dataset | Source |
|---|---|---|
| Tabular | Job Salary Prediction Dataset | [Kaggle](https://www.kaggle.com/datasets/nalisha/job-salary-prediction-dataset) |
| Text | News Category Dataset v3 | [Kaggle](https://www.kaggle.com/datasets/rmisra/news-category-dataset) |
| Image | PetFinder Adoption Prediction | [Kaggle](https://www.kaggle.com/competitions/petfinder-adoption-prediction/data) |

---

## Quick Start

Requirements:

- Python 3.10+
- [`uv`](https://docs.astral.sh/uv/)
- Dataset files downloaded separately from Kaggle

Install dependencies:

```bash
uv sync
```

Run the EDA exports and launch the dashboard:

```bash
./run.sh
```

The default server is available at:

```text
http://localhost:8081
```

Use another port by passing it as the first argument:

```bash
./run.sh 9000
```

If the dashboard assets already exist, serve the UI only:

```bash
cd ui
python3 -m http.server 8081
```

## Main Workflows

### EDA Exports

```bash
uv run python scripts/eda_text.py
uv run python scripts/eda_salary.py
uv run python scripts/eda_image.py
uv run python scripts/gallery_export.py
```

These commands generate dashboard-ready JSON, CSV, image samples, and figures under `ui/assets/`.

### Salary Regression

```bash
uv run python scripts/run_salary_ml.py
```

Outputs are written to:

```text
ui/assets/data/jobsalary_ml/
```

Current checked-in benchmark:

| Model | MAE | RMSE | R2 |
|---|---:|---:|---:|
| XGBoost | 4,651.68 | 5,818.09 | 0.9756 |
| LightGBM | 4,683.65 | 5,858.75 | 0.9753 |
| Random Forest | 5,246.05 | 6,587.19 | 0.9688 |
| Linear Regression | 6,487.02 | 8,361.01 | 0.9497 |

### News Category Text Classification

Traditional ML:

```bash
uv run python scripts/text_classification/train_traditional_ml.py
```

Reduced-dimension pipeline grid:

```bash
uv run python scripts/text_classification/train_pipeline_grid.py
```

Transformer fine-tuning:

```bash
uv run python scripts/text_classification/train_bert.py
```

Useful quick-run options:

```bash
uv run python scripts/text_classification/train_traditional_ml.py --sample-size 5000 --n-jobs -1
uv run python scripts/text_classification/train_pipeline_grid.py --sample-size 5000 --limit 10
uv run python scripts/text_classification/train_bert.py --sample-size 5000 --epochs 1 --limit 2
```

Current checked-in text benchmark:

| Model | Family | Accuracy | Macro F1 | Weighted F1 |
|---|---|---:|---:|---:|
| BERT base, mean pooling | Transformer | 0.6633 | 0.6005 | 0.6728 |
| BERT base, CLS pooling | Transformer | 0.6614 | 0.5988 | 0.6710 |
| BERT base, pooler | Transformer | 0.6597 | 0.5959 | 0.6693 |
| DistilBERT, CLS/pooler | Transformer | 0.6517 | 0.5880 | 0.6622 |
| Logistic regression | Traditional ML | 0.6058 | 0.4981 | 0.6063 |

### PetFinder Image and Adoption Modeling

Train the two-stage ResNet-18 CNN:

```bash
uv run python scripts/train_cnn.py --config configs/train_cnn.yaml
```

Run Grad-CAM from a saved checkpoint:

```bash
uv run python scripts/train_cnn.py \
  --config configs/train_cnn.yaml \
  --checkpoint results/checkpoints/train_cnn/stage2.pt
```

Train classical ML on ResNet image embeddings:

```bash
uv run python scripts/train_ml.py --model lightgbm
uv run python scripts/train_ml.py --model xgboost
uv run python scripts/train_ml.py --model catboost
uv run python scripts/train_ml.py --model decision_tree
```

Current checked-in PetFinder benchmarks:

| Model | Adoption Accuracy | Quadratic Weighted Kappa |
|---|---:|---:|
| LightGBM two-stage embedding model | 0.3630 | 0.2616 |
| XGBoost two-stage embedding model | 0.3555 | 0.2108 |

Selected Stage-1 validation accuracies from the LightGBM run:

| Attribute | Accuracy |
|---|---:|
| Type | 0.9652 |
| Health | 0.9655 |
| Sterilized | 0.6950 |
| MaturitySize | 0.6878 |
| Color1 | 0.6445 |
| FurLength | 0.6380 |
| Breed1 | 0.3115 |

## Project Structure

```text
.
├── configs/
│   ├── datasets/              # Dataset path and metadata configuration
│   ├── models/                # Model-specific configuration
│   ├── eda.yaml               # EDA parameters
│   ├── train_cnn.yaml         # Two-stage ResNet training config
│   └── train_lgbm.yaml        # PetFinder ML baseline config
├── notebooks/                 # Exploratory notebooks
├── results/
│   ├── reports/               # PetFinder model reports
│   ├── gradcam/               # Grad-CAM overlays
│   └── text_classification/   # Text metrics, artifacts, and comparisons
├── scripts/
│   ├── eda_*.py               # EDA export entry points
│   ├── train_*.py             # PetFinder training entry points
│   ├── run_salary_ml.py       # Salary ML pipeline
│   └── text_classification/   # Text training entry points
├── src/
│   ├── analysis/              # Ablation, SHAP, Grad-CAM
│   ├── datasets/              # PetFinder dataset loader
│   ├── eda/                   # EDA modules for all datasets
│   ├── models/                # CNN, ML classifiers, salary models
│   ├── preprocess/            # Image and tabular preprocessing
│   ├── text_classification/   # Traditional ML, pipeline-grid, transformer code
│   └── utils/                 # Training, distributed, reporting, checkpoint utilities
├── ui/                        # Static dashboards and generated assets
├── pyproject.toml
├── uv.lock
└── run.sh
```

## Dashboard Pages

| Page | Purpose |
|---|---|
| `ui/index.html` | Dashboard landing page |
| `ui/salary_dashboard.html` | Salary EDA |
| `ui/salary_ml_dashboard.html` | Salary regression results |
| `ui/text.html` | News Category EDA |
| `ui/text_results.html` | Text classification results |
| `ui/image.html` | PetFinder EDA |
| `ui/image_results.html` | PetFinder image-model results and Grad-CAM |

## Outputs

The codebase writes analysis artifacts to predictable locations:

| Output | Location |
|---|---|
| Static dashboard data | `ui/assets/data/` |
| Dashboard figures and image samples | `ui/assets/figures/`, `ui/assets/samples/` |
| Salary ML dashboard data | `ui/assets/data/jobsalary_ml/` |
| PetFinder model reports | `results/reports/` |
| Grad-CAM overlays | `results/gradcam/`, `ui/assets/gradcam/` |
| Text classification reports | `results/text_classification/` |
| Serialized text models | `results/text_classification/artifacts/` |

## Development Notes

- The dashboards are static and do not require a backend once assets are generated.
- Model artifacts and generated dashboard data can be large; keep raw Kaggle data outside git.
- Transformer training requires downloading Hugging Face checkpoints and benefits strongly from a GPU.
- PetFinder CNN training uses the first image per listing (`<PetID>-1.jpg`) in `PetFinderDataset`.
- Distributed CNN training can be enabled through `configs/train_cnn.yaml` and launched with the appropriate PyTorch distributed runner.

## License and Data Terms

This repository is for academic coursework. Dataset usage is governed by the terms of the respective Kaggle datasets and competitions.

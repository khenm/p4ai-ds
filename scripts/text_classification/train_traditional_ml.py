import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Train text classifiers for News Category")
    parser.add_argument("--test-size", type=float, default=0.2, help="Fraction for the hold-out test split")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--sample-size", type=int, default=None, help="Optional stratified sample size for quick experiments")
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=1,
        help="Number of parallel model-training processes. Use -1 for all available CPU cores.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    logger.info("=" * 60)
    logger.info("News Category Text Classification — Training Pipeline")
    logger.info("=" * 60)

    from src.text_classification.traditional_ml import run_text_classification

    result = run_text_classification(
        random_state=args.seed,
        test_size=args.test_size,
        sample_size=args.sample_size,
        n_jobs=args.n_jobs,
    )

    logger.info("Best model: %s", result["best_model_name"])
    logger.info("Metrics: %s", result["metrics"])
    logger.info("Artifact saved to: %s", result["artifact_path"])
    logger.info("JSON analysis saved to: results/text_classification/traditional_ml/ and ui/assets/data/text_classification/traditional_ml/")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.text_classification.pipeline_grid import run_text_pipeline_grid


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run reduced-dimension text classification pipelines as feature extraction x dimensionality reduction x classifier."
    )
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sample-size", type=int, default=None)
    parser.add_argument("--cpu-jobs", type=int, default=10)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit for quick debugging. Default runs the full pipeline grid.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    result = run_text_pipeline_grid(
        test_size=args.test_size,
        sample_size=args.sample_size,
        random_state=args.seed,
        cpu_jobs=args.cpu_jobs,
        limit=args.limit,
    )
    logger.info("Best pipeline: %s", result["best_pipeline"])
    logger.info("Comparison saved to: %s", result["comparison_path"])
    logger.info("Overview saved to: %s", result["overview_path"])
    logger.info("Best fitted pipeline saved to: %s", result["artifact_path"])


if __name__ == "__main__":
    main()

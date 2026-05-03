import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.text_classification.bert import _split_csv, run_transformer_text_pipeline_grid


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Train transformer text classification pipelines.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--val-size", type=float, default=0.1)
    parser.add_argument("--sample-size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument(
        "--model-names",
        type=str,
        default="distilbert-base-uncased,bert-base-uncased",
        help="Comma-separated encoder checkpoints, e.g. distilbert-base-uncased,bert-base-uncased",
    )
    parser.add_argument("--poolings", type=str, default="cls,mean,pooler", help="Comma-separated choices from cls,mean,pooler")
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    result = run_transformer_text_pipeline_grid(
        random_state=args.seed,
        test_size=args.test_size,
        val_size=args.val_size,
        sample_size=args.sample_size,
        model_names=_split_csv(args.model_names),
        poolings=_split_csv(args.poolings),
        freeze_modes=[False],
        max_length=args.max_length,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        dropout=args.dropout,
        limit=args.limit,
    )
    logger.info("Best transformer pipeline: %s", result["best_pipeline"])


if __name__ == "__main__":
    main()

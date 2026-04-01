import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    logger.info("=" * 60)
    logger.info("News Category Text EDA — Data Extraction Pipeline")
    logger.info("=" * 60)

    from src.eda.text_context import run_text_eda

    run_text_eda()

    logger.info("=" * 60)
    logger.info("Text EDA extraction complete!")
    logger.info("JSON files saved to: ui/assets/data/")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()

import logging
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src" / "eda"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    logger.info("=" * 60)
    logger.info("Global Air Pollution EDA — Data Export Pipeline")
    logger.info("=" * 60)

    from src.eda.air_pollution_eda import run_air_pollution_eda

    result = run_air_pollution_eda()
    logger.info("Records analyzed: %s", f"{result['records']:,}")
    logger.info("Countries covered: %s", result['countries'])
    logger.info("Cities covered: %s", f"{result['cities']:,}")
    logger.info("=" * 60)
    logger.info("Air pollution EDA export complete! JSON saved under ui/assets/data/air_pollution/")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()

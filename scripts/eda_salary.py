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
    logger.info("Salary EDA — Data Export Pipeline")
    logger.info("=" * 60)

    from src.eda.salary_eda import run_salary_eda

    run_salary_eda()
    
    logger.info("=" * 60)
    logger.info("Salary EDA export complete! JSON saved under ui/assets/data/jobsalary/")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()

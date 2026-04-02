import sys
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src" / "eda"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    logger.info("=" * 60)
    logger.info("PetFinder EDA — Full Data Extraction Pipeline")
    logger.info("=" * 60)

    from src.eda.tabular_context import run_tabular_eda
    from src.eda.image_metadata import run_metadata_eda
    from src.eda.image_quality import run_quality_eda
    from src.eda.image_advanced import run_advanced_eda, run_breed_cluster_eda

    logger.info("\n>>> Phase 1/5: Tabular EDA")
    run_tabular_eda()

    logger.info("\n>>> Phase 2/5: Image Metadata")
    run_metadata_eda()

    logger.info("\n>>> Phase 3/5: Image Quality Metrics")
    run_quality_eda()

    logger.info("\n>>> Phase 4/5: Advanced Image Analysis (PCA, t-SNE, Colors)")
    run_advanced_eda()

    logger.info("\n>>> Phase 5/5: Breed Image Similarity & Clustering")
    run_breed_cluster_eda()

    logger.info("=" * 60)
    logger.info("All EDA data extraction complete!")
    logger.info("JSON files saved to: ui/assets/data/")
    logger.info("Sample images saved to: ui/assets/samples/")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()

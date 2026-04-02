import argparse
import sys
import yaml
import logging
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.preprocess.image import process_split

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def run_preprocess_images(config_path="configs/eda.yaml"):
    """Run data preprocessing based on config."""
    config_file = Path(config_path)
    if not config_file.exists():
        logger.error(f"Configuration file {config_file} not found.")
        return

    with open(config_file, "r") as f:
        config = yaml.safe_load(f)

    dataset_name = config["data"]
    dataset_config_path = Path("configs/datasets") / f"{dataset_name}.yaml"
    with open(dataset_config_path, "r") as f:
        dataset_config = yaml.safe_load(f)

    raw_dir = Path(dataset_config["raw_dir"])
    images_dir = Path(dataset_config["images_dir"])
    top_k = config["eda"]["top_labels_k"]

    images_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Starting preprocessing for train split...")
    process_split("train", raw_dir, images_dir, top_k)
    
    logger.info("Starting preprocessing for test split...")
    process_split("test", raw_dir, images_dir, top_k)


def main():
    parser = argparse.ArgumentParser(description="Preprocess Petfinder data.")
    parser.add_argument(
        "--modal", 
        type=str, 
        choices=["images", "texts", "tabular"], 
        required=True,
        help="Which modality to preprocess"
    )
    args = parser.parse_args()

    if args.modal == "images":
        run_preprocess_images()
    elif args.modal == "texts":
        logger.info("Text preprocessing not implemented yet.")
    elif args.modal == "tabular":
        logger.info("Tabular preprocessing not implemented yet.")


if __name__ == "__main__":
    main()

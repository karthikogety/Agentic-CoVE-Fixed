
from pathlib import Path
from typing import Dict
from src.utils.paths import RAW_DIR, PROCESSED_DIR

def get_paths(dataset: str) -> Dict[str, Path]:
    name = dataset.lower()
    raw_dir = RAW_DIR / name
    processed_dir = PROCESSED_DIR / name
    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)
    return {
        "raw_dir": raw_dir, "processed_dir": processed_dir,
        "raw": raw_dir, "processed": processed_dir
    }

get_dataset_paths = get_paths
__all__ = ["get_paths", "get_dataset_paths"]

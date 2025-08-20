# src/AutomaticNumberPlateDetection/entity/__init__.py
from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    dataset_name: str
    local_data_file: Path
    unzip_dir: Path

@dataclass(frozen=True)
class DataPreprocessingConfig:
    root_dir: Path
    raw_data_dir: Path
    processed_images_dir: Path
    processed_labels_dir: Path
    train_split: float
    val_split: float
    img_format: list

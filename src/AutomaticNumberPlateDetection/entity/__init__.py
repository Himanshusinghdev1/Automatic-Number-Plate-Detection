# src/AutomaticNumberPlateDetection/entity/__init__.py
from dataclasses import dataclass
from pathlib import Path
from typing import List

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

@dataclass(frozen=True)
class YOLOv5SetupConfig:
    root_dir: Path
    repository_url: str
    data_yaml_path: Path
    weights_dir: Path

@dataclass(frozen=True)
class ModelTrainingConfig:
    root_dir: Path
    yolov5_dir: Path
    model_name: str
    base_weights: str
    img_size: int
    batch_size: int
    epochs: int
    workers: int
    device: str
    patience: int = 10
    min_delta: float = 0.001

@dataclass(frozen=True)
class ModelTestingConfig:
    root_dir: Path
    trained_model_path: Path
    test_images_dir: Path
    results_dir: Path
    confidence_threshold: float
    iou_threshold: float

@dataclass(frozen=True)
class OCRIntegrationConfig:
    root_dir: Path
    model_path: Path
    test_images_dir: Path
    output_dir: Path
    languages: List[str]
    detection_confidence: float
    ocr_confidence: float
    max_width: int
    output_format: str

@dataclass(frozen=True)
class VideoInferenceConfig:
    model_path: Path
    video_path: Path
    output_dir: Path
    languages: List[str]
    detection_confidence: float
    ocr_confidence: float
    process_every_n_frames: int
    save_video: bool
    max_video_length: int
    batch_size: int = 1
    device: str = ""
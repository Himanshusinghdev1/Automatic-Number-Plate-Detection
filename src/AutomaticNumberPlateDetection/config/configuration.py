from AutomaticNumberPlateDetection.constants import *
from AutomaticNumberPlateDetection.utils.common import read_yaml, create_directories
from AutomaticNumberPlateDetection.entity import (
    DataIngestionConfig, 
    DataPreprocessingConfig,
    YOLOv5SetupConfig
)
from pathlib import Path

class ConfigurationManager:
    def __init__(
        self,
        config_filepath = CONFIG_FILE_PATH,
        params_filepath = PARAMS_FILE_PATH):

        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)

        # Fix: Use dictionary key access
        create_directories([self.config['artifacts_root']])

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        # Fix: Use dictionary key access
        config = self.config['data_ingestion']

        create_directories([config['root_dir']])

        data_ingestion_config = DataIngestionConfig(
            root_dir=Path(config['root_dir']),
            dataset_name=config['dataset_name'],
            local_data_file=Path(config['local_data_file']),
            unzip_dir=Path(config['unzip_dir'])
        )

        return data_ingestion_config
    
    def get_data_preprocessing_config(self) -> DataPreprocessingConfig:
        # Fix: Use dictionary key access
        config = self.config['data_preprocessing']

        create_directories([
            config['root_dir'],
            config['processed_images_dir'],
            config['processed_labels_dir']
        ])

        data_preprocessing_config = DataPreprocessingConfig(
            root_dir=Path(config['root_dir']),
            raw_data_dir=Path(config['raw_data_dir']),
            processed_images_dir=Path(config['processed_images_dir']),
            processed_labels_dir=Path(config['processed_labels_dir']),
            train_split=config['train_split'],
            val_split=config['val_split'],
            img_format=config['img_format']
        )

        return data_preprocessing_config

    def get_yolov5_setup_config(self) -> YOLOv5SetupConfig:
        # Fix: Use dictionary key access
        config = self.config['yolov5_setup']

        create_directories([config['weights_dir']])

        yolov5_setup_config = YOLOv5SetupConfig(
            root_dir=Path(config['root_dir']),
            repository_url=config['repository_url'],
            data_yaml_path=Path(config['data_yaml_path']),
            weights_dir=Path(config['weights_dir'])
        )

        return yolov5_setup_config

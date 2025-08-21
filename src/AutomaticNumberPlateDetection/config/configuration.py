from AutomaticNumberPlateDetection.constants import *
from AutomaticNumberPlateDetection.utils.common import read_yaml, create_directories
from AutomaticNumberPlateDetection.entity import (
    DataIngestionConfig, 
    DataPreprocessingConfig,
    YOLOv5SetupConfig,
    ModelTrainingConfig,
    ModelTestingConfig
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

    def get_model_training_config(self) -> ModelTrainingConfig:
        config = self.config['model_training']

        create_directories([config['root_dir']])

        model_training_config = ModelTrainingConfig(
            root_dir=Path(config['root_dir']),
            yolov5_dir=Path(config['yolov5_dir']),
            model_name=config['model_name'],
            base_weights=config['base_weights'],
            img_size=config['img_size'],
            batch_size=config['batch_size'],
            epochs=config['epochs'],
            workers=config['workers'],
            device=config['device']
        )

        return model_training_config

    def get_model_testing_config(self) -> ModelTestingConfig:
        config = self.config['model_testing']

        create_directories([
            config['results_dir'],
            config['test_images_dir']
        ])

        model_testing_config = ModelTestingConfig(
            root_dir=Path(config['root_dir']),
            trained_model_path=Path(config['trained_model_path']),
            test_images_dir=Path(config['test_images_dir']),
            results_dir=Path(config['results_dir']),
            confidence_threshold=config['confidence_threshold'],
            iou_threshold=config['iou_threshold']
        )

        return model_testing_config
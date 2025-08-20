# src/AutomaticNumberPlateDetection/pipeline/stage_02_data_preprocessing.py
from AutomaticNumberPlateDetection.config.configuration import ConfigurationManager
from AutomaticNumberPlateDetection.components.data_preprocessing import DataPreprocessing
from AutomaticNumberPlateDetection import logger

STAGE_NAME = "Data Preprocessing stage"

class DataPreprocessingTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        data_preprocessing_config = config.get_data_preprocessing_config()
        data_preprocessing = DataPreprocessing(config=data_preprocessing_config)
        data_preprocessing.split_and_organize_data()

if __name__ == '__main__':
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = DataPreprocessingTrainingPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e

# main.py
from AutomaticNumberPlateDetection import logger
from AutomaticNumberPlateDetection.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline
from AutomaticNumberPlateDetection.pipeline.stage_02_data_preprocessing import DataPreprocessingTrainingPipeline
from AutomaticNumberPlateDetection.pipeline.stage_03_yolov5_setup import YOLOv5SetupTrainingPipeline
from AutomaticNumberPlateDetection.pipeline.stage_04_model_training import ModelTrainingTrainingPipeline
from AutomaticNumberPlateDetection.pipeline.stage_05_model_testing import ModelTestingTrainingPipeline

STAGE_NAME = "Data Ingestion stage"
try:
   logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<") 
   data_ingestion = DataIngestionTrainingPipeline()
   data_ingestion.main()
   logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
        logger.exception(e)
        raise e

STAGE_NAME = "Data Preprocessing stage"
try:
   logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<") 
   data_preprocessing = DataPreprocessingTrainingPipeline()
   data_preprocessing.main()
   logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
        logger.exception(e)
        raise e

STAGE_NAME = "YOLOv5 Setup stage"
try:
   logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<") 
   yolov5_setup = YOLOv5SetupTrainingPipeline()
   yolov5_setup.main()
   logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
        logger.exception(e)
        raise e


STAGE_NAME = "Model Training stage"
try:
   logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<") 
   model_training = ModelTrainingTrainingPipeline()
   model_training.main()
   logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
        logger.exception(e)
        raise e

STAGE_NAME = "Model Testing stage"
try:
   logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<") 
   model_testing = ModelTestingTrainingPipeline()
   model_testing.main()
   logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
        logger.exception(e)
        raise e

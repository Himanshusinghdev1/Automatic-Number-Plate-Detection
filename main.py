# main.py
from AutomaticNumberPlateDetection import logger
from AutomaticNumberPlateDetection.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline
from AutomaticNumberPlateDetection.pipeline.stage_02_data_preprocessing import DataPreprocessingTrainingPipeline
from AutomaticNumberPlateDetection.pipeline.stage_03_yolov5_setup import YOLOv5SetupTrainingPipeline

# Stage 01 - Data Ingestion
STAGE_NAME = "Data Ingestion stage"
try:
   logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<") 
   data_ingestion = DataIngestionTrainingPipeline()
   data_ingestion.main()
   logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
        logger.exception(e)
        raise e

# Stage 02 - Data Preprocessing
STAGE_NAME = "Data Preprocessing stage"
try:
   logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<") 
   data_preprocessing = DataPreprocessingTrainingPipeline()
   data_preprocessing.main()
   logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
        logger.exception(e)
        raise e

# Stage 03 - YOLOv5 Setup
STAGE_NAME = "YOLOv5 Setup stage"
try:
   logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<") 
   yolov5_setup = YOLOv5SetupTrainingPipeline()
   yolov5_setup.main()
   logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
        logger.exception(e)
        raise e

# src/AutomaticNumberPlateDetection/pipeline/stage_03_yolov5_setup.py
from AutomaticNumberPlateDetection.config.configuration import ConfigurationManager
from AutomaticNumberPlateDetection.components.yolov5_setup import YOLOv5Setup
from AutomaticNumberPlateDetection import logger

STAGE_NAME = "YOLOv5 Setup stage"

class YOLOv5SetupTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        yolov5_setup_config = config.get_yolov5_setup_config()
        yolov5_setup = YOLOv5Setup(config=yolov5_setup_config)
        yolov5_setup.setup_yolov5()

if __name__ == '__main__':
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = YOLOv5SetupTrainingPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e

# src/AutomaticNumberPlateDetection/pipeline/stage_05_model_testing.py
from AutomaticNumberPlateDetection.config.configuration import ConfigurationManager
from AutomaticNumberPlateDetection.components.model_testing import ModelTesting
from AutomaticNumberPlateDetection import logger

STAGE_NAME = "Model Testing stage"

class ModelTestingTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        model_testing_config = config.get_model_testing_config()
        model_testing = ModelTesting(config=model_testing_config)
        
        # Test batch of images
        model_testing.test_batch_images()
        
        # Benchmark model performance
        model_testing.benchmark_model()

if __name__ == '__main__':
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = ModelTestingTrainingPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e

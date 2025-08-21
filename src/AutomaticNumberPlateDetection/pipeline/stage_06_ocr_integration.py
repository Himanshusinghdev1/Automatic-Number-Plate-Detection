# src/AutomaticNumberPlateDetection/pipeline/stage_06_ocr_integration.py
from AutomaticNumberPlateDetection.config.configuration import ConfigurationManager
from AutomaticNumberPlateDetection.components.ocr_integration import OCRIntegration
from AutomaticNumberPlateDetection import logger

STAGE_NAME = "OCR Integration stage"

class OCRIntegrationPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        ocr_integration_config = config.get_ocr_integration_config()
        ocr_integration = OCRIntegration(config=ocr_integration_config)
        
        # Process all test images
        results = ocr_integration.process_batch_images()
        
        return results

if __name__ == '__main__':
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = OCRIntegrationPipeline()
        results = obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e

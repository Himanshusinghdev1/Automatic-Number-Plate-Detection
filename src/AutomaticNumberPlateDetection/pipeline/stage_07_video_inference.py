from AutomaticNumberPlateDetection.config.configuration import ConfigurationManager
from AutomaticNumberPlateDetection.components.video_anpr import VideoANPR
from AutomaticNumberPlateDetection import logger

STAGE_NAME = "Video Inference stage"

class VideoInferencePipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        video_inference_config = config.get_video_inference_config()
        video_anpr = VideoANPR(config=video_inference_config)
        
        # Process video with ANPR
        results = video_anpr.process_video()
        
        return results

if __name__ == '__main__':
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = VideoInferencePipeline()
        results = obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e

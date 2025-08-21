# src/AutomaticNumberPlateDetection/components/model_testing.py
import torch
import cv2
import numpy as np
from pathlib import Path
from AutomaticNumberPlateDetection import logger
from AutomaticNumberPlateDetection.entity import ModelTestingConfig

class ModelTesting:
    def __init__(self, config: ModelTestingConfig):
        self.config = config
        self.model = None

    def load_trained_model(self):
        """Load the trained YOLOv5 model"""
        try:
            if not self.config.trained_model_path.exists():
                raise FileNotFoundError(f"Trained model not found: {self.config.trained_model_path}")
            
            logger.info(f"🔮 Loading trained model from: {self.config.trained_model_path}")
            
            # Load custom trained model
            self.model = torch.hub.load(
                'ultralytics/yolov5', 
                'custom', 
                path=str(self.config.trained_model_path)
            )
            
            # Set model parameters
            self.model.conf = self.config.confidence_threshold
            self.model.iou = self.config.iou_threshold
            
            logger.info("✅ Model loaded successfully!")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error loading model: {str(e)}")
            return False

    def test_single_image(self, image_path: str, save_results: bool = True):
        """Test model on a single image"""
        try:
            if self.model is None:
                if not self.load_trained_model():
                    return None
            
            logger.info(f"🖼️ Testing image: {image_path}")
            
            if not Path(image_path).exists():
                logger.error(f"❌ Image not found: {image_path}")
                return None
            
            # Run inference
            results = self.model(image_path)
            
            # Get detection results
            detections = results.pandas().xyxy[0]
            
            logger.info(f"📊 Detections found: {len(detections)}")
            
            if save_results:
                # Create results directory
                self.config.results_dir.mkdir(exist_ok=True)
                
                # Save annotated image
                results.save(str(self.config.results_dir))
                logger.info(f"💾 Results saved to: {self.config.results_dir}")
            
            return results, detections
            
        except Exception as e:
            logger.error(f"❌ Error testing image: {str(e)}")
            return None

    def test_batch_images(self):
        """Test model on all images in test directory"""
        try:
            if not self.config.test_images_dir.exists():
                logger.warning(f"⚠️ Test images directory not found: {self.config.test_images_dir}")
                return []
            
            # Get all image files
            image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
            image_files = []
            
            for ext in image_extensions:
                image_files.extend(self.config.test_images_dir.glob(f"*{ext}"))
                image_files.extend(self.config.test_images_dir.glob(f"*{ext.upper()}"))
            
            if not image_files:
                logger.warning(f"⚠️ No images found in: {self.config.test_images_dir}")
                return []
            
            logger.info(f"📁 Found {len(image_files)} test images")
            
            results_summary = []
            
            for image_file in image_files:
                result = self.test_single_image(str(image_file), save_results=True)
                
                if result:
                    results_obj, detections = result
                    results_summary.append({
                        'image': image_file.name,
                        'detections': len(detections),
                        'max_confidence': detections['confidence'].max() if len(detections) > 0 else 0
                    })
                else:
                    results_summary.append({
                        'image': image_file.name,
                        'detections': 0,
                        'max_confidence': 0
                    })
            
            self._print_test_summary(results_summary)
            return results_summary
            
        except Exception as e:
            logger.error(f"❌ Error in batch testing: {str(e)}")
            return []

    def _print_test_summary(self, results_summary):
        """Print testing results summary"""
        logger.info("\n📋 Testing Summary:")
        logger.info("-" * 50)
        
        total_images = len(results_summary)
        total_detections = sum([r['detections'] for r in results_summary])
        
        for result in results_summary:
            logger.info(f"📷 {result['image']}: {result['detections']} plates detected "
                       f"(conf: {result['max_confidence']:.3f})")
        
        logger.info("-" * 50)
        logger.info(f"📊 Total Images: {total_images}")
        logger.info(f"📊 Total Detections: {total_detections}")
        logger.info(f"📊 Average per Image: {total_detections/total_images:.2f}")

    def benchmark_model(self):
        """Benchmark model performance"""
        try:
            if self.model is None:
                if not self.load_trained_model():
                    return
            
            logger.info("⚡ Running performance benchmark...")
            
            # Create dummy image for benchmarking
            dummy_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
            
            # Warm up
            for _ in range(5):
                _ = self.model(dummy_image)
            
            # Benchmark
            import time
            times = []
            
            for _ in range(20):
                start_time = time.time()
                _ = self.model(dummy_image)
                end_time = time.time()
                times.append(end_time - start_time)
            
            avg_time = np.mean(times)
            fps = 1.0 / avg_time
            
            logger.info(f"⚡ Average inference time: {avg_time*1000:.2f} ms")
            logger.info(f"⚡ FPS: {fps:.2f}")
            
        except Exception as e:
            logger.error(f"❌ Error in benchmarking: {str(e)}")

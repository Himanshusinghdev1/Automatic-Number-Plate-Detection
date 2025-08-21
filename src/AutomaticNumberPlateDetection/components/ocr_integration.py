# src/AutomaticNumberPlateDetection/components/ocr_integration.py
import torch
import easyocr
import cv2
import numpy as np
import json
import re
from pathlib import Path
from PIL import Image
from AutomaticNumberPlateDetection import logger
from AutomaticNumberPlateDetection.entity import OCRIntegrationConfig

class OCRIntegration:
    def __init__(self, config: OCRIntegrationConfig):
        self.config = config
        self.model = None
        self.reader = None

    def load_model(self):
        """Load YOLOv5 detection model"""
        try:
            if not self.config.model_path.exists():
                raise FileNotFoundError(f"Model not found: {self.config.model_path}")
            
            logger.info(f"🔮 Loading YOLOv5 model from: {self.config.model_path}")
            self.model = torch.hub.load('ultralytics/yolov5', 'custom', path=str(self.config.model_path))
            logger.info("✅ YOLOv5 model loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error loading model: {str(e)}")
            return False

    def load_ocr_reader(self):
        """Load EasyOCR reader"""
        try:
            logger.info(f"📖 Loading EasyOCR for languages: {self.config.languages}")
            self.reader = easyocr.Reader(self.config.languages)
            logger.info("✅ EasyOCR reader loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error loading OCR reader: {str(e)}")
            return False

    def preprocess_image(self, image_path):
        """Preprocess image for better detection and OCR"""
        image = cv2.imread(str(image_path))
        if image is None:
            return None
        
        # Resize if too large
        height, width = image.shape[:2]
        if width > self.config.max_width:
            scale = self.config.max_width / width
            new_width = int(width * scale)
            new_height = int(height * scale)
            image = cv2.resize(image, (new_width, new_height))
        
        return image

    def detect_license_plates(self, image_path):
        """Detect license plates in image"""
        try:
            if self.model is None:
                if not self.load_model():
                    return []
            
            logger.info(f"🔍 Detecting license plates in: {image_path}")
            
            # Run detection
            results = self.model(str(image_path))
            
            detections = []
            for *box, conf, cls in results.xyxy[0].cpu().numpy():
                if conf > self.config.detection_confidence:
                    detections.append({
                        'bbox': [int(x) for x in box],
                        'confidence': float(conf)
                    })
            
            logger.info(f"📊 Found {len(detections)} license plates")
            return detections
            
        except Exception as e:
            logger.error(f"❌ Error in detection: {str(e)}")
            return []

    def extract_text_from_plate(self, image, bbox):
        """Extract text from detected license plate region"""
        try:
            if self.reader is None:
                if not self.load_ocr_reader():
                    return ""
            
            # Crop plate region
            x1, y1, x2, y2 = bbox
            plate_crop = image[y1:y2, x1:x2]
            
            if plate_crop.size == 0:
                return ""
            
            # Enhance plate image for better OCR
            plate_crop = self._enhance_plate_image(plate_crop)
            
            # Run OCR
            ocr_results = self.reader.readtext(plate_crop)
            
            # Extract and clean text
            plate_text = self._extract_best_text(ocr_results)
            
            return plate_text
            
        except Exception as e:
            logger.error(f"❌ Error in OCR: {str(e)}")
            return ""

    def _enhance_plate_image(self, plate_crop):
        """Enhance plate image for better OCR accuracy"""
        # Convert to grayscale
        gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
        
        # Apply gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        
        # Increase contrast
        enhanced = cv2.convertScaleAbs(blurred, alpha=1.2, beta=10)
        
        # Apply adaptive threshold
        thresh = cv2.adaptiveThreshold(enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                     cv2.THRESH_BINARY, 11, 2)
        
        return thresh

    def _extract_best_text(self, ocr_results):
        """Extract and clean the best text from OCR results"""
        if not ocr_results:
            return ""
        
        # Filter by confidence and combine text
        texts = []
        for (bbox, text, confidence) in ocr_results:
            if confidence > self.config.ocr_confidence:
                # Clean text - remove special characters, keep only alphanumeric
                cleaned_text = re.sub(r'[^A-Z0-9]', '', text.upper())
                if 2 <= len(cleaned_text) <= 15:  # Reasonable plate length
                    texts.append(cleaned_text)
        
        # Join all text pieces
        return ''.join(texts)

    def process_single_image(self, image_path):
        """Process single image - complete ANPR pipeline"""
        try:
            logger.info(f"🔄 Processing image: {image_path}")
            
            # Preprocess image
            image = self.preprocess_image(image_path)
            if image is None:
                logger.error(f"❌ Could not load image: {image_path}")
                return []
            
            # Detect license plates
            detections = self.detect_license_plates(image_path)
            
            results = []
            for i, detection in enumerate(detections):
                # Extract text from each detected plate
                text = self.extract_text_from_plate(image, detection['bbox'])
                
                result = {
                    'plate_id': i + 1,
                    'bbox': detection['bbox'],
                    'detection_confidence': detection['confidence'],
                    'text': text,
                    'text_confidence': 0.8 if text else 0.0  # Simple confidence estimation
                }
                
                results.append(result)
                logger.info(f"📝 Plate {i+1}: '{text}' (detection: {detection['confidence']:.3f})")
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Error processing image: {str(e)}")
            return []

    def process_batch_images(self):
        """Process all images in test directory"""
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
            
            logger.info(f"📁 Processing {len(image_files)} images")
            
            all_results = []
            
            for image_file in image_files:
                image_results = self.process_single_image(image_file)
                
                all_results.append({
                    'image_name': image_file.name,
                    'image_path': str(image_file),
                    'plates': image_results,
                    'total_plates': len(image_results)
                })
            
            # Save results
            self._save_results(all_results)
            self._print_summary(all_results)
            
            return all_results
            
        except Exception as e:
            logger.error(f"❌ Error in batch processing: {str(e)}")
            return []

    def _save_results(self, results):
        """Save OCR results to file"""
        try:
            self.config.output_dir.mkdir(exist_ok=True)
            
            if self.config.output_format.lower() == 'json':
                output_file = self.config.output_dir / "ocr_results.json"
                with open(output_file, 'w') as f:
                    json.dump(results, f, indent=2)
                logger.info(f"💾 Results saved to: {output_file}")
            
        except Exception as e:
            logger.error(f"❌ Error saving results: {str(e)}")

    def _print_summary(self, results):
        """Print processing summary"""
        logger.info("\n🎯 OCR Integration Summary:")
        logger.info("-" * 50)
        
        total_images = len(results)
        total_plates = sum([r['total_plates'] for r in results])
        successful_reads = sum([len([p for p in r['plates'] if p['text']]) for r in results])
        
        logger.info(f"📊 Total Images Processed: {total_images}")
        logger.info(f"📊 Total Plates Detected: {total_plates}")
        logger.info(f"📊 Successful Text Reads: {successful_reads}")
        logger.info(f"📊 OCR Success Rate: {(successful_reads/total_plates*100):.1f}%" if total_plates > 0 else "📊 OCR Success Rate: 0%")
        
        logger.info("\n📋 Detailed Results:")
        for result in results:
            logger.info(f"📷 {result['image_name']}: {result['total_plates']} plates")
            for plate in result['plates']:
                if plate['text']:
                    logger.info(f"  ✅ Plate: '{plate['text']}' (conf: {plate['detection_confidence']:.3f})")
                else:
                    logger.info(f"  ❌ Plate detected but text not readable")

# src/AutomaticNumberPlateDetection/components/video_anpr.py
import cv2
import torch
import easyocr
import json
import time
import re
from pathlib import Path
from AutomaticNumberPlateDetection import logger
from AutomaticNumberPlateDetection.entity import VideoInferenceConfig

class VideoANPR:
    def __init__(self, config: VideoInferenceConfig):
        self.config = config
        self.model = None
        self.reader = None
        self.results = []

    def load_models(self):
        """Load YOLOv5 detection model and EasyOCR reader"""
        try:
            if not self.config.model_path.exists():
                raise FileNotFoundError(f"Model not found: {self.config.model_path}")
            
            logger.info(f"🎬 Loading YOLOv5 model from: {self.config.model_path}")
            self.model = torch.hub.load('ultralytics/yolov5', 'custom', path=str(self.config.model_path))
            logger.info("✅ YOLOv5 model loaded successfully")
            
            logger.info(f"📖 Loading EasyOCR for languages: {self.config.languages}")
            self.reader = easyocr.Reader(self.config.languages)
            logger.info("✅ EasyOCR reader loaded successfully")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error loading models: {str(e)}")
            return False

    def process_video(self):
        """Process video with license plate detection and OCR"""
        try:
            if not self.load_models():
                return []
            
            if not self.config.video_path.exists():
                logger.error(f"❌ Video file not found: {self.config.video_path}")
                return []
            
            logger.info(f"🎬 Processing video: {self.config.video_path}")
            
            # Open video capture
            cap = cv2.VideoCapture(str(self.config.video_path))
            
            if not cap.isOpened():
                logger.error(f"❌ Cannot open video: {self.config.video_path}")
                return []
            
            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            duration = total_frames / fps
            
            logger.info(f"📊 Video Properties:")
            logger.info(f"   • Frames: {total_frames}")
            logger.info(f"   • FPS: {fps:.2f}")
            logger.info(f"   • Resolution: {width}x{height}")
            logger.info(f"   • Duration: {duration:.1f}s")
            
            # Setup output video writer
            out = None
            if self.config.save_video:
                self.config.output_dir.mkdir(exist_ok=True)
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                output_path = self.config.output_dir / f"anpr_output_{self.config.video_path.stem}.mp4"
                out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
                logger.info(f"📹 Output video will be saved to: {output_path}")
            
            # Processing variables
            frame_count = 0
            processed_frames = 0
            start_time = time.time()
            last_progress_time = start_time
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                current_time = frame_count / fps
                
                # Skip frames for efficiency
                if frame_count % self.config.process_every_n_frames == 0:
                    processed_frames += 1
                    
                    # Detect license plates
                    detections = self._detect_plates(frame)
                    
                    # Process each detection with OCR
                    frame_results = []
                    for i, detection in enumerate(detections):
                        plate_text = self._extract_text(frame, detection['bbox'])
                        
                        result = {
                            'frame_number': frame_count,
                            'timestamp': current_time,
                            'bbox': detection['bbox'],
                            'detection_confidence': detection['confidence'],
                            'plate_text': plate_text,
                            'plate_id': i + 1
                        }
                        
                        frame_results.append(result)
                        
                        if plate_text:  # Only log successful text extractions
                            logger.info(f"📝 Frame {frame_count} ({current_time:.1f}s): '{plate_text}' (conf: {detection['confidence']:.3f})")
                    
                    self.results.extend(frame_results)
                    
                    # Annotate frame for output video
                    if out:
                        annotated_frame = self._annotate_frame(frame, detections)
                        out.write(annotated_frame)
                else:
                    # Write original frame if saving video
                    if out:
                        out.write(frame)
                
                # Progress reporting every 5 seconds
                current_processing_time = time.time()
                if current_processing_time - last_progress_time >= 5.0:
                    progress = (frame_count / total_frames) * 100
                    logger.info(f"⏳ Progress: {progress:.1f}% ({frame_count}/{total_frames} frames)")
                    last_progress_time = current_processing_time
                
                # Stop if max duration reached
                if current_time >= self.config.max_video_length:
                    logger.info(f"⏹️ Stopped processing at {self.config.max_video_length}s limit")
                    break
            
            # Cleanup
            cap.release()
            if out:
                out.release()
            
            # Calculate final statistics
            end_time = time.time()
            processing_duration = end_time - start_time
            avg_fps = processed_frames / processing_duration if processing_duration > 0 else 0
            
            # Count successful plate reads
            successful_reads = len([r for r in self.results if r['plate_text']])
            total_detections = len(self.results)
            
            logger.info(f"✅ Video processing completed!")
            logger.info(f"📊 Processing Statistics:")
            logger.info(f"   • Total frames processed: {processed_frames}")
            logger.info(f"   • Processing time: {processing_duration:.1f}s")
            logger.info(f"   • Average processing FPS: {avg_fps:.1f}")
            logger.info(f"   • Total plate detections: {total_detections}")
            logger.info(f"   • Successful text reads: {successful_reads}")
            logger.info(f"   • OCR success rate: {(successful_reads/total_detections*100):.1f}%" if total_detections > 0 else "   • OCR success rate: 0%")
            
            # Save results
            self._save_results()
            self._print_summary()
            
            return self.results
            
        except Exception as e:
            logger.error(f"❌ Error processing video: {str(e)}")
            return []

    def _detect_plates(self, frame):
        """Detect license plates in frame"""
        results = self.model(frame)
        
        detections = []
        for *box, conf, cls in results.xyxy[0].cpu().numpy():
            if conf > self.config.detection_confidence:
                detections.append({
                    'bbox': [int(x) for x in box],
                    'confidence': float(conf)
                })
        
        return detections

    def _extract_text(self, frame, bbox):
        """Extract text from license plate region"""
        try:
            x1, y1, x2, y2 = bbox
            plate_crop = frame[y1:y2, x1:x2]
            
            if plate_crop.size == 0:
                return ""
            
            # Enhance plate image
            plate_crop = self._enhance_plate_image(plate_crop)
            
            # Run OCR
            ocr_results = self.reader.readtext(plate_crop)
            
            # Extract and clean text
            plate_text = self._extract_best_text(ocr_results)
            
            return plate_text
            
        except Exception as e:
            return ""

    def _enhance_plate_image(self, plate_crop):
        """Enhance plate image for better OCR"""
        # Convert to grayscale
        gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
        
        # Apply gaussian blur
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        
        # Increase contrast
        enhanced = cv2.convertScaleAbs(blurred, alpha=1.2, beta=10)
        
        return enhanced

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

    def _annotate_frame(self, frame, detections):
        """Annotate frame with detections"""
        annotated_frame = frame.copy()
        
        for detection in detections:
            x1, y1, x2, y2 = detection['bbox']
            conf = detection['confidence']
            
            # Draw bounding box
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Add confidence label
            label = f"Plate: {conf:.2f}"
            cv2.putText(annotated_frame, label, (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Add frame number
            cv2.putText(annotated_frame, f"Frame: {len(self.results)}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return annotated_frame

    def _save_results(self):
        """Save video ANPR results"""
        try:
            self.config.output_dir.mkdir(exist_ok=True)
            
            # Save detailed JSON results
            results_file = self.config.output_dir / f"video_anpr_results_{self.config.video_path.stem}.json"
            
            # Prepare results summary
            summary = {
                'video_info': {
                    'filename': self.config.video_path.name,
                    'total_detections': len(self.results),
                    'successful_reads': len([r for r in self.results if r['plate_text']]),
                    'processing_date': time.strftime('%Y-%m-%d %H:%M:%S')
                },
                'detections': self.results
            }
            
            with open(results_file, 'w') as f:
                json.dump(summary, f, indent=2)
            
            logger.info(f"💾 Results saved to: {results_file}")
            
        except Exception as e:
            logger.error(f"❌ Error saving results: {str(e)}")

    def _print_summary(self):
        """Print processing summary"""
        logger.info("\n🎯 Video ANPR Summary:")
        logger.info("-" * 50)
        
        successful_reads = [r for r in self.results if r['plate_text']]
        unique_plates = set([r['plate_text'] for r in successful_reads])
        
        logger.info(f"📊 Total Detections: {len(self.results)}")
        logger.info(f"📊 Successful Reads: {len(successful_reads)}")
        logger.info(f"📊 Unique Plates: {len(unique_plates)}")
        
        if successful_reads:
            logger.info(f"\n📋 Detected Plates:")
            for plate in sorted(unique_plates):
                count = len([r for r in successful_reads if r['plate_text'] == plate])
                logger.info(f"   • {plate}: {count} times")

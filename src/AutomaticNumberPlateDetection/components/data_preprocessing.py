# src/AutomaticNumberPlateDetection/components/data_preprocessing.py
import os
import cv2
import xml.etree.ElementTree as ET
import shutil
from pathlib import Path
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from AutomaticNumberPlateDetection import logger
from AutomaticNumberPlateDetection.entity import DataPreprocessingConfig

class DataPreprocessing:
    def __init__(self, config: DataPreprocessingConfig):
        self.config = config

    def explore_dataset(self):
        """Explore dataset structure - XML files are in images/ directory"""
        raw_path = Path("artifacts/data_ingestion")
        logger.info(f"🔍 Exploring dataset at: {raw_path}")
        
        # Both images AND XML files are in images/ subdirectory
        images_dir = raw_path / "images"
        
        if not images_dir.exists():
            raise Exception(f"Images directory not found: {images_dir}")
        
        # Get images from images/ directory
        image_files = []
        for ext in self.config.img_format:
            image_files.extend(images_dir.glob(f"*{ext}"))
        
        # Get XML files from SAME images/ directory
        annotation_files = list(images_dir.glob("*.xml"))
        
        logger.info(f"📊 Found {len(image_files)} images and {len(annotation_files)} annotations")
        logger.info(f"⚠️ {len(image_files) - len(annotation_files)} images missing annotations")
        
        return image_files, annotation_files

    def xml_to_yolo(self, xml_file, img_width, img_height):
        """Convert XML annotation to YOLO format"""
        tree = ET.parse(xml_file)
        root = tree.getroot()
        
        yolo_annotations = []
        
        for obj in root.findall('object'):
            class_name = obj.find('name').text.lower()
            
            # Check if it's a license plate
            if any(keyword in class_name for keyword in ['plate', 'license', 'number']):
                bbox = obj.find('bndbox')
                xmin = float(bbox.find('xmin').text)
                ymin = float(bbox.find('ymin').text)
                xmax = float(bbox.find('xmax').text)
                ymax = float(bbox.find('ymax').text)
                
                # Convert to YOLO format (normalized center coordinates)
                center_x = (xmin + xmax) / 2.0 / img_width
                center_y = (ymin + ymax) / 2.0 / img_height
                width = (xmax - xmin) / img_width
                height = (ymax - ymin) / img_height
                
                # Class ID 0 for license plate
                yolo_annotations.append(f"0 {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}")
        
        return yolo_annotations

    def process_annotations(self):
        """Process all annotations to YOLO format with proper missing annotation handling"""
        logger.info("🔄 Converting annotations to YOLO format...")
        
        image_files, annotation_files = self.explore_dataset()
        
        processed_data = []
        skipped_count = 0
        
        for img_file in tqdm(image_files, desc="Processing images"):
            # Skip TEST.jpeg from training data
            if img_file.name.upper() == 'TEST.JPEG':
                logger.info(f"⏭️ Skipping test image: {img_file.name}")
                continue
            
            # Find corresponding annotation file in SAME directory
            annotation_file = None
            for ann_file in annotation_files:
                if img_file.stem == ann_file.stem:  # N1.jpeg matches N1.xml
                    annotation_file = ann_file
                    break
            
            if annotation_file is None:
                logger.warning(f"⚠️ No annotation found for {img_file.name}")
                skipped_count += 1
                continue
            
            # Get image dimensions
            image = cv2.imread(str(img_file))
            if image is None:
                logger.warning(f"⚠️ Could not read image: {img_file.name}")
                continue
                
            img_height, img_width = image.shape[:2]
            
            # Convert XML to YOLO format
            if annotation_file.suffix.lower() == '.xml':
                yolo_annotations = self.xml_to_yolo(annotation_file, img_width, img_height)
            else:
                logger.warning(f"⚠️ Unsupported annotation format: {annotation_file.suffix}")
                continue
            
            if yolo_annotations:
                processed_data.append({
                    'image_file': img_file,
                    'annotations': yolo_annotations
                })
        
        logger.info(f"✅ Processed {len(processed_data)} image-annotation pairs")
        logger.info(f"⚠️ Skipped {skipped_count + 1} images (missing annotations + test image)")
        
        return processed_data

    def split_and_organize_data(self):
        """Split data into train/val and organize in YOLO format"""
        logger.info("📊 Splitting and organizing dataset...")
        
        # Process annotations
        processed_data = self.process_annotations()
        
        if not processed_data:
            raise Exception("No valid data found to process")
        
        # Split data (225 valid pairs → ~180 train, ~45 val)
        train_data, val_data = train_test_split(
            processed_data, 
            train_size=self.config.train_split,
            random_state=42,
            shuffle=True
        )
        
        # Create directory structure
        splits = {
            'train': train_data,
            'val': val_data
        }
        
        for split_name, split_data in splits.items():
            # Create directories
            img_dir = Path(self.config.processed_images_dir) / split_name
            label_dir = Path(self.config.processed_labels_dir) / split_name
            
            img_dir.mkdir(parents=True, exist_ok=True)
            label_dir.mkdir(parents=True, exist_ok=True)
            
            # Copy files
            for item in tqdm(split_data, desc=f"Processing {split_name} split"):
                img_file = item['image_file']
                annotations = item['annotations']
                
                # Copy image
                dst_img = img_dir / img_file.name
                shutil.copy2(img_file, dst_img)
                
                # Save annotation
                dst_label = label_dir / f"{img_file.stem}.txt"
                with open(dst_label, 'w') as f:
                    f.write('\n'.join(annotations))
        
        logger.info(f"✅ Dataset organized: {len(train_data)} train, {len(val_data)} val samples")
        
        # Create data.yaml for YOLO
        self.create_data_yaml()

    def create_data_yaml(self):
        """Create data.yaml file for YOLO training with correct relative paths"""
        # Create paths relative to YOLOv5 directory (using ../ prefix)
        data_yaml_content = f"""train: ../artifacts/data_preprocessing/images/train
val: ../artifacts/data_preprocessing/images/val

nc: 1
names: ['license_plate']
"""
        
        # Save in preprocessing directory
        yaml_path = Path(self.config.root_dir) / "data.yaml"
        with open(yaml_path, 'w') as f:
            f.write(data_yaml_content)
        
        # Copy to YOLOv5 directory with correct relative paths
        yolov5_yaml = Path("yolov5") / "data.yaml"
        yolov5_yaml.parent.mkdir(exist_ok=True)
        with open(yolov5_yaml, 'w') as f:
            f.write(data_yaml_content)
        
        logger.info(f"📝 Created data.yaml for YOLOv5 training with relative paths")
        logger.info(f"📝 Created data.yaml at: {yaml_path}")

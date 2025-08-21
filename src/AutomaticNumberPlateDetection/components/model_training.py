# src/AutomaticNumberPlateDetection/components/model_training.py
import subprocess
import shutil
import sys
from pathlib import Path
from AutomaticNumberPlateDetection import logger
from AutomaticNumberPlateDetection.entity import ModelTrainingConfig

class ModelTraining:
    def __init__(self, config: ModelTrainingConfig):
        self.config = config

    def start_model_training(self):
        """Start YOLOv5 model training"""
        try:
            logger.info("🎯 Starting YOLO model training with early stopping...")
            
            # Ensure paths are Path objects
            data_yaml_path = self.config.yolov5_dir / "data.yaml"  # Path object
            
            if not data_yaml_path.exists():  # Now .exists() works on Path object
                raise FileNotFoundError(f"Data YAML not found: {data_yaml_path}")
            
            train_cmd = [
                sys.executable, "train.py",
                "--img", str(self.config.img_size),
                "--batch", str(self.config.batch_size),
                "--epochs", str(self.config.epochs),
                "--data", "data.yaml",  # Just the filename, run from yolov5 dir
                "--weights", self.config.base_weights,
                "--name", self.config.model_name,
                "--cache",
                "--workers", str(self.config.workers),
                "--patience", str(self.config.patience)
            ]
            
            if self.config.device:
                train_cmd.extend(["--device", self.config.device])
            
            logger.info(f"Training command: {' '.join(train_cmd)}")
            logger.info(f"Early stopping enabled: patience={self.config.patience} epochs")
            
            # Execute training from yolov5 directory
            result = subprocess.run(
                train_cmd, 
                cwd=str(self.config.yolov5_dir),
                # capture_output=True, 
                bufsize=1,
                universal_newlines=True,
                text=True
            )
            
            if result.returncode == 0:
                logger.info("✅ Training completed successfully!")
                self._copy_trained_model()
                return True
            else:
                logger.error(f"❌ Training failed: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Error during training: {str(e)}")
            return False

    def _copy_trained_model(self):
        """Copy trained model to models directory"""
        try:
            best_model_path = (
                self.config.yolov5_dir / 
                "runs" / "train" / self.config.model_name / "weights" / "best.pt"
            )
            
            if best_model_path.exists():  # Path object method
                self.config.root_dir.mkdir(exist_ok=True)  # Path object method
                destination = self.config.root_dir / "best.pt"
                shutil.copy2(str(best_model_path), str(destination))  # Convert to strings
                logger.info(f"📦 Model weights copied to: {destination}")
            else:
                logger.warning(f"⚠️ Best model not found at: {best_model_path}")
                
        except Exception as e:
            logger.error(f"❌ Error copying model: {str(e)}")

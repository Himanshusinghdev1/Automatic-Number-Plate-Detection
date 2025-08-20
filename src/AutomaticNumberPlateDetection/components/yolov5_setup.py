# src/AutomaticNumberPlateDetection/components/yolov5_setup.py
import subprocess
import shutil
import sys
from pathlib import Path
from AutomaticNumberPlateDetection import logger
from AutomaticNumberPlateDetection.entity import YOLOv5SetupConfig

class YOLOv5Setup:
    def __init__(self, config: YOLOv5SetupConfig):
        self.config = config

    def check_python_version(self):
        """Check if Python version is compatible"""
        python_version = sys.version_info
        logger.info(f"🐍 Python version: {python_version.major}.{python_version.minor}.{python_version.micro}")
        
        if python_version.major == 3 and python_version.minor >= 8:
            logger.info("✅ Python version is compatible with YOLOv5")
            return True
        else:
            logger.error("❌ Python version must be >= 3.8 for YOLOv5")
            return False

    def clone_yolov5_repository(self):
        """Clone YOLOv5 repository from GitHub"""
        try:
            if self.config.root_dir.exists():
                logger.info(f"🔄 YOLOv5 directory already exists at: {self.config.root_dir}")
                return True
            
            logger.info(f"📥 Cloning YOLOv5 repository from: {self.config.repository_url}")
            
            result = subprocess.run([
                "git", "clone", self.config.repository_url, str(self.config.root_dir)
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                logger.info(f"✅ YOLOv5 repository cloned successfully to: {self.config.root_dir}")
                return True
            else:
                logger.error(f"❌ Failed to clone repository: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Error cloning YOLOv5: {str(e)}")
            return False

    def install_yolov5_requirements(self):
        """Install YOLOv5 dependencies"""
        try:
            requirements_file = self.config.root_dir / "requirements.txt"
            
            if not requirements_file.exists():
                logger.error(f"❌ Requirements file not found: {requirements_file}")
                return False
            
            logger.info("📦 Installing YOLOv5 requirements...")
            
            result = subprocess.run([
                sys.executable, "-m", "pip", "install", "-r", str(requirements_file)
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                logger.info("✅ YOLOv5 requirements installed successfully")
                return True
            else:
                logger.error(f"❌ Failed to install requirements: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Error installing requirements: {str(e)}")
            return False

    def setup_data_yaml(self):
        """Copy data.yaml to YOLOv5 directory"""
        try:
            if not self.config.data_yaml_path.exists():
                logger.error(f"❌ Data YAML file not found: {self.config.data_yaml_path}")
                return False
            
            destination = self.config.root_dir / "data.yaml"
            shutil.copy2(self.config.data_yaml_path, destination)
            
            logger.info(f"📝 Data YAML copied to: {destination}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error copying data.yaml: {str(e)}")
            return False

    def verify_installation(self):
        """Verify YOLOv5 installation"""
        try:
            # Check if main files exist
            main_files = [
                self.config.root_dir / "train.py",
                self.config.root_dir / "val.py",
                self.config.root_dir / "detect.py",
                self.config.root_dir / "data.yaml"
            ]
            
            missing_files = [f for f in main_files if not f.exists()]
            
            if missing_files:
                logger.error(f"❌ Missing files: {missing_files}")
                return False
            
            # Test import
            logger.info("🔍 Testing YOLOv5 import...")
            test_script = f"""
import sys
sys.path.append('{self.config.root_dir}')
try:
    from models.experimental import attempt_load
    print("✅ YOLOv5 import successful")
except ImportError as e:
    print(f"❌ YOLOv5 import failed: {{e}}")
"""
            
            result = subprocess.run([
                sys.executable, "-c", test_script
            ], capture_output=True, text=True)
            
            if "✅ YOLOv5 import successful" in result.stdout:
                logger.info("✅ YOLOv5 installation verified successfully")
                return True
            else:
                logger.error(f"❌ YOLOv5 verification failed: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Error verifying installation: {str(e)}")
            return False

    def setup_yolov5(self):
        """Complete YOLOv5 setup process"""
        logger.info("🚀 Starting YOLOv5 setup...")
        
        # Step 1: Check Python version
        if not self.check_python_version():
            return False
        
        # Step 2: Clone repository
        if not self.clone_yolov5_repository():
            return False
        
        # Step 3: Install requirements
        if not self.install_yolov5_requirements():
            return False
        
        # Step 4: Setup data.yaml
        if not self.setup_data_yaml():
            return False
        
        # Step 5: Verify installation
        if not self.verify_installation():
            return False
        
        logger.info("🎉 YOLOv5 setup completed successfully!")
        return True

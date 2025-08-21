# Automatic-Number-Plate-Detection

## Project Overview
This project implements a complete Automatic Number Plate Recognition system using YOLOv5 for license plate detection and EasyOCR for text recognition. It includes a professional MLOps pipeline covering data ingestion, preprocessing, model training, testing, OCR integration, and video inference.

---

## Project Structure
```
Automatic-Number-Plate-Detection/
├── artifacts/               # Dataset and preprocessed data
│   ├── data_ingestion/      # Original images and annotations
│   ├── data_preprocessing/  # YOLO formatted annotated data
├── models/                  # Trained model weights
├── yolov5/                  # YOLOv5 framework
├── src/                     # Source code with pipeline components
├── test_images/             # Sample test images
├── test_videos/             # Sample test videos
├── ocr_results/             # OCR output results
├── video_results/           # Video processing outputs
├── config/                  # Configuration YAML files
├── params.yaml              # Parameters YAML file
└── README.md                # Project documentation
```

---

## Pipeline Overview
The project pipeline is divided into these stages:

1. **Data Ingestion**: Downloading and organizing the license plate dataset from Kaggle.
2. **Data Preprocessing**: Parsing XML annotations to YOLO format, splitting training and validation datasets.
3. **YOLOv5 Setup**: Installing and configuring YOLOv5.
4. **Model Training**: Training the YOLOv5 model with early stopping to optimize performance.
5. **Model Testing**: Validating the trained model on static test images.
6. **OCR Integration**: Integrating EasyOCR to read license plate text from detected bounding boxes.
7. **Video Inference**: Running ANPR on test videos for real-time plate detection and recognition.
8. **Results Validation**: Evaluating detection and OCR accuracy with quantitative metrics.

---

## Installation

1. Clone repository:
```bash
git clone <repository-url>
cd Automatic-Number-Plate-Recognition
```

2. Create and activate Python virtual environment:
```bash
python -m venv plate
source plate/bin/activate  # Linux/macOS
plate\Scriptsctivate   # Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

---

## Configuration

Modify `config/config.yaml` and `params.yaml` to adjust:
- Dataset paths
- Training settings (epochs, batch size, early stopping)
- Languages for OCR
- Video processing options

---

## Usage

You can run individual stages or run the full pipeline using the `main.py` script.

### Running Full Pipeline
```bash
python main.py
```

### Run Individual Stage
```bash
python src/AutomaticNumberPlateDetection/pipeline/stage_01_data_ingestion.py
python src/AutomaticNumberPlateDetection/pipeline/stage_02_data_preprocessing.py
...
python src/AutomaticNumberPlateDetection/pipeline/stage_07_video_inference.py
```

### Testing on Images
Place test images in `test_images/` and run:
```bash
python src/AutomaticNumberPlateDetection/pipeline/stage_05_model_testing.py
```

### Running Video Inference
Place test videos in `test_videos/` and run:
```bash
python src/AutomaticNumberPlateDetection/pipeline/stage_07_video_inference.py
```

---

## Results

- Trained model weights are saved in `models/` as `best.pt` and `last.pt`.
- OCR results including detected plate texts and confidences are saved in `ocr_results/`.
- Annotated videos and JSON logs for video ANPR are saved in `video_results/`.

---

## Performance

- License Plate Detection Accuracy: 99.5% mAP@0.5
- OCR Text Recognition: Initial 40-70% success rate
- Early stopping enabled to optimize training time
- Video processing speed: 2-8 FPS depending on hardware

---

## Contact
For questions, issues, or collaborations, please contact: singhhimanshu33456@gmail.com

---

## License
MIT License

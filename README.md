# Automatic-Number-Plate-Detection

conda create -p plate python==3.10 -y

conda activate /Users/himanshu/Downloads/DataScience/DA/Automatic-Number-Plate-Detection/plate

create template.py

AutomaticNumberPlateDetection/
 ├── .github/workflows/.gitkeep
 ├── src/AutomaticNumberPlateDetection/
 │   ├── __init__.py
 │   ├── data/
 │   │   ├── __init__.py
 │   │   └── dataloader.py
 │   ├── components/
 │   │   ├── __init__.py
 │   │   ├── detection.py
 │   │   └── ocr.py
 │   ├── utils/
 │   │   ├── __init__.py
 │   │   └── common.py
 │   ├── config/
 │   │   ├── __init__.py
 │   │   └── configuration.py
 │   ├── pipeline/
 │   │   ├── __init__.py
 │   │   ├── training_pipeline.py
 │   │   └── inference_pipeline.py
 │   ├── entity/__init__.py
 │   └── constants/__init__.py
 ├── config/config.yaml
 ├── dvc.yaml
 ├── params.


create setup.py

pip install -r requirements.txt


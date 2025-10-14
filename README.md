🚀 Falcon 7-Class Object Detection – YOLOv8 Project
📘 Overview
This repository is built for the Duality AI Global Hackathon (GenIgnite 2025) under the challenge “7-Class Object Detection on the Falcon Dataset.”
The project implements YOLOv8 for detecting and classifying safety-related objects on the Falcon Space Station dataset.

🗂️ Dataset
Dataset Source: Falcon AI Hackathon Dataset
Classes (7 total):

Helmet
Gloves
Harness
Shoes
Oxygen Mask
Fire Extinguisher
Wrench
Dataset follows YOLO format:

data/
├─ images/
│  ├─ train/
│  ├─ val/
│  └─ test/
├─ labels/
│  ├─ train/
│  ├─ val/
│  └─ test/
└─ data.yaml
Example data.yaml:

train: ../data/images/train
val: ../data/images/val
test: ../data/images/test

nc: 7
names: ['helmet', 'gloves', 'harness', 'shoes', 'oxygen_mask', 'fire_extinguisher', 'wrench']
⚙️ Environment Setup
python -m venv .venv
source .venv/bin/activate     # Linux / WSL
# or .\.venv\Scripts\Activate.ps1   # Windows PowerShell

pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install ultralytics==8.* opencv-python matplotlib pandas scikit-learn seaborn albumentations onnx onnxruntime
🧠 Training the Model
yolo detect train model=yolov8n.pt data=data/data.yaml epochs=50 imgsz=640 batch=16 name=falcon_yolov8n
To train with a medium model:

yolo detect train model=yolov8m.pt data=data/data.yaml epochs=100 batch=32 name=falcon_yolov8m
🔍 Inference
python scripts/predict.py --weights runs/detect/falcon_yolov8n/weights/best.pt --source data/images/test
For saving predictions:

python scripts/predict.py --weights runs/detect/falcon_yolov8n/weights/best.pt --source data/images/test --save-txt --save-conf
📊 Evaluation
python scripts/evaluate_and_plot.py --weights runs/detect/falcon_yolov8n/weights/best.pt --data data/data.yaml --save plots/
🚀 Optimization
Export for ONNX or TensorRT:

yolo export model=runs/detect/falcon_yolov8n/weights/best.pt format=onnx
trtexec --onnx=model.onnx --saveEngine=model.trt --fp16 --workspace=4096
🔗 Notes
Built for Duality Falcon AI Hackathon 2025
Focus: Space safety detection with robust, lightweight vision models
Model Type: YOLOv8
Optimization: On-device deployment with TensorRT

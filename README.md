# YOLOv8 Project Template

## Directory Layout
```
.
├─ data/
│  ├─ images/
│  │  ├─ train/
│  │  ├─ val/
│  │  └─ test/
│  ├─ labels/
│  │  ├─ train/
│  │  ├─ val/
│  │  └─ test/
│  └─ data.yaml
├─ cfg/
├─ scripts/
│  ├─ train.sh
│  ├─ train_windows.bat
│  ├─ predict.py
│  └─ evaluate_and_plot.py
├─ weights/
├─ report/
└─ README.md
```

## Environment Setup

### Linux / WSL (recommended)
```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install ultralytics==8.* opencv-python matplotlib pandas scikit-learn seaborn albumentations onnx onnxruntime
pip install optuna  # optional for hyper-parameter optimization
```

### Windows (PowerShell)
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install ultralytics==8.* opencv-python matplotlib pandas scikit-learn seaborn albumentations onnx onnxruntime optuna
```

> **Note:** Install the NVIDIA CUDA Toolkit & matching drivers before installing PyTorch with GPU support. TensorRT setup varies by system; follow NVIDIA docs. Add `optuna` only if you plan to run HPO.

## Dataset Configuration
`data/data.yaml`
```yaml
train: ../data/images/train
val: ../data/images/val
test: ../data/images/test

nc: 3
names: ['classA', 'classB', 'classC']
```

Place images under `data/images/` and YOLO-format label `.txt` files under `data/labels/` matching the split.

## Baseline Training
From the project root:
```bash
yolo detect train model=yolov8n.pt data=data/data.yaml epochs=50 imgsz=640 batch=16 name=baseline_run
```

Helper scripts:
- Linux: `bash scripts/train.sh`
- Windows: `scripts\train_windows.bat`

Override defaults via flags, e.g.:
```bash
bash scripts/train.sh --model yolov8m.pt --epochs 100 --batch 32 --name exp_yolov8m
```

## Inference
```bash
python scripts/predict.py --weights runs/detect/baseline_run/weights/best.pt --source data/images/test/img001.jpg
python scripts/predict.py --weights runs/detect/baseline_run/weights/best.pt --source data/images/test --save-txt --save-conf
```

## Evaluation & Plots
```bash
python scripts/evaluate_and_plot.py --weights runs/detect/baseline_run/weights/best.pt --data data/data.yaml --save plots/
```
Generates validation metrics and saves training curves if `results.csv` exists in the run directory.

## Optimization
```bash
yolo export model=runs/detect/baseline_run/weights/best.pt format=onnx
trtexec --onnx=model.onnx --saveEngine=model.trt --fp16 --workspace=4096
```
For INT8 quantization, follow TensorRT calibration workflow with a representative dataset.

## Best Practices
- Keep `runs/detect/.../weights/best.pt` and `last.pt` for reproducibility.
- Track metrics and plots; commit relevant artifacts under `report/` if needed.
- Use deterministic seeds (`--seed`) for fair comparisons.

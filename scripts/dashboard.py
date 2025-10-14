import io
from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image
from ultralytics import YOLO

st.set_page_config(page_title="YOLO Dashboard", layout="wide")

default_weights = Path("runs/detect/cpu_test2/weights/best.pt")
weights_path_text = st.sidebar.text_input("Weights path", str(default_weights))
imgsz = st.sidebar.number_input("Image size", min_value=128, max_value=1280, value=640, step=32)
conf = st.sidebar.slider("Confidence", 0.01, 0.99, 0.25, 0.01)
iou = st.sidebar.slider("IoU", 0.1, 0.9, 0.45, 0.01)
device = st.sidebar.text_input("Device", "cpu")
source_option = st.sidebar.selectbox("Source", ("Upload image", "Sample val image"))

@st.cache_resource(show_spinner=False)
def load_model(path: Path) -> YOLO:
    return YOLO(str(path))

weights_path = Path(weights_path_text)
if not weights_path.exists():
    st.error(f"Weights not found: {weights_path}")
    st.stop()

model = load_model(weights_path)
st.title("YOLO Model Dashboard")
st.caption("Interactive inference and training metrics")

col_pred, col_metrics = st.columns((0.6, 0.4))

with col_pred:
    if source_option == "Upload image":
        uploaded = st.file_uploader("Upload image", type=["jpg", "jpeg", "png", "bmp"])
        if uploaded is not None:
            image = Image.open(uploaded).convert("RGB")
            results = model.predict(image, imgsz=imgsz, conf=conf, iou=iou, device=device)
            for idx, result in enumerate(results):
                st.subheader(f"Prediction {idx + 1}")
                plotted = result.plot()[:, :, ::-1]
                st.image(plotted, use_column_width=True)
                boxes = result.boxes
                if boxes is not None and boxes.xyxy is not None:
                    df = pd.DataFrame({
                        "x1": boxes.xyxy[:, 0].cpu().numpy(),
                        "y1": boxes.xyxy[:, 1].cpu().numpy(),
                        "x2": boxes.xyxy[:, 2].cpu().numpy(),
                        "y2": boxes.xyxy[:, 3].cpu().numpy(),
                        "confidence": boxes.conf.cpu().numpy(),
                        "class": boxes.cls.cpu().numpy().astype(int),
                    })
                    st.dataframe(df, hide_index=True)
    else:
        run_dir = weights_path.parent.parent
        sample_images = sorted((run_dir / "val_batch0_pred.jpg", run_dir / "val_batch1_pred.jpg", run_dir / "val_batch2_pred.jpg"))
        available = [img for img in sample_images if img.exists()]
        if not available:
            st.info("No sample predictions found in run directory.")
        else:
            selected = st.selectbox("Sample output", available)
            st.image(str(selected), use_column_width=True)

with col_metrics:
    run_dir = weights_path.parent.parent
    st.subheader("Training metrics")
    results_csv = run_dir / "results.csv"
    if results_csv.exists():
        df_metrics = pd.read_csv(results_csv)
        metric_choice = st.selectbox("Metric", ["metrics/precision(B)", "metrics/recall(B)", "metrics/mAP50(B)", "metrics/mAP50-95(B)", "train/box_loss", "train/cls_loss", "train/dfl_loss", "val/box_loss"])
        if metric_choice in df_metrics.columns:
            st.line_chart(df_metrics[metric_choice])
        else:
            st.info(f"Metric {metric_choice} not found in results.csv")
    else:
        st.info("results.csv not available for this run")
    st.subheader("Visual summaries")
    image_names = ["confusion_matrix.png", "confusion_matrix_normalized.png", "BoxPR_curve.png", "BoxF1_curve.png", "BoxP_curve.png", "BoxR_curve.png", "results.png"]
    for name in image_names:
        image_path = run_dir / name
        if image_path.exists():
            st.image(str(image_path), caption=name)

st.sidebar.header("Batch prediction")
folder_input = st.sidebar.text_input("Image folder", "data/images/val")
run_predictions = st.sidebar.button("Run folder inference")
if run_predictions:
    folder_path = Path(folder_input)
    if not folder_path.exists():
        st.sidebar.error(f"Folder not found: {folder_path}")
    else:
        results = model.predict(source=str(folder_path), imgsz=imgsz, conf=conf, iou=iou, device=device, save=True)
        output_dir = getattr(results, "save_dir", None)
        if output_dir is not None:
            st.sidebar.success(f"Saved batch predictions to {output_dir}")
        else:
            st.sidebar.warning("Batch prediction completed without saved outputs")

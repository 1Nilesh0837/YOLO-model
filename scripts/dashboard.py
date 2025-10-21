# app.py
import time
import io
from pathlib import Path
from typing import Optional, List

import pandas as pd
import streamlit as st
from PIL import Image
from ultralytics import YOLO

# ---------- Page config ----------
st.set_page_config(page_title="AI Space Safety — YOLO Dashboard", layout="wide", initial_sidebar_state="expanded")

# ---------- Helpers ----------
@st.cache_resource(show_spinner=False)
def load_model(path: str) -> YOLO:
    """Load and cache the YOLO model."""
    return YOLO(path)

def boxes_to_df(boxes) -> pd.DataFrame:
    """Convert Ultralyitcs boxes object to dataframe if possible."""
    try:
        xyxy = boxes.xyxy.cpu().numpy()
        conf = boxes.conf.cpu().numpy()
        cls = boxes.cls.cpu().numpy().astype(int)
        return pd.DataFrame({
            "x1": xyxy[:, 0],
            "y1": xyxy[:, 1],
            "x2": xyxy[:, 2],
            "y2": xyxy[:, 3],
            "confidence": conf,
            "class": cls,
        })
    except Exception:
        return pd.DataFrame()

def add_event(msg: str):
    """Push an event to session state event log (most recent first)."""
    ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    entry = f"{ts} — {msg}"
    st.session_state["events"].insert(0, entry)
    # keep a limit
    st.session_state["events"] = st.session_state["events"][:200]

def pretty_alerts_from_df(df: pd.DataFrame, required_classes: Optional[dict] = None) -> List[str]:
    """Create simple alert strings based on detected classes and expected required_classes mapping."""
    messages = []
    if df.empty:
        messages.append("No detections.")
        return messages
    counts = df["class"].value_counts().to_dict()
    for cls_id, cnt in counts.items():
        messages.append(f"Class {cls_id}: {cnt} detected")
    # If user provided required classes, check missing
    if required_classes:
        present = set(df["class"].tolist())
        for cls_id, name in required_classes.items():
            if cls_id not in present:
                messages.append(f"⚠️ Required object missing: {name}")
    return messages

# ---------- Session state ----------
if "events" not in st.session_state:
    st.session_state["events"] = []
if "live" not in st.session_state:
    st.session_state["live"] = False

# ---------- Sidebar: grouped settings ----------
st.sidebar.title("AI Space Safety — Controls")

with st.sidebar.expander("Model Settings", expanded=True):
    default_weights = Path("runs/detect/cpu_test2/weights/best.pt")
    weights_path_text = st.text_input("Weights path", str(default_weights))
    device = st.selectbox("Device", ["cpu", "cuda"], index=0)
    quantize = st.checkbox("Use INT8 (edge) — note: must prepare quantized model separately", value=False)

with st.sidebar.expander("Inference Settings", expanded=True):
    imgsz = st.number_input("Image size", min_value=128, max_value=1280, value=640, step=32)
    conf = st.slider("Confidence threshold", 0.01, 0.99, 0.25, 0.01)
    iou = st.slider("IoU threshold", 0.1, 0.9, 0.45, 0.01)

with st.sidebar.expander("Source & Batch", expanded=False):
    source_option = st.selectbox("Source", ("Upload image", "Sample run image", "Folder for batch"))
    folder_input = st.text_input("Image folder (for batch)", "data/images/val")
    run_predictions = st.button("Run batch inference", key="run_batch")

with st.sidebar.expander("Live Mode", expanded=False):
    live_toggle = st.button("Start Live Mode" if not st.session_state["live"] else "Stop Live Mode", key="live_toggle")

# ---------- Top status bar ----------
status_col1, status_col2, status_col3 = st.columns([3, 6, 2])
with status_col1:
    st.markdown("### 🚀 AI Space Safety Monitor")
with status_col2:
    if Path(weights_path_text).exists():
        st.success(f"Model weights found: {Path(weights_path_text).name}", icon="✅")
    else:
        st.error(f"Weights not found: {weights_path_text}", icon="❗")
with status_col3:
    st.metric("Device", device.upper())

# ---------- Try loading model ----------
model = None
try:
    if not Path(weights_path_text).exists():
        raise FileNotFoundError(weights_path_text)
    model = load_model(str(weights_path_text))
    add_event(f"Model loaded from {weights_path_text}")
except Exception as e:
    st.error(f"Failed to load model: {e}")
    st.stop()

# ---------- Main layout ----------
st.title("Mission Control — Live Detection")
st.caption("Real-time safety object detection for space station environments")

main_col, side_col = st.columns((0.7, 0.3))

# Side panel: alerts, event log, quick metrics
with side_col:
    st.subheader("Instant Alerts")
    alert_placeholder = st.empty()
    st.subheader("Recent Events")
    with st.container():
        for ev in st.session_state["events"][:10]:
            st.write(f"- {ev}")
    st.subheader("Quick Metrics")
    cols = st.columns(3)
    cols[0].metric("Last FPS", "—")
    cols[1].metric("Last Inference (ms)", "—")
    cols[2].metric("Objects (last)", "—")

# Main panel: camera / image feed and detection table
with main_col:
    feed_placeholder = st.empty()
    info_cols = st.columns(3)
    info_cols[0].metric("Confidence", f"{conf:.2f}")
    info_cols[1].metric("IoU", f"{iou:.2f}")
    info_cols[2].metric("Image Size", f"{imgsz}px")

    # Helper to run inference on a PIL image and show outputs
    def run_inference_and_show(pil_img: Image.Image):
        t0 = time.time()
        results = model.predict(pil_img, imgsz=imgsz, conf=conf, iou=iou, device=device)
        t_elapsed = (time.time() - t0) * 1000.0  # ms
        # Show first result only (YOLO may return list)
        result = results[0]
        plotted = result.plot()  # numpy BGR
        # convert for PIL/streamlit
        if hasattr(plotted, "shape"):
            # convert BGR->RGB if needed
            import numpy as np
            if plotted.shape[2] == 3:
                plotted = plotted[:, :, ::-1]
            feed_placeholder.image(plotted, use_column_width=True)
        else:
            feed_placeholder.image(pil_img, use_column_width=True)
        boxes = getattr(result, "boxes", None)
        df = boxes_to_df(boxes) if boxes is not None else pd.DataFrame()
        # Update side metrics
        side_metrics = st.session_state.get("_side_metrics_last", {})
        side_metrics.update({"last_inference_ms": f"{int(t_elapsed)}ms", "last_objects": len(df)})
        st.session_state["_side_metrics_last"] = side_metrics
        # show detections table
        if not df.empty:
            st.table(df.head(10))
        else:
            st.info("No objects detected in this frame.")

        # events & alerts
        alerts = pretty_alerts_from_df(df, required_classes=None)  # pass mapping if you want human names
        if alerts:
            alert_placeholder.markdown("\n".join([f"- {a}" for a in alerts]))
        add_event(f"Inference done — {len(df)} objects — {int(t_elapsed)}ms")
        return df, int(t_elapsed)

    # Source handling
    if source_option == "Upload image":
        uploaded = st.file_uploader("Upload image", type=["jpg", "jpeg", "png", "bmp"])
        if uploaded is not None:
            pil = Image.open(io.BytesIO(uploaded.read())).convert("RGB")
            df, t_ms = run_inference_and_show(pil)

    elif source_option == "Sample run image":
        run_dir = Path(weights_path_text).parent.parent
        sample_images = sorted([p for p in run_dir.glob("val_batch*_pred.jpg")])
        if not sample_images:
            st.info("No sample predictions found in run directory.")
        else:
            selected = st.selectbox("Sample output", sample_images, format_func=lambda p: p.name)
            st.image(str(selected), use_column_width=True)
            try:
                pil = Image.open(selected).convert("RGB")
                df, t_ms = run_inference_and_show(pil)
            except Exception as e:
                st.error(f"Could not open sample image: {e}")

    else:  # Folder for batch (single-image preview + ability to run batch)
        folder_path = Path(folder_input)
        if folder_path.exists() and folder_path.is_dir():
            sample_list = sorted([p for p in folder_path.glob("*") if p.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]])
            if sample_list:
                st.write("Sample from folder:")
                sample = sample_list[0]
                st.image(str(sample), use_column_width=True)
                if st.button("Run single image from folder"):
                    pil = Image.open(sample).convert("RGB")
                    run_inference_and_show(pil)
            else:
                st.info("No images found in folder.")
        else:
            st.warning(f"Folder not found: {folder_path}")

# ---------- Batch inference handling ----------
if run_predictions:
    folder_path = Path(folder_input)
    if not folder_path.exists():
        st.sidebar.error(f"Folder not found: {folder_path}")
    else:
        with st.sidebar:
            st.info("Running batch inference. Outputs will be saved next to run directory.")
        try:
            results = model.predict(source=str(folder_path), imgsz=imgsz, conf=conf, iou=iou, device=device, save=True)
            # Ultralyitcs returns a results list; attempt to discover save_dir
            save_dir = getattr(results, "save_dir", None) or getattr(results, "saved", None)
            if save_dir:
                st.sidebar.success(f"Saved batch predictions to: {save_dir}")
                add_event(f"Batch inference saved to {save_dir}")
            else:
                st.sidebar.success("Batch inference completed (no explicit save dir found).")
                add_event("Batch inference completed (no save dir).")
        except Exception as e:
            st.sidebar.error(f"Batch inference failed: {e}")
            add_event(f"Batch inference failed: {e}")

# ---------- Live mode toggle logic ----------
if live_toggle:
    st.session_state["live"] = not st.session_state["live"]
    # immediate UI feedback
    if st.session_state["live"]:
        add_event("Live mode started")
    else:
        add_event("Live mode stopped")

# If live mode ON, run a loop (stoppable by toggling the button above)
if st.session_state["live"]:
    # try webcam (0) first; fallback to sample folder
    try:
        import cv2
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise RuntimeError("Webcam not available")
        add_event("Webcam opened for live stream")
        st.info("Live mode: Webcam feed (Press 'Start Live Mode' button again to stop)")
        while st.session_state["live"]:
            ret, frame = cap.read()
            if not ret:
                add_event("Webcam frame read failed")
                break
            # convert frame to PIL
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(frame_rgb)
            run_inference_and_show(pil_img)
            # update small metrics on the side
            sm = st.session_state.get("_side_metrics_last", {})
            # update UI metrics (best-effort)
            try:
                cols = side_col.columns(3)
                cols[0].metric("Last FPS",  int(1000 / max(1, sm.get("last_inference_ms", 1000).replace("ms", ""))))
                cols[1].metric("Last Inference (ms)", sm.get("last_inference_ms", "—"))
                cols[2].metric("Objects (last)", sm.get("last_objects", 0))
            except Exception:
                pass
            time.sleep(0.5)
        cap.release()
        add_event("Live mode ended (webcam closed)")
    except Exception as e:
        add_event(f"Live mode webcam failed: {e}")
        st.warning("Webcam not available — live mode requires a webcam. You can use the folder or upload modes instead.")
        st.session_state["live"] = False

# ---------- Post-run: show compact logs & tips ----------
st.markdown("---")
st.subheader("Operator Notes & Tips")
st.write(
    """
- Use INT8 quantized weights on edge devices for lower latency and power consumption.
- For GPU / high-throughput, deploy with TensorRT or use a CUDA-enabled instance.
- Keep a curated dataset of safety objects only — mislabeled classes hurt performance.
- Consider a small ensemble + majority vote for mission-critical redundancy.
"""
)
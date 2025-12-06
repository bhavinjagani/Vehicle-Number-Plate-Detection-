"""
ANPR Web Application - Vehicle Number Plate Detection UI
Upload a video and detect license plates in real-time!
Supports both YOLOv5 and YOLOv9 models.
"""

import gradio as gr
import cv2
import numpy as np
import tempfile
import sys
from pathlib import Path
import torch
import easyocr
from collections import defaultdict
import subprocess
import json
import os

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent

# Global cached models for speed
_cached_models = {
    'yolov5': None,
    'yolov9': None,
    'ocr': None
}

# Model paths
YOLOV5_WEIGHTS = PROJECT_ROOT / "runs" / "train_yolov5" / "exp" / "weights" / "best.pt"
YOLOV9_WEIGHTS = PROJECT_ROOT / "runs" / "train_yolov9" / "exp" / "weights" / "best.pt"

# Performance settings
FRAME_SKIP = 2  # Process every Nth frame (1 = all, 2 = every other, 3 = every 3rd)


def get_device():
    """Detect best available device"""
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def get_yolov5_model():
    """Get cached YOLOv5 model or load it once"""
    global _cached_models
    
    if _cached_models['yolov5'] is None:
        print("Loading YOLOv5 model (one-time)...")
        yolov5_path = str(PROJECT_ROOT / "external" / "yolov5")
        
        # Clear conflicting modules only on first load
        mods = [k for k in list(sys.modules.keys()) if k.startswith(('models.', 'utils.')) and 'gradio' not in k]
        for m in mods:
            try:
                del sys.modules[m]
            except:
                pass
        
        model = torch.hub.load(
            yolov5_path,
            'custom',
            path=str(YOLOV5_WEIGHTS),
            source='local',
            force_reload=False
        )
        model.conf = 0.15
        model.iou = 0.45
        _cached_models['yolov5'] = model
        print("✓ YOLOv5 loaded!")
    
    return _cached_models['yolov5']


def get_yolov9_model():
    """Get cached YOLOv9 model or load it once"""
    global _cached_models
    
    if _cached_models['yolov9'] is None:
        print("Loading YOLOv9 model (one-time)...")
        yolov9_path = str(PROJECT_ROOT / "external" / "yolov9")
        
        # Clear conflicting modules
        mods = [k for k in list(sys.modules.keys()) if k.startswith(('models.', 'utils.')) and 'gradio' not in k]
        for m in mods:
            try:
                del sys.modules[m]
            except:
                pass
        
        model = torch.hub.load(
            yolov9_path,
            'custom',
            path=str(YOLOV9_WEIGHTS),
            source='local',
            force_reload=False
        )
        model.conf = 0.15
        model.iou = 0.45
        _cached_models['yolov9'] = model
        print("✓ YOLOv9 loaded!")
    
    return _cached_models['yolov9']


def load_ocr():
    """Load OCR reader (cached)"""
    global _cached_models
    if _cached_models['ocr'] is None:
        print("Loading EasyOCR...")
        _cached_models['ocr'] = easyocr.Reader(['en'], gpu=torch.cuda.is_available())
        print("✓ OCR loaded!")
    return _cached_models['ocr']


def preprocess_plate_for_ocr(plate_img):
    """Fast preprocessing for OCR"""
    if plate_img is None or plate_img.size == 0:
        return None
    try:
        h, w = plate_img.shape[:2]
        if w < 80:
            scale = 150 / w
            plate_img = cv2.resize(plate_img, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
        
        if len(plate_img.shape) == 3:
            gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
        else:
            gray = plate_img
        
        # Simple contrast enhancement
        gray = cv2.convertScaleAbs(gray, alpha=1.5, beta=20)
        return gray
    except:
        return None


def clean_plate_text(text):
    """Clean and validate plate text"""
    if not text:
        return ""
    cleaned = ''.join(c for c in text.upper() if c.isalnum() or c == ' ')
    cleaned = ' '.join(cleaned.split())
    return cleaned if len(cleaned) >= 2 else ""


def run_ocr_on_plate(plate_img, ocr_reader):
    """Run OCR once with preprocessing"""
    if plate_img is None or plate_img.size == 0:
        return ""
    
    try:
        preprocessed = preprocess_plate_for_ocr(plate_img)
        img_to_read = preprocessed if preprocessed is not None else plate_img
        ocr_result = ocr_reader.readtext(img_to_read, detail=0, paragraph=True)
        if ocr_result:
            return clean_plate_text(' '.join(ocr_result))
    except:
        pass
    
    return ""


def run_yolov5_detection(frame, conf_threshold=0.15):
    """Run YOLOv5 detection using cached model"""
    model = get_yolov5_model()
    model.conf = conf_threshold
    results = model(frame)
    return results.xyxy[0].cpu().numpy()


def run_yolov9_detection(frame, conf_threshold=0.15):
    """Run YOLOv9 detection via subprocess to avoid module conflicts"""
    # Save frame to temp file
    temp_img = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
    temp_path = temp_img.name
    temp_img.close()
    
    try:
        cv2.imwrite(temp_path, frame)
        
        script_path = PROJECT_ROOT / "src" / "yolov9_detect.py"
        result = subprocess.run(
            [
                sys.executable, str(script_path),
                "--image", temp_path,
                "--weights", str(YOLOV9_WEIGHTS),
                "--conf", str(conf_threshold)
            ],
            capture_output=True,
            text=True,
            timeout=30
        )
        
        for line in result.stdout.split('\n'):
            if line.startswith("RESULT:"):
                preds = json.loads(line[7:])
                return np.array(preds) if preds else np.array([])
        
        return np.array([])
        
    except Exception as e:
        print(f"YOLOv9 error: {e}")
        return np.array([])
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


def process_frame_simple(frame, model_name, ocr_reader, conf_threshold=0.15):
    """Process a single frame - optimized version"""
    detections = []
    
    # Run detection (no heavy enhancement - model handles it)
    if model_name == "yolov5":
        preds = run_yolov5_detection(frame, conf_threshold)
    else:
        preds = run_yolov9_detection(frame, conf_threshold)
    
    h, w = frame.shape[:2]
    color = (0, 255, 0) if model_name == "yolov5" else (0, 165, 255)
    
    for pred in preds:
        x1, y1, x2, y2, conf, cls = pred
        if conf < conf_threshold:
            continue
        
        pad = 3
        x1, y1 = max(0, int(x1) - pad), max(0, int(y1) - pad)
        x2, y2 = min(w, int(x2) + pad), min(h, int(y2) + pad)
        
        plate_img = frame[y1:y2, x1:x2]
        plate_text = run_ocr_on_plate(plate_img, ocr_reader)
        
        detections.append({'bbox': (x1, y1, x2, y2), 'conf': conf, 'text': plate_text})
        
        # Draw box and label
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = plate_text if plate_text else "Plate"
        cv2.putText(frame, f"{label} {conf:.0%}", (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    return frame, detections


def draw_overlay(frame, model_name, frame_count, total_frames, det_count):
    """Draw info overlay on frame"""
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 60), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
    
    color = (0, 255, 0) if model_name == "yolov5" else (0, 165, 255)
    cv2.putText(frame, f"{model_name.upper()}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    cv2.putText(frame, f"Frame: {frame_count}/{total_frames}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(frame, f"Plates: {det_count}", (w - 100, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    return frame


def process_video(video_path, model_choice, conf_threshold):
    """Process video with selected model - OPTIMIZED with frame skipping"""
    if video_path is None:
        return None, "⚠️ Please upload a video first!"
    
    model_name = "yolov5" if "YOLOv5" in model_choice else "yolov9"
    
    try:
        ocr = load_ocr()
        
        # Pre-load model to avoid first-frame delay
        if model_name == "yolov5":
            get_yolov5_model()
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None, "❌ Could not open video"
        
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        output_path = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False).name
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        plate_tracker = defaultdict(int)
        last_detections = []  # Cache for skipped frames
        
        print(f"Processing with {model_name.upper()} (skip={FRAME_SKIP})...")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            annotated = frame.copy()
            
            # Only process every Nth frame, reuse detections for others
            if frame_count % FRAME_SKIP == 1 or FRAME_SKIP == 1:
                try:
                    annotated, detections = process_frame_simple(annotated, model_name, ocr, conf_threshold)
                    last_detections = detections
                except Exception as e:
                    print(f"Frame {frame_count} error: {e}")
                    detections = []
            else:
                # Reuse last detections, just draw them
                detections = last_detections
                color = (0, 255, 0) if model_name == "yolov5" else (0, 165, 255)
                for det in detections:
                    x1, y1, x2, y2 = det['bbox']
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                    label = det['text'] if det['text'] else "Plate"
                    cv2.putText(annotated, f"{label} {det['conf']:.0%}", (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # Track plates
            for det in detections:
                if det['text']:
                    plate_tracker[det['text']] += 1
            
            # Draw overlay
            annotated = draw_overlay(annotated, model_name, frame_count, total_frames, len(detections))
            out.write(annotated)
            
            if frame_count % 50 == 0:
                print(f"  {frame_count}/{total_frames} frames")
        
        cap.release()
        out.release()
        
        unique_plates = sorted(plate_tracker.items(), key=lambda x: x[1], reverse=True)
        
        emoji = "🟢" if model_name == "yolov5" else "🟠"
        summary = f"## ✅ Done!\n\n**Model:** {emoji} {model_name.upper()}\n**Video:** {total_frames} frames @ {fps}fps\n\n"
        
        if unique_plates:
            summary += "**Plates Found:**\n"
            for plate, count in unique_plates[:10]:
                summary += f"- `{plate}` ({count}x)\n"
        else:
            summary += "⚠️ No plates found. Try lower confidence."
        
        return output_path, summary
        
    except Exception as e:
        import traceback
        return None, f"❌ Error: {e}\n```\n{traceback.format_exc()}\n```"


def process_image(image, model_choice, conf_threshold):
    """Process image with selected model"""
    if image is None:
        return None, "⚠️ Please upload an image!"
    
    model_name = "yolov5" if "YOLOv5" in model_choice else "yolov9"
    
    try:
        ocr = load_ocr()
        frame = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        annotated, detections = process_frame_simple(frame, model_name, ocr, conf_threshold)
        result = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        
        emoji = "🟢" if model_name == "yolov5" else "🟠"
        summary = f"## {emoji} {model_name.upper()} - {len(detections)} plate(s)\n\n"
        
        if detections:
            for det in detections:
                plate = det['text'] if det['text'] else "(unreadable)"
                summary += f"- **`{plate}`** ({det['conf']:.0%})\n"
        else:
            summary += "No plates detected."
        
        return result, summary
        
    except Exception as e:
        import traceback
        return None, f"❌ Error: {e}\n```\n{traceback.format_exc()}\n```"


# Check model availability
def get_available_models():
    models = []
    if YOLOV5_WEIGHTS.exists():
        models.append("🟢 YOLOv5 (Fast & Accurate)")
    if YOLOV9_WEIGHTS.exists():
        models.append("🟠 YOLOv9 (Newer Architecture)")
    return models if models else ["No models found!"]


# Create UI
print("🚀 Starting ANPR App...")

available = get_available_models()

with gr.Blocks(title="ANPR") as app:
    gr.Markdown("""
    # 🚗 License Plate Detection (ANPR)
    ### YOLOv5 / YOLOv9 + EasyOCR
    ---
    """)
    
    with gr.Tabs():
        with gr.TabItem("🎬 Video"):
            with gr.Row():
                with gr.Column():
                    vid_in = gr.Video(label="Upload Video")
                    vid_model = gr.Radio(available, value=available[0], label="Model")
                    vid_conf = gr.Slider(0.05, 0.9, 0.15, 0.05, label="Confidence")
                    vid_btn = gr.Button("🔍 Detect", variant="primary")
                with gr.Column():
                    vid_out = gr.Video(label="Result")
                    vid_txt = gr.Markdown()
            
            vid_btn.click(process_video, [vid_in, vid_model, vid_conf], [vid_out, vid_txt])
        
        with gr.TabItem("🖼️ Image"):
            with gr.Row():
                with gr.Column():
                    img_in = gr.Image(label="Upload Image", type="numpy")
                    img_model = gr.Radio(available, value=available[0], label="Model")
                    img_conf = gr.Slider(0.05, 0.9, 0.15, 0.05, label="Confidence")
                    img_btn = gr.Button("🔍 Detect", variant="primary")
                with gr.Column():
                    img_out = gr.Image(label="Result")
                    img_txt = gr.Markdown()
            
            img_btn.click(process_image, [img_in, img_model, img_conf], [img_out, img_txt])
    
    gr.Markdown("---\n*Computer Vision Project - SEM3*")

if __name__ == "__main__":
    print("=" * 40)
    print("📍 http://localhost:7860")
    print("=" * 40)
    app.launch(server_name="0.0.0.0", server_port=7860)
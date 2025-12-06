"""
ANPR Web Application - YOLOv5 & YOLOv9
Proper solution with subprocess isolation for YOLOv9.
"""

import gradio as gr
import cv2
import numpy as np
import tempfile
import subprocess
import sys
import json
import os
from pathlib import Path
import torch
import easyocr
from collections import defaultdict

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
SRC_DIR = Path(__file__).parent
YOLOV5_PATH = PROJECT_ROOT / "external" / "yolov5"
YOLOV9_DETECTOR = SRC_DIR / "yolov9_detect.py"
YOLOV5_WEIGHTS = PROJECT_ROOT / "runs" / "train_yolov5" / "exp" / "weights" / "best.pt"
YOLOV9_WEIGHTS = PROJECT_ROOT / "runs" / "train_yolov9" / "exp" / "weights" / "best.pt"

# Global
ocr_reader = None
yolov5_model = None


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_ocr():
    global ocr_reader
    if ocr_reader is None:
        print("📝 Loading EasyOCR...")
        ocr_reader = easyocr.Reader(['en'], gpu=torch.cuda.is_available())
        print("✅ OCR loaded!")
    return ocr_reader


def load_yolov5():
    """Load YOLOv5 in main process"""
    global yolov5_model
    if yolov5_model is None:
        print("📦 Loading YOLOv5...")
        if str(YOLOV5_PATH) not in sys.path:
            sys.path.insert(0, str(YOLOV5_PATH))
        yolov5_model = torch.hub.load(
            str(YOLOV5_PATH), 'custom', 
            path=str(YOLOV5_WEIGHTS), 
            source='local',
            force_reload=True
        )
        yolov5_model.conf = 0.15
        yolov5_model.iou = 0.45
        print("✅ YOLOv5 loaded!")
    return yolov5_model


def run_yolov5(frame, conf):
    """Run YOLOv5 detection"""
    model = load_yolov5()
    model.conf = conf
    results = model(frame)
    return results.xyxy[0].cpu().numpy().tolist()


def run_yolov9(image_path, conf):
    """Run YOLOv9 via subprocess"""
    try:
        cmd = [
            sys.executable,
            str(YOLOV9_DETECTOR),
            "--image", str(image_path),
            "--weights", str(YOLOV9_WEIGHTS),
            "--conf", str(conf)
        ]
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120,
            cwd=str(PROJECT_ROOT)
        )
        
        # Parse output
        for line in result.stdout.split('\n'):
            if line.startswith("RESULT:"):
                data = line.replace("RESULT:", "")
                return json.loads(data)
        
        if result.stderr:
            print(f"YOLOv9 stderr: {result.stderr[:200]}")
        
        return []
    except subprocess.TimeoutExpired:
        print("YOLOv9 timeout!")
        return []
    except Exception as e:
        print(f"YOLOv9 error: {e}")
        return []


def get_models():
    models = []
    if YOLOV5_WEIGHTS.exists():
        models.append("🟢 YOLOv5")
    if YOLOV9_WEIGHTS.exists():
        models.append("🟠 YOLOv9")
    return models if models else ["No models"]


def clean_text(text):
    if not text:
        return ""
    cleaned = ''.join(c for c in text.upper() if c.isalnum() or c == ' ')
    return ' '.join(cleaned.split()) if len(cleaned) >= 2 else ""


def run_ocr(plate_img, ocr):
    if plate_img is None or plate_img.size == 0:
        return ""
    try:
        h, w = plate_img.shape[:2]
        if w < 80:
            plate_img = cv2.resize(plate_img, None, fx=120/w, fy=120/w)
        result = ocr.readtext(plate_img, detail=0, paragraph=True)
        return clean_text(' '.join(result)) if result else ""
    except:
        return ""


def detect(frame, model_choice, conf):
    """Run detection with selected model"""
    is_yolov5 = "YOLOv5" in model_choice
    
    if is_yolov5:
        return run_yolov5(frame, conf)
    else:
        # Save to temp file for YOLOv9 subprocess
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as f:
            cv2.imwrite(f.name, frame)
            preds = run_yolov9(f.name, conf)
            os.unlink(f.name)
            return preds


def process_frame(frame, model_choice, ocr, conf):
    """Process single frame"""
    h, w = frame.shape[:2]
    detections = []
    
    is_yolov5 = "YOLOv5" in model_choice
    color = (0, 255, 0) if is_yolov5 else (0, 165, 255)
    
    preds = detect(frame, model_choice, conf)
    
    for pred in preds:
        x1, y1, x2, y2, confidence, cls = pred[:6]
        if confidence < conf:
            continue
        
        pad = 3
        x1, y1 = max(0, int(x1) - pad), max(0, int(y1) - pad)
        x2, y2 = min(w, int(x2) + pad), min(h, int(y2) + pad)
        
        plate_img = frame[y1:y2, x1:x2]
        plate_text = run_ocr(plate_img, ocr)
        
        detections.append({
            'bbox': (x1, y1, x2, y2),
            'conf': float(confidence),
            'text': plate_text
        })
        
        # Draw box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = plate_text if plate_text else "Plate"
        cv2.rectangle(frame, (x1, y1 - 22), (x1 + len(label) * 11 + 5, y1), color, -1)
        cv2.putText(frame, label, (x1 + 3, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
    
    return frame, detections


def process_video(video_path, model_choice, conf, progress=gr.Progress()):
    if video_path is None:
        return None, "⚠️ Upload a video first!"
    
    model_name = "YOLOv5" if "YOLOv5" in model_choice else "YOLOv9"
    emoji = "🟢" if "YOLOv5" in model_choice else "🟠"
    color = (0, 255, 0) if "YOLOv5" in model_choice else (0, 165, 255)
    
    try:
        ocr = load_ocr()
        
        cap = cv2.VideoCapture(video_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        output = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False).name
        out = cv2.VideoWriter(output, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
        
        count = 0
        all_det = []
        plates = defaultdict(int)
        
        # Skip more frames for YOLOv9 (subprocess is slower)
        if "YOLOv9" in model_choice:
            skip = max(10, total // 30)
        else:
            skip = max(1, total // 100)
        
        last_det = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            count += 1
            if count % 5 == 0:
                progress(count / total, f"{model_name}: Frame {count}/{total}")
            
            if count % skip == 0 or count == 1:
                annotated, det = process_frame(frame.copy(), model_choice, ocr, conf)
                last_det = det
            else:
                annotated = frame.copy()
                # Redraw last detections
                for d in last_det:
                    x1, y1, x2, y2 = d['bbox']
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                    label = d['text'] if d['text'] else "Plate"
                    cv2.rectangle(annotated, (x1, y1 - 22), (x1 + len(label) * 11 + 5, y1), color, -1)
                    cv2.putText(annotated, label, (x1 + 3, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
                det = last_det
            
            for d in det:
                if d['text']:
                    plates[d['text']] += 1
            all_det.extend(det)
            
            # HUD
            cv2.rectangle(annotated, (5, 5), (220, 55), (0, 0, 0), -1)
            cv2.putText(annotated, f"{model_name} ANPR", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            cv2.putText(annotated, f"Frame {count}/{total}", (10, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
            
            out.write(annotated)
        
        cap.release()
        out.release()
        
        unique = sorted(plates.items(), key=lambda x: x[1], reverse=True)
        
        summary = f"""## ✅ Complete!

**Model:** {emoji} {model_name}
**Frames:** {total} @ {fps} FPS
**Processed:** Every {skip} frames

### 📊 Results
- **Detections:** {len(all_det)}
- **Unique Plates:** {len(unique)}

### 🚗 Plates Found:
"""
        for p, c in unique[:10]:
            summary += f"\n- **`{p}`** ({c}×)"
        if not unique:
            summary += "\n⚠️ No plates detected. Try lower confidence."
        
        return output, summary
    except Exception as e:
        import traceback
        return None, f"❌ Error: {e}\n```\n{traceback.format_exc()}\n```"


def process_image(image, model_choice, conf):
    if image is None:
        return None, "⚠️ Upload an image first!"
    
    model_name = "YOLOv5" if "YOLOv5" in model_choice else "YOLOv9"
    emoji = "🟢" if "YOLOv5" in model_choice else "🟠"
    
    try:
        ocr = load_ocr()
        frame = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        annotated, det = process_frame(frame, model_choice, ocr, conf)
        result = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        
        summary = f"## {emoji} {model_name} Detection\n\n**Found:** {len(det)} plate(s)\n\n"
        if det:
            for i, d in enumerate(det, 1):
                text = d['text'] if d['text'] else "(unreadable)"
                summary += f"{i}. **`{text}`** — {d['conf']:.0%}\n"
        else:
            summary += "⚠️ No plates detected. Try lower confidence."
        
        return result, summary
    except Exception as e:
        import traceback
        return None, f"❌ Error: {e}\n```\n{traceback.format_exc()}\n```"


# ============== STARTUP ==============
print("=" * 50)
print("🚀 ANPR - YOLOv5 + YOLOv9")
print(f"📦 YOLOv5: {'✓' if YOLOV5_WEIGHTS.exists() else '✗'} {YOLOV5_WEIGHTS}")
print(f"📦 YOLOv9: {'✓' if YOLOV9_WEIGHTS.exists() else '✗'} {YOLOV9_WEIGHTS}")
print(f"🔧 YOLOv9 detector: {YOLOV9_DETECTOR}")
print(f"🔧 Device: {get_device()}")
print("=" * 50)

models = get_models()

# Pre-load YOLOv5
if YOLOV5_WEIGHTS.exists():
    print("⏳ Pre-loading YOLOv5...")
    try:
        load_yolov5()
    except Exception as e:
        print(f"⚠️ YOLOv5 load failed: {e}")

# ============== UI ==============
with gr.Blocks(title="ANPR - License Plate Detection") as app:
    gr.Markdown("""
    # 🚗 License Plate Detection (ANPR)
    ### YOLOv5 + YOLOv9 + EasyOCR
    
    Select model and upload video/image to detect plates.
    
    ---
    """)
    
    with gr.Tabs():
        with gr.TabItem("🎬 Video"):
            with gr.Row():
                with gr.Column():
                    video_in = gr.Video(label="📤 Upload Video")
                    model_vid = gr.Radio(models, value=models[0], label="🤖 Model")
                    conf_vid = gr.Slider(0.05, 0.9, 0.15, 0.05, label="🎯 Confidence")
                    btn_vid = gr.Button("🔍 Detect Plates", variant="primary", size="lg")
                with gr.Column():
                    video_out = gr.Video(label="📹 Result")
                    txt_vid = gr.Markdown()
            btn_vid.click(process_video, [video_in, model_vid, conf_vid], [video_out, txt_vid])
        
        with gr.TabItem("🖼️ Image"):
            with gr.Row():
                with gr.Column():
                    img_in = gr.Image(label="📤 Upload Image", type="numpy")
                    model_img = gr.Radio(models, value=models[0], label="🤖 Model")
                    conf_img = gr.Slider(0.05, 0.9, 0.15, 0.05, label="🎯 Confidence")
                    btn_img = gr.Button("🔍 Detect Plates", variant="primary", size="lg")
                with gr.Column():
                    img_out = gr.Image(label="🎯 Result")
                    txt_img = gr.Markdown()
            btn_img.click(process_image, [img_in, model_img, conf_img], [img_out, txt_img])
    
    gr.Markdown("""
    ---
    | Model | Speed | Method |
    |-------|-------|--------|
    | 🟢 **YOLOv5** | Fast ⚡ | Main process |
    | 🟠 **YOLOv9** | Slower | Subprocess (isolated) |
    
    *Computer Vision Project - SEM3*
    """)


if __name__ == "__main__":
    print("\n" + "=" * 50)
    print("📍 Open: http://localhost:7860")
    print("=" * 50 + "\n")
    app.launch(server_name="0.0.0.0", server_port=7860)

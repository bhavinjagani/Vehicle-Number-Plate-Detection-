"""
ANPR Web Application - Vehicle Number Plate Detection UI
Fast version with cached models and optimized processing.
"""

import gradio as gr
import cv2
import numpy as np
import tempfile
import sys
import os
from pathlib import Path
import torch
import easyocr
from collections import defaultdict

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
YOLOV5_PATH = PROJECT_ROOT / "external" / "yolov5"
YOLOV9_PATH = PROJECT_ROOT / "external" / "yolov9"
YOLOV5_WEIGHTS = PROJECT_ROOT / "runs" / "train_yolov5" / "exp" / "weights" / "best.pt"
YOLOV9_WEIGHTS = PROJECT_ROOT / "runs" / "train_yolov9" / "exp" / "weights" / "best.pt"

# Global cached models
cached_models = {}
ocr_reader = None
current_model_type = None


def get_device():
    """Detect best available device"""
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def clear_modules():
    """Clear YOLO-related modules from cache"""
    mods_to_clear = [k for k in list(sys.modules.keys()) 
                     if k.startswith(('models.', 'utils.')) and 'gradio' not in k.lower()]
    for m in mods_to_clear:
        try:
            del sys.modules[m]
        except:
            pass


def load_model(model_type):
    """Load and cache model - only reloads if model type changes"""
    global cached_models, current_model_type
    
    # Return cached model if same type
    if model_type in cached_models:
        return cached_models[model_type]
    
    # Clear other models to free memory and avoid conflicts
    if current_model_type and current_model_type != model_type:
        cached_models.clear()
        clear_modules()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    print(f"📦 Loading {model_type} model...")
    
    if "YOLOv5" in model_type:
        yolo_path = str(YOLOV5_PATH)
        weights_path = str(YOLOV5_WEIGHTS)
    else:
        yolo_path = str(YOLOV9_PATH)
        weights_path = str(YOLOV9_WEIGHTS)
    
    # Add to path
    if yolo_path not in sys.path:
        sys.path.insert(0, yolo_path)
    
    model = torch.hub.load(
        yolo_path,
        'custom',
        path=weights_path,
        source='local',
        force_reload=True
    )
    model.conf = 0.15
    model.iou = 0.45
    
    # Cache model
    cached_models[model_type] = model
    current_model_type = model_type
    print(f"✅ {model_type} loaded!")
    
    return model


def load_ocr():
    """Load OCR reader"""
    global ocr_reader
    if ocr_reader is None:
        print("📝 Loading EasyOCR...")
        ocr_reader = easyocr.Reader(['en'], gpu=torch.cuda.is_available())
        print("✅ OCR loaded!")
    return ocr_reader


def get_available_models():
    """Check which models are available"""
    models = []
    if YOLOV5_WEIGHTS.exists():
        models.append("🟢 YOLOv5")
    if YOLOV9_WEIGHTS.exists():
        models.append("🟠 YOLOv9")
    return models if models else ["No models found"]


def clean_plate_text(text):
    """Clean and validate plate text"""
    if not text:
        return ""
    cleaned = ''.join(c for c in text.upper() if c.isalnum() or c == ' ')
    cleaned = ' '.join(cleaned.split())
    return cleaned if len(cleaned) >= 2 else ""


def run_ocr_fast(plate_img, ocr):
    """Fast OCR - single pass"""
    if plate_img is None or plate_img.size == 0:
        return ""
    try:
        # Resize if too small
        h, w = plate_img.shape[:2]
        if w < 80:
            scale = 120 / w
            plate_img = cv2.resize(plate_img, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
        
        result = ocr.readtext(plate_img, detail=0, paragraph=True)
        return clean_plate_text(' '.join(result)) if result else ""
    except:
        return ""


def process_frame_fast(frame, model, ocr, conf_threshold, color):
    """Fast frame processing - minimal overhead"""
    detections = []
    h, w = frame.shape[:2]
    
    # Run detection directly
    model.conf = conf_threshold
    results = model(frame)
    preds = results.xyxy[0].cpu().numpy()
    
    for pred in preds:
        x1, y1, x2, y2, conf, cls = pred[:6]
        if conf < conf_threshold:
            continue
        
        # Convert to int with padding
        pad = 3
        x1, y1 = max(0, int(x1) - pad), max(0, int(y1) - pad)
        x2, y2 = min(w, int(x2) + pad), min(h, int(y2) + pad)
        
        # Crop and OCR
        plate_img = frame[y1:y2, x1:x2]
        plate_text = run_ocr_fast(plate_img, ocr)
        
        detections.append({
            'bbox': (x1, y1, x2, y2),
            'conf': float(conf),
            'text': plate_text
        })
        
        # Draw - simple and fast
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        label = plate_text if plate_text else "Plate"
        cv2.rectangle(frame, (x1, y1 - 20), (x1 + len(label) * 10, y1), color, -1)
        cv2.putText(frame, label, (x1 + 2, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    
    return frame, detections


def process_video(video_path, model_choice, conf_threshold, progress=gr.Progress()):
    """Optimized video processing"""
    if video_path is None:
        return None, "⚠️ Please upload a video first!"
    
    model_name = "YOLOv5" if "YOLOv5" in model_choice else "YOLOv9"
    model_emoji = "🟢" if "YOLOv5" in model_choice else "🟠"
    color = (0, 255, 0) if "YOLOv5" in model_choice else (0, 165, 255)
    
    try:
        # Load model ONCE
        model = load_model(model_choice)
        ocr = load_ocr()
        
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
        all_detections = []
        plate_tracker = defaultdict(int)
        last_detections = []  # Cache last detection for non-processed frames
        
        # Process every Nth frame (adjust based on video length)
        skip_frames = max(1, min(5, total_frames // 100))  # 1-5 based on video length
        
        print(f"⚡ Processing with {model_name} (every {skip_frames} frames)...")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Update progress less frequently
            if frame_count % 10 == 0:
                progress(frame_count / total_frames, desc=f"Frame {frame_count}/{total_frames}")
            
            # Process only every Nth frame
            if frame_count % skip_frames == 0:
                try:
                    annotated, detections = process_frame_fast(frame.copy(), model, ocr, conf_threshold, color)
                    last_detections = detections
                except Exception as e:
                    annotated = frame.copy()
                    detections = last_detections
            else:
                # Use cached detections for skipped frames
                annotated = frame.copy()
                detections = last_detections
                
                # Redraw cached boxes
                for det in detections:
                    x1, y1, x2, y2 = det['bbox']
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                    label = det['text'] if det['text'] else "Plate"
                    cv2.rectangle(annotated, (x1, y1 - 20), (x1 + len(label) * 10, y1), color, -1)
                    cv2.putText(annotated, label, (x1 + 2, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            
            # Track plates
            for det in detections:
                if det['text']:
                    plate_tracker[det['text']] += 1
            all_detections.extend(detections)
            
            # Simple HUD
            cv2.rectangle(annotated, (5, 5), (200, 60), (0, 0, 0), -1)
            cv2.putText(annotated, f"{model_name}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            cv2.putText(annotated, f"Frame {frame_count}/{total_frames}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            out.write(annotated)
        
        cap.release()
        out.release()
        
        # Summary
        unique_plates = sorted(plate_tracker.items(), key=lambda x: x[1], reverse=True)
        
        summary = f"""
## ✅ Done! 

**Model:** {model_emoji} {model_name}  
**Video:** {total_frames} frames @ {fps} FPS  
**Skip:** Every {skip_frames} frame(s)

### 📊 Results
- **Detections:** {len(all_detections)}
- **Unique Plates:** {len(unique_plates)}

### 🚗 Plates:
"""
        for plate, count in unique_plates[:10]:
            summary += f"\n- **`{plate}`** ({count}×)"
        
        if not unique_plates:
            summary += "\n⚠️ No plates found. Lower confidence?"
        
        return output_path, summary
        
    except Exception as e:
        import traceback
        return None, f"❌ Error: {e}\n```\n{traceback.format_exc()}\n```"


def process_image(image, model_choice, conf_threshold):
    """Process single image"""
    if image is None:
        return None, "⚠️ Please upload an image!"
    
    model_name = "YOLOv5" if "YOLOv5" in model_choice else "YOLOv9"
    model_emoji = "🟢" if "YOLOv5" in model_choice else "🟠"
    color = (0, 255, 0) if "YOLOv5" in model_choice else (0, 165, 255)
    
    try:
        model = load_model(model_choice)
        ocr = load_ocr()
        
        frame = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        annotated, detections = process_frame_fast(frame, model, ocr, conf_threshold, color)
        result = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        
        summary = f"## {model_emoji} {model_name}\n\n**Found:** {len(detections)} plate(s)\n\n"
        
        if detections:
            for i, det in enumerate(detections, 1):
                plate = det['text'] if det['text'] else "(unreadable)"
                summary += f"{i}. **`{plate}`** — {det['conf']:.0%}\n"
        else:
            summary += "⚠️ No plates found."
        
        return result, summary
        
    except Exception as e:
        import traceback
        return None, f"❌ Error: {e}\n```\n{traceback.format_exc()}\n```"


# Startup
print("=" * 50)
print("🚀 ANPR Web App - Fast Edition")
print(f"📦 YOLOv5: {'✓' if YOLOV5_WEIGHTS.exists() else '✗'}")
print(f"📦 YOLOv9: {'✓' if YOLOV9_WEIGHTS.exists() else '✗'}")
print(f"🔧 Device: {get_device()}")
print("=" * 50)

available_models = get_available_models()

# Pre-load first model for faster first inference
if available_models and "No models" not in available_models[0]:
    print("⏳ Pre-loading model...")
    try:
        load_model(available_models[0])
    except Exception as e:
        print(f"⚠️ Pre-load failed: {e}")

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
                    video_input = gr.Video(label="Upload Video")
                    model_vid = gr.Radio(available_models, value=available_models[0], label="Model")
                    conf_vid = gr.Slider(0.05, 0.9, 0.15, 0.05, label="Confidence")
                    btn_vid = gr.Button("🔍 Detect", variant="primary")
                with gr.Column():
                    video_output = gr.Video(label="Result")
                    text_vid = gr.Markdown()
            
            btn_vid.click(process_video, [video_input, model_vid, conf_vid], [video_output, text_vid])
        
        with gr.TabItem("🖼️ Image"):
            with gr.Row():
                with gr.Column():
                    image_input = gr.Image(label="Upload Image", type="numpy")
                    model_img = gr.Radio(available_models, value=available_models[0], label="Model")
                    conf_img = gr.Slider(0.05, 0.9, 0.15, 0.05, label="Confidence")
                    btn_img = gr.Button("🔍 Detect", variant="primary")
                with gr.Column():
                    image_output = gr.Image(label="Result")
                    text_img = gr.Markdown()
            
            btn_img.click(process_image, [image_input, model_img, conf_img], [image_output, text_img])
    
    gr.Markdown("""
    ---
    **🟢 YOLOv5** — Fast & stable | **🟠 YOLOv9** — Newer architecture
    
    *⚡ Optimized: Model cached, frames skipped for speed*
    """)


if __name__ == "__main__":
    print("\n📍 http://localhost:7860\n")
    app.launch(server_name="0.0.0.0", server_port=7860)

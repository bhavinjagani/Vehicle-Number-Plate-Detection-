"""
ANPR Web Application - Vehicle Number Plate Detection UI
Upload a video or image and detect license plates!
Supports YOLOv5 and YOLOv9 models with subprocess isolation.
"""

import gradio as gr
import cv2
import numpy as np
import tempfile
import sys
import os
import json
import subprocess
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

# Global OCR reader
ocr_reader = None


def get_device():
    """Detect best available device"""
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


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


def preprocess_plate_for_ocr(plate_img):
    """Enhanced preprocessing for better OCR results"""
    if plate_img is None or plate_img.size == 0:
        return None
    try:
        h, w = plate_img.shape[:2]
        if w < 100:
            scale = 200 / w
            plate_img = cv2.resize(plate_img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        
        if len(plate_img.shape) == 3:
            gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
        else:
            gray = plate_img
        
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        denoised = cv2.fastNlMeansDenoising(enhanced, None, 10, 7, 21)
        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        sharpened = cv2.filter2D(denoised, -1, kernel)
        return sharpened
    except:
        return None


def clean_plate_text(text):
    """Clean and validate plate text"""
    if not text:
        return ""
    cleaned = ''.join(c for c in text.upper() if c.isalnum() or c == ' ')
    cleaned = ' '.join(cleaned.split())
    return cleaned if len(cleaned) >= 2 else ""


def run_ocr_on_plate(plate_img, ocr):
    """Run OCR with multiple preprocessing attempts"""
    if plate_img is None or plate_img.size == 0:
        return ""
    
    results = []
    try:
        ocr_result = ocr.readtext(plate_img, detail=0, paragraph=True)
        if ocr_result:
            results.extend(ocr_result)
    except:
        pass
    
    preprocessed = preprocess_plate_for_ocr(plate_img)
    if preprocessed is not None:
        try:
            ocr_result = ocr.readtext(preprocessed, detail=0, paragraph=True)
            if ocr_result:
                results.extend(ocr_result)
        except:
            pass
    
    return clean_plate_text(' '.join(results))


def enhance_frame(frame):
    """Enhance frame for better detection"""
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    enhanced = cv2.merge([l, a, b])
    enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
    return enhanced


def run_detection_subprocess(image_path, model_type, conf_threshold):
    """Run detection in a subprocess to avoid module conflicts"""
    
    # Create detection script based on model type
    if "YOLOv5" in model_type:
        yolo_path = str(YOLOV5_PATH)
        weights_path = str(YOLOV5_WEIGHTS)
    else:
        yolo_path = str(YOLOV9_PATH)
        weights_path = str(YOLOV9_WEIGHTS)
    
    # Python script to run detection
    detect_script = f'''
import sys
import json
sys.path.insert(0, r"{yolo_path}")
import torch

model = torch.hub.load(
    r"{yolo_path}",
    'custom',
    path=r"{weights_path}",
    source='local',
    force_reload=True
)
model.conf = {conf_threshold}
model.iou = 0.45

results = model(r"{image_path}")
preds = results.xyxy[0].cpu().numpy().tolist()
print(json.dumps(preds))
'''
    
    try:
        # Run in subprocess with clean environment
        result = subprocess.run(
            [sys.executable, '-c', detect_script],
            capture_output=True,
            text=True,
            timeout=60,
            cwd=str(PROJECT_ROOT)
        )
        
        if result.returncode == 0 and result.stdout.strip():
            # Parse JSON output
            preds = json.loads(result.stdout.strip().split('\n')[-1])
            return preds
        else:
            print(f"Detection subprocess error: {result.stderr}")
            return []
    except Exception as e:
        print(f"Subprocess error: {e}")
        return []


def process_frame(frame, model_type, ocr, conf_threshold=0.15):
    """Process a single frame using subprocess detection"""
    detections = []
    h, w = frame.shape[:2]
    
    # Enhance frame
    enhanced = enhance_frame(frame)
    
    # Save frame temporarily
    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as f:
        temp_path = f.name
        cv2.imwrite(temp_path, enhanced)
    
    try:
        # Run detection in subprocess
        preds = run_detection_subprocess(temp_path, model_type, conf_threshold)
        
        # Colors based on model
        if "YOLOv5" in model_type:
            color = (0, 255, 0)  # Green
        else:
            color = (0, 165, 255)  # Orange
        
        for pred in preds:
            x1, y1, x2, y2, conf, cls = pred[:6]
            if conf < conf_threshold:
                continue
            
            # Add padding
            pad = 5
            x1 = max(0, int(x1) - pad)
            y1 = max(0, int(y1) - pad)
            x2 = min(w, int(x2) + pad)
            y2 = min(h, int(y2) + pad)
            
            # Crop plate and run OCR
            plate_img = frame[y1:y2, x1:x2]
            plate_text = run_ocr_on_plate(plate_img, ocr)
            
            detections.append({
                'bbox': (x1, y1, x2, y2),
                'conf': float(conf),
                'text': plate_text
            })
            
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
            
            # Label
            label = plate_text if plate_text else "Plate"
            conf_label = f"{conf:.0%}"
            
            (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.rectangle(frame, (x1, y1 - text_h - 10), (x1 + text_w + 10, y1), color, -1)
            cv2.putText(frame, label, (x1 + 5, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
            
            # Confidence badge
            (conf_w, conf_h), _ = cv2.getTextSize(conf_label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(frame, (x2 - conf_w - 10, y1), (x2, y1 - conf_h - 8), (255, 200, 0), -1)
            cv2.putText(frame, conf_label, (x2 - conf_w - 5, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    
    finally:
        # Cleanup temp file
        try:
            os.unlink(temp_path)
        except:
            pass
    
    return frame, detections


def process_video(video_path, model_choice, conf_threshold, progress=gr.Progress()):
    """Process video with selected model"""
    if video_path is None:
        return None, "⚠️ Please upload a video first!"
    
    model_name = "YOLOv5" if "YOLOv5" in model_choice else "YOLOv9"
    model_emoji = "🟢" if "YOLOv5" in model_choice else "🟠"
    
    try:
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
        
        # Process every Nth frame for speed (skip frames in between)
        process_every = 3  # Process every 3rd frame
        
        print(f"Processing with {model_name}...")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            progress(frame_count / total_frames, desc=f"Processing frame {frame_count}/{total_frames}")
            
            # Process only every Nth frame for speed
            if frame_count % process_every == 0:
                try:
                    annotated, detections = process_frame(frame.copy(), model_choice, ocr, conf_threshold)
                except Exception as e:
                    print(f"Frame {frame_count} error: {e}")
                    annotated = frame.copy()
                    detections = []
            else:
                annotated = frame.copy()
                detections = []
            
            # Track plates
            for det in detections:
                if det['text']:
                    plate_tracker[det['text']] += 1
                all_detections.append(det)
            
            # HUD overlay
            overlay = annotated.copy()
            cv2.rectangle(overlay, (0, 0), (width, 90), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.6, annotated, 0.4, 0, annotated)
            
            color = (0, 255, 0) if "YOLOv5" in model_choice else (0, 165, 255)
            cv2.putText(annotated, f"{model_name} ANPR", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            cv2.putText(annotated, f"Frame: {frame_count}/{total_frames}", (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(annotated, f"Plates: {len(detections)}", (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            if detections and detections[0]['text']:
                cv2.putText(annotated, f"Plate: {detections[0]['text']}", (width - 350, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            
            out.write(annotated)
        
        cap.release()
        out.release()
        
        # Summary
        unique_plates = sorted(plate_tracker.items(), key=lambda x: x[1], reverse=True)
        
        summary = f"""
## ✅ Processing Complete!

**Model:** {model_emoji} {model_name}  
**Video:** {total_frames} frames, {width}×{height} @ {fps} FPS  
**Confidence:** {conf_threshold:.0%}

### 📊 Results
- **Total Detections:** {len(all_detections)}
- **Unique Plates:** {len(unique_plates)}

### 🚗 Plates Found:
"""
        for plate, count in unique_plates[:10]:
            summary += f"\n- **`{plate}`** (detected {count}×)"
        
        if not unique_plates:
            summary += "\n\n⚠️ No plates detected. Try lowering confidence threshold."
        
        return output_path, summary
        
    except Exception as e:
        import traceback
        return None, f"❌ Error: {e}\n\n```\n{traceback.format_exc()}\n```"


def process_image(image, model_choice, conf_threshold):
    """Process image with selected model"""
    if image is None:
        return None, "⚠️ Please upload an image!"
    
    model_name = "YOLOv5" if "YOLOv5" in model_choice else "YOLOv9"
    model_emoji = "🟢" if "YOLOv5" in model_choice else "🟠"
    
    try:
        ocr = load_ocr()
        
        # Convert RGB to BGR
        frame = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        annotated, detections = process_frame(frame, model_choice, ocr, conf_threshold)
        
        # Convert back to RGB
        result = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        
        summary = f"## {model_emoji} {model_name} Detection\n\n**Found:** {len(detections)} plate(s)\n\n"
        
        if detections:
            summary += "### 🚗 Detected Plates:\n"
            for i, det in enumerate(detections, 1):
                plate = det['text'] if det['text'] else "(text unreadable)"
                summary += f"\n{i}. **`{plate}`** — Confidence: {det['conf']:.0%}"
        else:
            summary += "⚠️ No plates detected. Try lowering confidence threshold."
        
        return result, summary
        
    except Exception as e:
        import traceback
        return None, f"❌ Error: {e}\n\n```\n{traceback.format_exc()}\n```"


# Build UI
print("=" * 50)
print("🚀 Starting ANPR Web Application...")
print(f"📦 YOLOv5: {YOLOV5_WEIGHTS} ({'✓' if YOLOV5_WEIGHTS.exists() else '✗'})")
print(f"📦 YOLOv9: {YOLOV9_WEIGHTS} ({'✓' if YOLOV9_WEIGHTS.exists() else '✗'})")
print(f"🔧 Device: {get_device()}")
print("=" * 50)

available_models = get_available_models()

with gr.Blocks(title="ANPR - License Plate Detection") as app:
    gr.Markdown("""
    # 🚗 License Plate Detection (ANPR)
    ### YOLOv5 / YOLOv9 + EasyOCR | Computer Vision Project
    
    Upload a **video** or **image** to detect and read license plates automatically.
    
    ---
    """)
    
    with gr.Tabs():
        # Video Tab
        with gr.TabItem("🎬 Video Detection"):
            with gr.Row():
                with gr.Column(scale=1):
                    video_input = gr.Video(label="📤 Upload Video")
                    model_choice_vid = gr.Radio(
                        choices=available_models,
                        value=available_models[0] if available_models else None,
                        label="🤖 Select Model"
                    )
                    conf_slider_vid = gr.Slider(
                        minimum=0.05, 
                        maximum=0.9, 
                        value=0.15, 
                        step=0.05,
                        label="🎯 Confidence Threshold",
                        info="Lower = more detections, Higher = more accurate"
                    )
                    detect_btn_vid = gr.Button("🔍 Detect Plates", variant="primary", size="lg")
                
                with gr.Column(scale=1):
                    video_output = gr.Video(label="📹 Processed Video")
                    result_text_vid = gr.Markdown(label="Results")
            
            detect_btn_vid.click(
                process_video, 
                inputs=[video_input, model_choice_vid, conf_slider_vid], 
                outputs=[video_output, result_text_vid]
            )
        
        # Image Tab
        with gr.TabItem("🖼️ Image Detection"):
            with gr.Row():
                with gr.Column(scale=1):
                    image_input = gr.Image(label="📤 Upload Image", type="numpy")
                    model_choice_img = gr.Radio(
                        choices=available_models,
                        value=available_models[0] if available_models else None,
                        label="🤖 Select Model"
                    )
                    conf_slider_img = gr.Slider(
                        minimum=0.05, 
                        maximum=0.9, 
                        value=0.15, 
                        step=0.05,
                        label="🎯 Confidence Threshold"
                    )
                    detect_btn_img = gr.Button("🔍 Detect Plates", variant="primary", size="lg")
                
                with gr.Column(scale=1):
                    image_output = gr.Image(label="🎯 Detection Result")
                    result_text_img = gr.Markdown(label="Results")
            
            detect_btn_img.click(
                process_image, 
                inputs=[image_input, model_choice_img, conf_slider_img], 
                outputs=[image_output, result_text_img]
            )
    
    gr.Markdown("""
    ---
    ### 📋 Model Info:
    - **🟢 YOLOv5** — Fast, well-tested architecture  
    - **🟠 YOLOv9** — Newer architecture with GELAN
    
    **Supported Formats:** MP4, AVI, MOV | JPG, PNG, BMP, WEBP
    
    *Computer Vision Project - SEM3*
    """)


if __name__ == "__main__":
    print("\n" + "=" * 50)
    print("📍 Open in browser: http://localhost:7860")
    print("=" * 50 + "\n")
    app.launch(server_name="0.0.0.0", server_port=7860)

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

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent

# Global OCR reader
ocr_reader = None

# Model paths
YOLOV5_WEIGHTS = PROJECT_ROOT / "runs" / "train_yolov5" / "exp" / "weights" / "best.pt"
YOLOV9_WEIGHTS = PROJECT_ROOT / "runs" / "train_yolov9" / "exp" / "weights" / "best.pt"


def get_device():
    """Detect best available device"""
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_yolov5_model():
    """Load YOLOv5 model in isolated context"""
    # Clear any conflicting modules
    mods_to_clear = [k for k in sys.modules.keys() if k.startswith(('models.', 'utils.')) and 'gradio' not in k]
    for m in mods_to_clear:
        del sys.modules[m]
    
    # Set up path
    yolov5_path = str(PROJECT_ROOT / "external" / "yolov5")
    if yolov5_path not in sys.path:
        sys.path.insert(0, yolov5_path)
    
    # Import YOLOv5 specific modules
    from models.common import DetectMultiBackend
    from utils.general import non_max_suppression, scale_boxes
    from utils.torch_utils import select_device
    
    device = select_device(get_device())
    model = DetectMultiBackend(str(YOLOV5_WEIGHTS), device=device, fp16=False)
    model.warmup(imgsz=(1, 3, 640, 640))
    
    return model, device


def load_ocr():
    """Load OCR reader"""
    global ocr_reader
    if ocr_reader is None:
        print("Loading EasyOCR...")
        ocr_reader = easyocr.Reader(['en'], gpu=torch.cuda.is_available())
        print("✓ OCR loaded!")
    return ocr_reader


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


def run_ocr_on_plate(plate_img, ocr_reader):
    """Run OCR with multiple preprocessing attempts"""
    if plate_img is None or plate_img.size == 0:
        return ""
    
    results = []
    
    try:
        ocr_result = ocr_reader.readtext(plate_img, detail=0, paragraph=True)
        if ocr_result:
            results.extend(ocr_result)
    except:
        pass
    
    preprocessed = preprocess_plate_for_ocr(plate_img)
    if preprocessed is not None:
        try:
            ocr_result = ocr_reader.readtext(preprocessed, detail=0, paragraph=True)
            if ocr_result:
                results.extend(ocr_result)
        except:
            pass
    
    all_text = ' '.join(results)
    return clean_plate_text(all_text)


def run_yolov5_detection(frame, conf_threshold=0.15):
    """Run YOLOv5 detection using torch.hub"""
    # Use torch.hub with YOLOv5
    yolov5_path = str(PROJECT_ROOT / "external" / "yolov5")
    
    # Clear modules
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
    model.conf = conf_threshold
    model.iou = 0.45
    
    results = model(frame)
    return results.xyxy[0].cpu().numpy()


def run_yolov9_detection(frame, conf_threshold=0.15):
    """Run YOLOv9 detection via subprocess to avoid module conflicts"""
    import tempfile
    import os
    
    # Save frame to temp file
    temp_img = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
    temp_path = temp_img.name
    temp_img.close()
    
    try:
        # Save the frame as image
        cv2.imwrite(temp_path, frame)
        
        # Run YOLOv9 in separate process
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
            timeout=60
        )
        
        # Parse output
        for line in result.stdout.split('\n'):
            if line.startswith("RESULT:"):
                json_str = line[7:]  # Remove "RESULT:" prefix
                preds = json.loads(json_str)
                return np.array(preds) if preds else np.array([])
        
        # If no result found, check stderr
        if result.stderr:
            print(f"YOLOv9 stderr: {result.stderr}")
        
        return np.array([])
        
    except subprocess.TimeoutExpired:
        print("YOLOv9 detection timed out")
        return np.array([])
    except Exception as e:
        print(f"YOLOv9 detection error: {e}")
        return np.array([])
    finally:
        # Cleanup temp file
        if os.path.exists(temp_path):
            os.remove(temp_path)


def process_frame_simple(frame, model_name, ocr_reader, conf_threshold=0.15):
    """Process a single frame - simplified version"""
    detections = []
    
    # Enhance frame
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    enhanced = cv2.merge([l, a, b])
    enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
    
    # Run detection based on model
    if model_name == "yolov5":
        preds = run_yolov5_detection(enhanced, conf_threshold)
    else:
        preds = run_yolov9_detection(enhanced, conf_threshold)
    
    h, w = frame.shape[:2]
    
    for pred in preds:
        x1, y1, x2, y2, conf, cls = pred
        if conf < conf_threshold:
            continue
        
        pad = 5
        x1 = max(0, int(x1) - pad)
        y1 = max(0, int(y1) - pad)
        x2 = min(w, int(x2) + pad)
        y2 = min(h, int(y2) + pad)
        
        plate_img = frame[y1:y2, x1:x2]
        plate_text = run_ocr_on_plate(plate_img, ocr_reader)
        
        detections.append({
            'bbox': (x1, y1, x2, y2),
            'conf': conf,
            'text': plate_text
        })
        
        color = (0, 255, 0) if model_name == "yolov5" else (0, 165, 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
        
        label = plate_text if plate_text else "Plate"
        conf_label = f"{conf:.0%}"
        
        (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(frame, (x1, y1 - text_h - 10), (x1 + text_w + 10, y1), color, -1)
        cv2.putText(frame, label, (x1 + 5, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        
        (conf_w, conf_h), _ = cv2.getTextSize(conf_label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x2 - conf_w - 10, y1), (x2, y1 - conf_h - 8), (255, 200, 0), -1)
        cv2.putText(frame, conf_label, (x2 - conf_w - 5, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    
    return frame, detections


def process_video(video_path, model_choice, conf_threshold):
    """Process video with selected model"""
    if video_path is None:
        return None, "⚠️ Please upload a video first!"
    
    model_name = "yolov5" if "YOLOv5" in model_choice else "yolov9"
    
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
        
        print(f"Processing with {model_name.upper()}...")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            try:
                annotated, detections = process_frame_simple(frame.copy(), model_name, ocr, conf_threshold)
            except Exception as e:
                print(f"Frame {frame_count} error: {e}")
                annotated = frame.copy()
                detections = []
            
            for det in detections:
                if det['text']:
                    plate_tracker[det['text']] += 1
                all_detections.append(det)
            
            # Overlay
            overlay = annotated.copy()
            cv2.rectangle(overlay, (0, 0), (width, 90), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.6, annotated, 0.4, 0, annotated)
            
            model_color = (0, 255, 0) if model_name == "yolov5" else (0, 165, 255)
            cv2.putText(annotated, f"{model_name.upper()}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, model_color, 2)
            cv2.putText(annotated, f"Frame: {frame_count}/{total_frames}", (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(annotated, f"Plates: {len(detections)}", (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            if detections and detections[0]['text']:
                cv2.putText(annotated, f"Plate: {detections[0]['text']}", (width - 350, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            
            out.write(annotated)
            
            if frame_count % 30 == 0:
                print(f"Processed {frame_count}/{total_frames}")
        
        cap.release()
        out.release()
        
        unique_plates = sorted(plate_tracker.items(), key=lambda x: x[1], reverse=True)
        
        emoji = "🟢" if model_name == "yolov5" else "🟠"
        summary = f"""
## ✅ Done!

**Model:** {emoji} {model_name.upper()}
**Video:** {total_frames} frames, {width}x{height}, {fps} FPS

**Results:** {len(all_detections)} detections, {len(unique_plates)} unique plates

**Plates Found:**
"""
        for plate, count in unique_plates[:10]:
            summary += f"\n- `{plate}` (seen {count}x)"
        
        if not unique_plates:
            summary += "\n⚠️ No plates found. Try lower confidence."
        
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
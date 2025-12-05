#!/usr/bin/env python
"""
Unified inference pipeline for YOLOv5/YOLOv9 + EasyOCR.
Detects license plates and reads the text using OCR.

Supports: images, folders, webcam, RTSP streams, or video files.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from rich.console import Console
from rich.table import Table

console = Console()
ROOT = Path(__file__).resolve().parents[2]
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Detect license plates and read text with EasyOCR.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single image with YOLOv5
  python src/inference/detect_and_read.py --model yolov5 \\
      --weights runs/train_yolov5/exp/weights/best.pt \\
      --source test_image.jpg

  # Folder of images with YOLOv9
  python src/inference/detect_and_read.py --model yolov9 \\
      --weights runs/train_yolov9/exp/weights/best.pt \\
      --source data/processed/test/images

  # Webcam
  python src/inference/detect_and_read.py --model yolov5 \\
      --weights runs/train_yolov5/exp/weights/best.pt \\
      --source 0
        """
    )
    parser.add_argument("--model", choices=("yolov5", "yolov9"), required=True,
                        help="Model type (yolov5 or yolov9)")
    parser.add_argument("--weights", type=Path, required=True,
                        help="Path to trained .pt weights file")
    parser.add_argument("--source", required=True,
                        help="Image/video path, folder, or webcam ID (0)")
    parser.add_argument("--img-size", type=int, default=640,
                        help="Inference image size")
    parser.add_argument("--conf", type=float, default=0.25,
                        help="Detection confidence threshold")
    parser.add_argument("--device", default="",
                        help="Device: '', 0, 1, cpu, mps")
    parser.add_argument("--ocr-langs", nargs="+", default=["en"],
                        help="OCR languages (default: en)")
    parser.add_argument("--ocr-gpu", action="store_true",
                        help="Use GPU for EasyOCR")
    parser.add_argument("--save-crops", action="store_true",
                        help="Save cropped plate images")
    parser.add_argument("--save-txt", action="store_true",
                        help="Save results to text file")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "runs" / "detect_ocr",
                        help="Output directory")
    parser.add_argument("--view-img", action="store_true",
                        help="Display results in window")
    parser.add_argument("--no-ocr", action="store_true",
                        help="Skip OCR (detection only)")
    return parser.parse_args()


def get_device_string(device: str) -> str:
    """Normalize device string."""
    if device == "":
        # Auto-detect
        try:
            import torch
            if torch.cuda.is_available():
                return "0"
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return "mps"
        except ImportError:
            pass
        return "cpu"
    return device


def load_model(model_type: str, weights: Path, device: str):
    """Load YOLOv5 or YOLOv9 model."""
    import torch
    
    # Add repo to path
    if model_type == "yolov5":
        repo_path = ROOT / "external" / "yolov5"
    else:
        repo_path = ROOT / "external" / "yolov9"
    
    sys.path.insert(0, str(repo_path))
    
    # Import and load model
    from models.common import DetectMultiBackend
    from utils.general import check_img_size
    from utils.torch_utils import select_device
    
    device_obj = select_device(device)
    model = DetectMultiBackend(str(weights), device=device_obj, fp16=False)
    stride = model.stride
    imgsz = check_img_size(640, s=stride)
    
    # Warmup
    model.warmup(imgsz=(1, 3, imgsz, imgsz))
    
    return model, device_obj, stride, imgsz


def letterbox(img, new_shape=640, color=(114, 114, 114), auto=True, stride=32):
    """Resize and pad image while meeting stride-multiple constraints."""
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    
    # Compute padding
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    
    if auto:
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)
    
    dw /= 2
    dh /= 2
    
    if shape[::-1] != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    
    return img, r, (dw, dh)


def preprocess(img, imgsz, stride, device):
    """Preprocess image for inference."""
    import torch
    
    # Letterbox
    img_letterbox, ratio, pad = letterbox(img, imgsz, stride=stride)
    
    # Convert
    img_input = img_letterbox.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    img_input = np.ascontiguousarray(img_input)
    img_input = torch.from_numpy(img_input).to(device)
    img_input = img_input.float() / 255.0
    
    if img_input.ndimension() == 3:
        img_input = img_input.unsqueeze(0)
    
    return img_input, ratio, pad


def postprocess(pred, conf_thres, img_shape, orig_shape, ratio, pad):
    """Post-process predictions."""
    from utils.general import non_max_suppression, scale_boxes
    
    # NMS
    pred = non_max_suppression(pred, conf_thres, 0.45, max_det=100)
    
    detections = []
    for det in pred:
        if len(det):
            # Rescale boxes
            det[:, :4] = scale_boxes(img_shape[2:], det[:, :4], orig_shape).round()
            
            for *xyxy, conf, cls in det:
                x1, y1, x2, y2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])
                detections.append({
                    "bbox": [x1, y1, x2, y2],
                    "confidence": float(conf),
                    "class": int(cls)
                })
    
    return detections


def run_ocr(reader, crop: np.ndarray) -> Tuple[str, float]:
    """Run EasyOCR on a cropped plate image."""
    if crop.size == 0:
        return "", 0.0
    
    try:
        # Convert BGR to RGB
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        
        # Run OCR
        results = reader.readtext(rgb, detail=1)
        
        if not results:
            return "", 0.0
        
        # Get best result (highest confidence)
        best = max(results, key=lambda x: x[2])
        text = best[1].strip().upper()
        conf = float(best[2])
        
        # Clean up text - remove common OCR errors
        text = text.replace(" ", "").replace("-", "")
        
        return text, conf
    except Exception as e:
        console.log(f"[yellow]OCR error: {e}")
        return "", 0.0


def draw_results(img: np.ndarray, detections: List[Dict], ocr_results: List[Tuple[str, float]]) -> np.ndarray:
    """Draw bounding boxes and OCR text on image."""
    img_draw = img.copy()
    
    for det, (text, ocr_conf) in zip(detections, ocr_results):
        x1, y1, x2, y2 = det["bbox"]
        conf = det["confidence"]
        
        # Box color based on OCR success
        color = (0, 255, 0) if text else (0, 165, 255)  # Green if OCR success, orange otherwise
        
        # Draw box
        cv2.rectangle(img_draw, (x1, y1), (x2, y2), color, 2)
        
        # Build label
        label = f"{conf:.2f}"
        if text:
            label = f"{text} ({ocr_conf:.2f})"
        
        # Draw label background
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(img_draw, (x1, y1 - h - 10), (x1 + w + 4, y1), color, -1)
        
        # Draw label text
        cv2.putText(img_draw, label, (x1 + 2, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    return img_draw


def get_sources(source: str) -> List[Tuple[str, bool]]:
    """Get list of sources (path, is_video)."""
    sources = []
    
    # Check if webcam
    if source.isdigit():
        return [(int(source), True)]
    
    path = Path(source)
    
    if path.is_file():
        ext = path.suffix.lower()
        is_video = ext in VIDEO_EXTS
        return [(str(path), is_video)]
    
    if path.is_dir():
        # Get all images
        for ext in IMAGE_EXTS:
            sources.extend([(str(p), False) for p in sorted(path.glob(f"*{ext}"))])
            sources.extend([(str(p), False) for p in sorted(path.glob(f"*{ext.upper()}"))])
        return sources
    
    console.log(f"[red]Source not found: {source}")
    return []


def main() -> None:
    args = parse_args()
    
    # Validate weights
    if not args.weights.exists():
        console.log(f"[red]Weights not found: {args.weights}")
        sys.exit(1)
    
    # Setup output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = args.output_dir / f"{args.model}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    
    if args.save_crops:
        (run_dir / "crops").mkdir(exist_ok=True)
    
    console.rule(f"[bold green]License Plate Detection + OCR")
    console.log(f"Model: {args.model}")
    console.log(f"Weights: {args.weights}")
    console.log(f"Output: {run_dir}")
    
    # Get device
    device = get_device_string(args.device)
    console.log(f"Device: {device}")
    
    # Load YOLO model
    console.log("Loading YOLO model...")
    model, device_obj, stride, imgsz = load_model(args.model, args.weights, device)
    
    # Load EasyOCR
    reader = None
    if not args.no_ocr:
        console.log(f"Loading EasyOCR (languages: {args.ocr_langs})...")
        import easyocr
        reader = easyocr.Reader(args.ocr_langs, gpu=args.ocr_gpu)
    
    # Get sources
    sources = get_sources(args.source)
    if not sources:
        console.log("[red]No valid sources found!")
        sys.exit(1)
    
    console.log(f"Processing {len(sources)} source(s)...")
    
    # Results storage
    all_results = []
    
    # Process each source
    for source_path, is_video in sources:
        if is_video or isinstance(source_path, int):
            # Video/webcam processing
            cap = cv2.VideoCapture(source_path)
            frame_id = 0
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Preprocess
                img_input, ratio, pad = preprocess(frame, imgsz, stride, device_obj)
                
                # Inference
                pred = model(img_input)
                
                # Postprocess
                detections = postprocess(pred, args.conf, img_input.shape, frame.shape, ratio, pad)
                
                # OCR
                ocr_results = []
                for det in detections:
                    x1, y1, x2, y2 = det["bbox"]
                    crop = frame[y1:y2, x1:x2]
                    
                    if reader and not args.no_ocr:
                        text, conf = run_ocr(reader, crop)
                    else:
                        text, conf = "", 0.0
                    
                    ocr_results.append((text, conf))
                    
                    if args.save_crops and crop.size > 0:
                        crop_name = f"frame{frame_id:06d}_det{len(ocr_results)}.jpg"
                        cv2.imwrite(str(run_dir / "crops" / crop_name), crop)
                
                # Store results
                for det, (text, ocr_conf) in zip(detections, ocr_results):
                    all_results.append({
                        "source": str(source_path),
                        "frame_id": frame_id,
                        "bbox": det["bbox"],
                        "det_conf": det["confidence"],
                        "plate_text": text,
                        "ocr_conf": ocr_conf
                    })
                
                # Draw and display
                if detections:
                    img_result = draw_results(frame, detections, ocr_results)
                else:
                    img_result = frame
                
                if args.view_img:
                    cv2.imshow("License Plate Detection", img_result)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                
                frame_id += 1
            
            cap.release()
        
        else:
            # Image processing
            img = cv2.imread(source_path)
            if img is None:
                console.log(f"[yellow]Could not read: {source_path}")
                continue
            
            # Preprocess
            img_input, ratio, pad = preprocess(img, imgsz, stride, device_obj)
            
            # Inference
            pred = model(img_input)
            
            # Postprocess
            detections = postprocess(pred, args.conf, img_input.shape, img.shape, ratio, pad)
            
            # OCR
            ocr_results = []
            for i, det in enumerate(detections):
                x1, y1, x2, y2 = det["bbox"]
                crop = img[y1:y2, x1:x2]
                
                if reader and not args.no_ocr:
                    text, conf = run_ocr(reader, crop)
                else:
                    text, conf = "", 0.0
                
                ocr_results.append((text, conf))
                
                if args.save_crops and crop.size > 0:
                    crop_name = f"{Path(source_path).stem}_det{i}.jpg"
                    cv2.imwrite(str(run_dir / "crops" / crop_name), crop)
            
            # Store results
            for det, (text, ocr_conf) in zip(detections, ocr_results):
                all_results.append({
                    "source": source_path,
                    "frame_id": 0,
                    "bbox": det["bbox"],
                    "det_conf": det["confidence"],
                    "plate_text": text,
                    "ocr_conf": ocr_conf
                })
            
            # Draw results
            if detections:
                img_result = draw_results(img, detections, ocr_results)
            else:
                img_result = img
            
            # Save result image
            out_path = run_dir / Path(source_path).name
            cv2.imwrite(str(out_path), img_result)
            
            # Print detection info
            if detections:
                for det, (text, ocr_conf) in zip(detections, ocr_results):
                    plate_str = text if text else "(no text)"
                    console.log(f"[green]{Path(source_path).name}: {plate_str} (conf: {det['confidence']:.2f}, ocr: {ocr_conf:.2f})")
            
            if args.view_img:
                cv2.imshow("License Plate Detection", img_result)
                cv2.waitKey(0)
    
    cv2.destroyAllWindows()
    
    # Save results
    if all_results:
        # Save JSON
        with open(run_dir / "results.json", "w") as f:
            json.dump(all_results, f, indent=2)
        
        # Save CSV
        with open(run_dir / "results.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["source", "frame_id", "bbox", "det_conf", "plate_text", "ocr_conf"])
            writer.writeheader()
            writer.writerows(all_results)
        
        if args.save_txt:
            with open(run_dir / "plates.txt", "w") as f:
                for r in all_results:
                    if r["plate_text"]:
                        f.write(f"{r['plate_text']}\n")
    
    # Print summary
    console.rule("[bold green]Summary")
    
    table = Table(title="Detection Results")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    
    total_detections = len(all_results)
    plates_with_text = sum(1 for r in all_results if r["plate_text"])
    unique_plates = len(set(r["plate_text"] for r in all_results if r["plate_text"]))
    
    table.add_row("Total Detections", str(total_detections))
    table.add_row("Plates with OCR Text", str(plates_with_text))
    table.add_row("Unique Plates", str(unique_plates))
    table.add_row("Output Directory", str(run_dir))
    
    console.print(table)
    
    # Print detected plates
    if plates_with_text:
        console.print("\n[bold]Detected Plate Numbers:[/bold]")
        seen = set()
        for r in all_results:
            if r["plate_text"] and r["plate_text"] not in seen:
                console.print(f"  • {r['plate_text']} (conf: {r['ocr_conf']:.2f})")
                seen.add(r["plate_text"])


if __name__ == "__main__":
    main()

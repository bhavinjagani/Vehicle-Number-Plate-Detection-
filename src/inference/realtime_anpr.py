#!/usr/bin/env python
"""
Real-time Automatic Number Plate Recognition (ANPR) System

Performs vehicle number plate detection and recognition on:
- Video streams (MP4, AVI, etc.)
- Webcam feeds (live camera)
- Static images (JPG, PNG, etc.)
- RTSP/HTTP streams

Usage:
    # Webcam
    python src/inference/realtime_anpr.py --source 0
    
    # Video file
    python src/inference/realtime_anpr.py --source video.mp4
    
    # Image
    python src/inference/realtime_anpr.py --source image.jpg
    
    # Folder of images
    python src/inference/realtime_anpr.py --source images/
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

# Add project root to path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

# Rich console for pretty output
try:
    from rich.console import Console
    from rich.table import Table
    from rich.live import Live
    from rich.panel import Panel
    console = Console()
    HAS_RICH = True
except ImportError:
    HAS_RICH = False
    class Console:
        def log(self, msg): print(msg)
        def print(self, msg): print(msg)
        def rule(self, msg): print(f"\n{'='*50}\n{msg}\n{'='*50}")
    console = Console()


class ANPRSystem:
    """Automatic Number Plate Recognition System."""
    
    def __init__(
        self,
        model_type: str = "yolov5",
        weights: Optional[Path] = None,
        conf_threshold: float = 0.25,
        device: str = "",
        ocr_langs: List[str] = ["en"],
        use_gpu_ocr: bool = False
    ):
        self.model_type = model_type
        self.conf_threshold = conf_threshold
        self.device = device
        self.ocr_langs = ocr_langs
        
        # Default weights paths
        if weights is None:
            if model_type == "yolov5":
                weights = ROOT / "runs" / "train_yolov5" / "exp" / "weights" / "best.pt"
            else:
                weights = ROOT / "runs" / "train_yolov9" / "exp" / "weights" / "best.pt"
        
        self.weights = Path(weights)
        
        # Initialize components
        self.model = None
        self.ocr_reader = None
        self.device_obj = None
        self.stride = 32
        self.imgsz = 640
        
        # Detection history
        self.detected_plates: deque = deque(maxlen=50)
        self.fps_history: deque = deque(maxlen=30)
        
        # Stats
        self.frame_count = 0
        self.detection_count = 0
        self.start_time = None
        
        console.log(f"[cyan]Initializing ANPR System...")
        console.log(f"[cyan]Model: {model_type}")
        console.log(f"[cyan]Weights: {self.weights}")
        
    def load_model(self):
        """Load YOLO detection model."""
        import torch
        
        if not self.weights.exists():
            raise FileNotFoundError(f"Weights not found: {self.weights}")
        
        # Add repo to path
        if self.model_type == "yolov5":
            repo_path = ROOT / "external" / "yolov5"
        else:
            repo_path = ROOT / "external" / "yolov9"
        
        sys.path.insert(0, str(repo_path))
        
        from models.common import DetectMultiBackend
        from utils.general import check_img_size
        from utils.torch_utils import select_device
        
        # Select device
        if self.device == "":
            if torch.cuda.is_available():
                self.device = "0"
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        
        console.log(f"[green]Using device: {self.device}")
        
        self.device_obj = select_device(self.device)
        self.model = DetectMultiBackend(str(self.weights), device=self.device_obj, fp16=False)
        self.stride = self.model.stride
        self.imgsz = check_img_size(640, s=self.stride)
        
        # Warmup
        console.log("[cyan]Warming up model...")
        self.model.warmup(imgsz=(1, 3, self.imgsz, self.imgsz))
        console.log("[green]✓ Model loaded successfully!")
        
    def load_ocr(self):
        """Load EasyOCR reader."""
        console.log(f"[cyan]Loading EasyOCR (languages: {self.ocr_langs})...")
        import easyocr
        self.ocr_reader = easyocr.Reader(self.ocr_langs, gpu=False)
        console.log("[green]✓ OCR loaded successfully!")
    
    def preprocess(self, img: np.ndarray) -> Tuple[np.ndarray, float, Tuple[float, float]]:
        """Preprocess image for inference."""
        import torch
        
        # Letterbox resize
        shape = img.shape[:2]
        new_shape = (self.imgsz, self.imgsz)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
        dw, dh = np.mod(dw, self.stride), np.mod(dh, self.stride)
        dw /= 2
        dh /= 2
        
        if shape[::-1] != new_unpad:
            img_resized = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        else:
            img_resized = img
        
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img_padded = cv2.copyMakeBorder(img_resized, top, bottom, left, right, 
                                         cv2.BORDER_CONSTANT, value=(114, 114, 114))
        
        # Convert to tensor
        img_input = img_padded.transpose((2, 0, 1))[::-1]
        img_input = np.ascontiguousarray(img_input)
        img_input = torch.from_numpy(img_input).to(self.device_obj)
        img_input = img_input.float() / 255.0
        
        if img_input.ndimension() == 3:
            img_input = img_input.unsqueeze(0)
        
        return img_input, r, (dw, dh)
    
    def detect(self, img: np.ndarray) -> List[Dict]:
        """Run detection on image."""
        from utils.general import non_max_suppression, scale_boxes
        
        # Preprocess
        img_input, ratio, pad = self.preprocess(img)
        
        # Inference
        pred = self.model(img_input)
        
        # NMS
        pred = non_max_suppression(pred, self.conf_threshold, 0.45, max_det=100)
        
        detections = []
        for det in pred:
            if len(det):
                # Rescale boxes
                det[:, :4] = scale_boxes(img_input.shape[2:], det[:, :4], img.shape).round()
                
                for *xyxy, conf, cls in det:
                    x1, y1, x2, y2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])
                    detections.append({
                        "bbox": [x1, y1, x2, y2],
                        "confidence": float(conf),
                    })
        
        return detections
    
    def read_plate(self, crop: np.ndarray) -> Tuple[str, float]:
        """Read text from plate crop using OCR."""
        if crop.size == 0 or self.ocr_reader is None:
            return "", 0.0
        
        try:
            rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            results = self.ocr_reader.readtext(rgb, detail=1)
            
            if not results:
                return "", 0.0
            
            # Get best result
            best = max(results, key=lambda x: x[2])
            text = best[1].strip().upper().replace(" ", "").replace("-", "")
            conf = float(best[2])
            
            return text, conf
        except Exception:
            return "", 0.0
    
    def draw_results(self, frame: np.ndarray, detections: List[Dict], 
                     ocr_results: List[Tuple[str, float]], fps: float) -> np.ndarray:
        """Draw detection results on frame."""
        img = frame.copy()
        
        # Draw detections
        for det, (text, ocr_conf) in zip(detections, ocr_results):
            x1, y1, x2, y2 = det["bbox"]
            conf = det["confidence"]
            
            # Color based on OCR success
            if text and ocr_conf > 0.3:
                color = (0, 255, 0)  # Green - good OCR
            elif text:
                color = (0, 255, 255)  # Yellow - low confidence OCR
            else:
                color = (0, 165, 255)  # Orange - no OCR
            
            # Draw box
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            
            # Label
            if text:
                label = f"{text} ({ocr_conf:.0%})"
            else:
                label = f"Plate ({conf:.0%})"
            
            # Label background
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.rectangle(img, (x1, y1 - h - 10), (x1 + w + 4, y1), color, -1)
            cv2.putText(img, label, (x1 + 2, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        
        # Draw info panel
        self._draw_info_panel(img, fps, len(detections))
        
        return img
    
    def _draw_info_panel(self, img: np.ndarray, fps: float, num_detections: int):
        """Draw info panel on frame."""
        h, w = img.shape[:2]
        
        # Panel background
        panel_h = 90
        overlay = img.copy()
        cv2.rectangle(overlay, (10, 10), (300, panel_h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, img, 0.4, 0, img)
        
        # Text
        cv2.putText(img, f"ANPR System - {self.model_type.upper()}", (20, 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(img, f"FPS: {fps:.1f}", (20, 55),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(img, f"Detections: {num_detections}", (120, 55),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        cv2.putText(img, f"Total Plates: {len(self.detected_plates)}", (20, 75),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        
        # Recent plates
        if self.detected_plates:
            recent = list(self.detected_plates)[-3:]
            y_offset = h - 20
            cv2.putText(img, "Recent Plates:", (20, y_offset - 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            for i, plate in enumerate(reversed(recent)):
                cv2.putText(img, f"• {plate}", (20, y_offset - 40 + i*20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    
    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, List[Dict]]:
        """Process a single frame."""
        start = time.time()
        
        # Detect plates
        detections = self.detect(frame)
        
        # OCR on each detection
        ocr_results = []
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            crop = frame[y1:y2, x1:x2]
            text, conf = self.read_plate(crop)
            ocr_results.append((text, conf))
            
            # Store detected plates
            if text and conf > 0.3 and text not in self.detected_plates:
                self.detected_plates.append(text)
                self.detection_count += 1
        
        # Calculate FPS
        elapsed = time.time() - start
        fps = 1.0 / elapsed if elapsed > 0 else 0
        self.fps_history.append(fps)
        avg_fps = sum(self.fps_history) / len(self.fps_history)
        
        # Draw results
        result_frame = self.draw_results(frame, detections, ocr_results, avg_fps)
        
        self.frame_count += 1
        
        return result_frame, detections
    
    def run(self, source: str, output_dir: Optional[Path] = None, 
            save_video: bool = False, show: bool = True):
        """Run ANPR on source."""
        
        # Initialize
        self.load_model()
        self.load_ocr()
        self.start_time = time.time()
        
        # Setup output
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
        
        # Determine source type
        is_webcam = source.isdigit() or source.startswith("/dev/")
        is_video = Path(source).suffix.lower() in {".mp4", ".avi", ".mov", ".mkv", ".webm"}
        is_image = Path(source).suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
        is_folder = Path(source).is_dir()
        
        console.rule("[bold green]Starting ANPR System")
        
        if is_webcam:
            self._process_video(int(source), output_dir, save_video, show, is_webcam=True)
        elif is_video:
            self._process_video(source, output_dir, save_video, show)
        elif is_image:
            self._process_image(source, output_dir, show)
        elif is_folder:
            self._process_folder(source, output_dir, show)
        else:
            console.log(f"[red]Unknown source type: {source}")
            return
        
        # Print summary
        self._print_summary()
    
    def _process_video(self, source, output_dir, save_video, show, is_webcam=False):
        """Process video or webcam."""
        cap = cv2.VideoCapture(source)
        
        if not cap.isOpened():
            console.log(f"[red]Could not open video source: {source}")
            return
        
        # Video writer
        writer = None
        if save_video and output_dir:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            fps = cap.get(cv2.CAP_PROP_FPS) or 20
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            out_path = output_dir / f"output_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
            writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))
        
        source_name = "Webcam" if is_webcam else Path(source).name
        console.log(f"[cyan]Processing: {source_name}")
        console.log("[yellow]Press 'q' to quit, 's' to save screenshot")
        
        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    if is_webcam:
                        continue
                    break
                
                # Process frame
                result_frame, detections = self.process_frame(frame)
                
                # Save video
                if writer:
                    writer.write(result_frame)
                
                # Display
                if show:
                    cv2.imshow("ANPR System", result_frame)
                    key = cv2.waitKey(1) & 0xFF
                    
                    if key == ord('q'):
                        break
                    elif key == ord('s') and output_dir:
                        # Save screenshot
                        ss_path = output_dir / f"screenshot_{self.frame_count}.jpg"
                        cv2.imwrite(str(ss_path), result_frame)
                        console.log(f"[green]Screenshot saved: {ss_path}")
        
        finally:
            cap.release()
            if writer:
                writer.release()
            cv2.destroyAllWindows()
    
    def _process_image(self, source, output_dir, show):
        """Process single image."""
        img = cv2.imread(source)
        if img is None:
            console.log(f"[red]Could not read image: {source}")
            return
        
        console.log(f"[cyan]Processing: {Path(source).name}")
        
        result_frame, detections = self.process_frame(img)
        
        # Save output
        if output_dir:
            out_path = output_dir / f"result_{Path(source).name}"
            cv2.imwrite(str(out_path), result_frame)
            console.log(f"[green]Saved: {out_path}")
        
        # Display
        if show:
            cv2.imshow("ANPR Result", result_frame)
            console.log("[yellow]Press any key to close...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    
    def _process_folder(self, source, output_dir, show):
        """Process folder of images."""
        folder = Path(source)
        image_files = []
        for ext in [".jpg", ".jpeg", ".png", ".bmp"]:
            image_files.extend(folder.glob(f"*{ext}"))
            image_files.extend(folder.glob(f"*{ext.upper()}"))
        
        image_files = sorted(image_files)
        console.log(f"[cyan]Found {len(image_files)} images")
        
        for img_path in image_files:
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            
            result_frame, detections = self.process_frame(img)
            
            # Log detections
            for det in detections:
                if self.detected_plates:
                    plate = list(self.detected_plates)[-1] if self.detected_plates else ""
                    console.log(f"[green]{img_path.name}: {plate}")
            
            # Save output
            if output_dir:
                out_path = output_dir / img_path.name
                cv2.imwrite(str(out_path), result_frame)
            
            # Display
            if show:
                cv2.imshow("ANPR Result", result_frame)
                key = cv2.waitKey(100) & 0xFF
                if key == ord('q'):
                    break
        
        cv2.destroyAllWindows()
    
    def _print_summary(self):
        """Print detection summary."""
        elapsed = time.time() - self.start_time if self.start_time else 0
        
        console.rule("[bold green]Detection Summary")
        
        if HAS_RICH:
            table = Table(title="ANPR Results")
            table.add_column("Metric", style="cyan")
            table.add_column("Value", style="green")
            
            table.add_row("Frames Processed", str(self.frame_count))
            table.add_row("Total Time", f"{elapsed:.1f}s")
            table.add_row("Average FPS", f"{self.frame_count/elapsed:.1f}" if elapsed > 0 else "N/A")
            table.add_row("Plates Detected", str(len(self.detected_plates)))
            
            console.print(table)
        else:
            console.log(f"Frames: {self.frame_count}")
            console.log(f"Time: {elapsed:.1f}s")
            console.log(f"Plates: {len(self.detected_plates)}")
        
        # Print detected plates
        if self.detected_plates:
            console.print("\n[bold]Detected License Plates:[/bold]" if HAS_RICH else "\nDetected Plates:")
            for plate in sorted(set(self.detected_plates)):
                console.print(f"  • {plate}")


def main():
    parser = argparse.ArgumentParser(
        description="Real-time ANPR System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Webcam (default camera)
  python realtime_anpr.py --source 0
  
  # Video file
  python realtime_anpr.py --source video.mp4
  
  # Single image
  python realtime_anpr.py --source image.jpg
  
  # Folder of images  
  python realtime_anpr.py --source images/
  
  # Save output video
  python realtime_anpr.py --source 0 --save-video --output runs/output
        """
    )
    
    parser.add_argument("--source", default="0",
                        help="Video/image path, folder, or webcam ID (default: 0)")
    parser.add_argument("--model", choices=["yolov5", "yolov9"], default="yolov5",
                        help="Model type (default: yolov5)")
    parser.add_argument("--weights", type=Path, default=None,
                        help="Path to weights file")
    parser.add_argument("--conf", type=float, default=0.25,
                        help="Confidence threshold (default: 0.25)")
    parser.add_argument("--device", default="",
                        help="Device: '', 0, cpu, mps")
    parser.add_argument("--output", type=Path, default=None,
                        help="Output directory")
    parser.add_argument("--save-video", action="store_true",
                        help="Save output video")
    parser.add_argument("--no-display", action="store_true",
                        help="Don't show output window")
    parser.add_argument("--ocr-langs", nargs="+", default=["en"],
                        help="OCR languages")
    
    args = parser.parse_args()
    
    # Create ANPR system
    anpr = ANPRSystem(
        model_type=args.model,
        weights=args.weights,
        conf_threshold=args.conf,
        device=args.device,
        ocr_langs=args.ocr_langs
    )
    
    # Run
    anpr.run(
        source=args.source,
        output_dir=args.output,
        save_video=args.save_video,
        show=not args.no_display
    )


if __name__ == "__main__":
    main()


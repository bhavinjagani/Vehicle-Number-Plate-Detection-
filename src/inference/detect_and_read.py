#!/usr/bin/env python
"""
Unified inference pipeline for YOLOv5/YOLOv9 + EasyOCR.
Accepts images, folders, webcam IDs, RTSP streams, or video files.
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import easyocr
import numpy as np
from rich.console import Console
from ultralytics import YOLO

console = Console()
ROOT = Path(__file__).resolve().parents[2]
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Detect plates and read text.")
    parser.add_argument("--model-type", choices=("yolov5", "yolov9"), required=True)
    parser.add_argument("--weights", type=Path, required=True, help="Path to trained .pt file")
    parser.add_argument("--source", required=True, help="Image/video path or stream (e.g. 0)")
    parser.add_argument("--img-size", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--device", default="0", help="CUDA device string or 'cpu'")
    parser.add_argument("--ocr-langs", nargs="+", default=["en"])
    parser.add_argument("--ocr-min-conf", type=float, default=0.45)
    parser.add_argument("--save-frames", action="store_true", help="Save annotated frames/images")
    parser.add_argument("--save-video", action="store_true", help="Encode annotated video(s)")
    parser.add_argument("--save-crops", action="store_true", help="Store detected plate crops")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/runs"))
    parser.add_argument("--output-fps", type=float, default=20.0)
    parser.add_argument("--cpu", action="store_true", help="Force EasyOCR to run on CPU")
    return parser.parse_args()


def build_run_dir(base: Path, model_type: str) -> Path:
    timestamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    run_dir = base / f"{model_type}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def ensure_dirs(run_dir: Path, save_frames: bool, save_crops: bool) -> Tuple[Path, Path]:
    frames_dir = run_dir / "frames"
    crops_dir = run_dir / "crops"
    if save_frames:
        frames_dir.mkdir(parents=True, exist_ok=True)
    if save_crops:
        crops_dir.mkdir(parents=True, exist_ok=True)
    return frames_dir, crops_dir


def run_easyocr(reader: easyocr.Reader, crop: np.ndarray, min_conf: float) -> Tuple[str, float]:
    if crop.size == 0:
        return "", 0.0
    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    results = reader.readtext(rgb, detail=1)
    if not results:
        return "", 0.0
    best = max(results, key=lambda x: x[2])
    text = best[1]
    conf = float(best[2])
    if conf < min_conf:
        return "", conf
    return text, conf


def overlay_box(frame: np.ndarray, bbox: List[int], text: str, conf: float, text_conf: float) -> np.ndarray:
    x1, y1, x2, y2 = bbox
    color = (0, 255, 0)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    label = f"{conf:.2f}"
    if text:
        label += f" | {text} ({text_conf:.2f})"
    cv2.putText(
        frame,
        label,
        (x1, max(y1 - 10, 0)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        color,
        2,
        cv2.LINE_AA,
    )
    return frame


def write_jsonl(records: List[Dict], path: Path) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")


def write_csv(records: List[Dict], path: Path) -> None:
    flat_rows = []
    for record in records:
        for det in record["detections"]:
            flat_rows.append(
                {
                    "source": record["source"],
                    "frame_id": record["frame_id"],
                    "bbox": det["bbox"],
                    "det_conf": det["det_confidence"],
                    "plate_text": det["plate_text"],
                    "ocr_conf": det["ocr_confidence"],
                }
            )
    if not flat_rows:
        return
    with path.open("w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=list(flat_rows[0].keys()))
        writer.writeheader()
        writer.writerows(flat_rows)


def main() -> None:
    args = parse_args()
    if not args.weights.exists():
        raise FileNotFoundError(f"Weights file not found: {args.weights}")

    run_dir = build_run_dir(args.output_dir, args.model_type)
    save_frames = args.save_frames or not args.save_video
    frames_dir, crops_dir = ensure_dirs(run_dir, save_frames, args.save_crops)

    console.rule(f"[bold green]Inference :: {args.model_type.upper()}")
    console.log(f"Writing artifacts to {run_dir}")

    model = YOLO(str(args.weights))
    use_gpu_for_ocr = not args.cpu and args.device != "cpu"
    reader = easyocr.Reader(args.ocr_langs, gpu=use_gpu_for_ocr)

    writers: Dict[str, cv2.VideoWriter] = {}
    records: List[Dict] = []

    stream = model(
        source=args.source,
        imgsz=args.img_size,
        conf=args.conf,
        stream=True,
        device=args.device,
        verbose=False,
    )

    for result in stream:
        frame = result.orig_img
        if frame is None:
            continue
        annotated = frame.copy()
        detections = []

        boxes = getattr(result, "boxes", None)
        if boxes is None:
            continue

        for det_idx, box in enumerate(boxes):
            xyxy = box.xyxy[0].cpu().numpy().astype(int)
            x1, y1, x2, y2 = xyxy.tolist()
            x1, y1 = max(x1, 0), max(y1, 0)
            x2, y2 = min(x2, frame.shape[1] - 1), min(y2, frame.shape[0] - 1)
            crop = frame[y1:y2, x1:x2]
            text, text_conf = run_easyocr(reader, crop, args.ocr_min_conf)

            annotated = overlay_box(
                annotated,
                [x1, y1, x2, y2],
                text,
                float(box.conf[0]),
                text_conf,
            )

            if args.save_crops and crop.size:
                crop_name = f"{Path(result.path).stem}_f{getattr(result, 'frame', det_idx)}_{det_idx}.png"
                cv2.imwrite(str(crops_dir / crop_name), crop)

            detections.append(
                {
                    "bbox": [x1, y1, x2, y2],
                    "det_confidence": float(box.conf[0]),
                    "plate_text": text,
                    "ocr_confidence": text_conf,
                }
            )

        record = {
            "source": str(result.path),
            "frame_id": int(getattr(result, "frame", len(records))),
            "detections": detections,
        }
        records.append(record)

        source_key = str(result.path)
        if args.save_video:
            writer = writers.get(source_key)
            if writer is None:
                out_path = run_dir / f"{Path(source_key).stem or 'stream'}.mp4"
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                height, width = annotated.shape[:2]
                writer = cv2.VideoWriter(str(out_path), fourcc, args.output_fps, (width, height))
                writers[source_key] = writer
            writer.write(annotated)
        else:
            if save_frames:
                filename = Path(source_key).name or f"frame_{record['frame_id']:06d}.jpg"
                out_path = frames_dir / filename
                cv2.imwrite(str(out_path), annotated)

    for writer in writers.values():
        writer.release()

    write_jsonl(records, run_dir / "results.jsonl")
    write_csv(records, run_dir / "results.csv")

    console.log(f"[green]Inference complete. Records saved to {run_dir}")


if __name__ == "__main__":
    main()

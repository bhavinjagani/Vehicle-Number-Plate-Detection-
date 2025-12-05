#!/usr/bin/env python
"""
Model Comparison Script - YOLOv5 vs YOLOv9

Compares performance metrics of trained YOLOv5 and YOLOv9 models for ANPR.
Generates a comprehensive report with:
- Accuracy metrics (mAP, Precision, Recall)
- Speed benchmarks (FPS, inference time)
- Model size comparison
- Visual comparison charts
"""

from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]

try:
    from rich.console import Console
    from rich.table import Table
    console = Console()
except ImportError:
    class Console:
        def log(self, msg): print(msg)
        def print(self, msg): print(msg)
        def rule(self, msg): print(f"\n{'='*60}\n{msg}\n{'='*60}")
    console = Console()


def get_model_file_info(weights_path: Path) -> Dict:
    """Get basic model file information."""
    file_size_mb = weights_path.stat().st_size / (1024 * 1024)
    return {
        "weights_file": weights_path.name,
        "file_size_mb": round(file_size_mb, 2),
    }


def load_training_results(model_type: str) -> Dict:
    """Load training results from results.csv."""
    if model_type == "yolov5":
        results_path = ROOT / "runs" / "train_yolov5" / "exp" / "results.csv"
    else:
        results_path = ROOT / "runs" / "train_yolov9" / "exp" / "results.csv"
    
    if not results_path.exists():
        return {}
    
    results = {}
    try:
        with open(results_path) as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            if rows:
                last_row = rows[-1]
                cleaned = {k.strip(): v.strip() for k, v in last_row.items()}
                
                results = {
                    "epochs": len(rows),
                    "final_precision": float(cleaned.get("metrics/precision", 0)),
                    "final_recall": float(cleaned.get("metrics/recall", 0)),
                    "final_mAP50": float(cleaned.get("metrics/mAP_0.5", cleaned.get("metrics/mAP50", 0))),
                    "final_mAP50_95": float(cleaned.get("metrics/mAP_0.5:0.95", cleaned.get("metrics/mAP50-95", 0))),
                    "final_box_loss": float(cleaned.get("train/box_loss", 0)),
                    "final_cls_loss": float(cleaned.get("train/cls_loss", 0)),
                }
    except Exception as e:
        console.log(f"[yellow]Could not parse results for {model_type}: {e}")
    
    return results


def run_speed_benchmark(model_type: str, weights_path: Path, test_images: List[Path]) -> Dict:
    """Run speed benchmark using subprocess to avoid module conflicts."""
    
    # Prepare test images list
    test_imgs_str = str([str(p) for p in test_images[:20]])
    
    benchmark_script = f'''
import sys
import time
import cv2
import numpy as np
from pathlib import Path

ROOT = Path(r"{ROOT}")
sys.path.insert(0, str(ROOT / "external" / "{model_type}"))

import torch
from models.common import DetectMultiBackend
from utils.general import check_img_size, non_max_suppression
from utils.torch_utils import select_device

# Setup device
if torch.cuda.is_available():
    device = select_device("0")
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = select_device("mps")
else:
    device = select_device("cpu")

# Load model
model = DetectMultiBackend(r"{weights_path}", device=device, fp16=False)
stride = model.stride
imgsz = check_img_size(640, s=stride)
model.warmup(imgsz=(1, 3, imgsz, imgsz))

# Get model info
total_params = sum(p.numel() for p in model.model.parameters())
num_layers = len(list(model.model.modules()))

# Test images
test_images = {test_imgs_str}

inference_times = []
detection_counts = []

for img_path in test_images:
    img = cv2.imread(img_path)
    if img is None:
        continue
    
    # Preprocess
    shape = img.shape[:2]
    r = min(imgsz / shape[0], imgsz / shape[1])
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = imgsz - new_unpad[0], imgsz - new_unpad[1]
    dw, dh = np.mod(dw, stride), np.mod(dh, stride)
    dw /= 2
    dh /= 2
    
    img_resized = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img_padded = cv2.copyMakeBorder(img_resized, top, bottom, left, right,
                                     cv2.BORDER_CONSTANT, value=(114, 114, 114))
    
    img_input = img_padded.transpose((2, 0, 1))[::-1]
    img_input = np.ascontiguousarray(img_input)
    img_input = torch.from_numpy(img_input).to(device)
    img_input = img_input.float() / 255.0
    img_input = img_input.unsqueeze(0)
    
    # Inference
    start = time.perf_counter()
    pred = model(img_input)
    pred = non_max_suppression(pred, 0.25, 0.45, max_det=100)
    inference_time = (time.perf_counter() - start) * 1000
    
    inference_times.append(inference_time)
    detection_counts.append(sum(len(d) for d in pred))

# Results
avg_time = sum(inference_times) / len(inference_times) if inference_times else 0
fps = 1000 / avg_time if avg_time > 0 else 0

print("RESULT:total_params=" + str(total_params))
print("RESULT:num_layers=" + str(num_layers))
print("RESULT:avg_inference_ms=" + str(round(avg_time, 2)))
print("RESULT:fps=" + str(round(fps, 1)))
print("RESULT:total_detections=" + str(sum(detection_counts)))
print("RESULT:avg_detections=" + str(round(sum(detection_counts)/len(detection_counts), 2) if detection_counts else 0))
print("RESULT:device=" + str(device))
'''
    
    # Run benchmark in subprocess
    result = subprocess.run(
        [sys.executable, "-c", benchmark_script],
        capture_output=True,
        text=True,
        cwd=str(ROOT)
    )
    
    # Parse results
    benchmark_results = {}
    for line in result.stdout.split("\n"):
        if line.startswith("RESULT:"):
            key, value = line[7:].split("=")
            try:
                if "." in value:
                    benchmark_results[key] = float(value)
                else:
                    benchmark_results[key] = int(value)
            except:
                benchmark_results[key] = value
    
    if result.returncode != 0:
        console.log(f"[yellow]Benchmark warning: {result.stderr[:200]}")
    
    return benchmark_results


def generate_report(yolov5_results: Dict, yolov9_results: Dict, output_dir: Path) -> Path:
    """Generate markdown comparison report."""
    
    report_path = output_dir / "model_comparison_report.md"
    
    with open(report_path, "w") as f:
        f.write("# 🚗 YOLOv5 vs YOLOv9 - ANPR Model Comparison Report\n\n")
        f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("---\n\n")
        f.write("## 📊 Executive Summary\n\n")
        f.write("This report compares YOLOv5s and YOLOv9 GELAN-c models trained for ")
        f.write("Automatic Number Plate Recognition (ANPR) on the same dataset.\n\n")
        
        # Quick comparison table
        f.write("### Quick Comparison\n\n")
        f.write("| Metric | YOLOv5s | YOLOv9 GELAN-c | Winner |\n")
        f.write("|--------|---------|----------------|--------|\n")
        
        comparisons = [
            ("Model Size", f"{yolov5_results['file_info']['file_size_mb']:.1f} MB", 
             f"{yolov9_results['file_info']['file_size_mb']:.1f} MB",
             "YOLOv5 ✅" if yolov5_results['file_info']['file_size_mb'] < yolov9_results['file_info']['file_size_mb'] else "YOLOv9 ✅"),
            ("Parameters", f"{yolov5_results['benchmark'].get('total_params', 0)/1e6:.1f}M",
             f"{yolov9_results['benchmark'].get('total_params', 0)/1e6:.1f}M",
             "YOLOv5 ✅" if yolov5_results['benchmark'].get('total_params', 0) < yolov9_results['benchmark'].get('total_params', 0) else "YOLOv9 ✅"),
            ("Inference Speed", f"{yolov5_results['benchmark'].get('fps', 0):.1f} FPS",
             f"{yolov9_results['benchmark'].get('fps', 0):.1f} FPS",
             "YOLOv5 ✅" if yolov5_results['benchmark'].get('fps', 0) > yolov9_results['benchmark'].get('fps', 0) else "YOLOv9 ✅"),
            ("mAP@0.5", f"{yolov5_results['training'].get('final_mAP50', 0):.3f}",
             f"{yolov9_results['training'].get('final_mAP50', 0):.3f}",
             "YOLOv5 ✅" if yolov5_results['training'].get('final_mAP50', 0) > yolov9_results['training'].get('final_mAP50', 0) else "YOLOv9 ✅"),
            ("mAP@0.5:0.95", f"{yolov5_results['training'].get('final_mAP50_95', 0):.3f}",
             f"{yolov9_results['training'].get('final_mAP50_95', 0):.3f}",
             "YOLOv5 ✅" if yolov5_results['training'].get('final_mAP50_95', 0) > yolov9_results['training'].get('final_mAP50_95', 0) else "YOLOv9 ✅"),
        ]
        
        for metric, y5, y9, winner in comparisons:
            f.write(f"| {metric} | {y5} | {y9} | {winner} |\n")
        
        f.write("\n---\n\n")
        
        # Model Architecture
        f.write("## 🏗️ Model Architecture\n\n")
        f.write("| Property | YOLOv5s | YOLOv9 GELAN-c |\n")
        f.write("|----------|---------|----------------|\n")
        f.write(f"| File Size | {yolov5_results['file_info']['file_size_mb']:.1f} MB | {yolov9_results['file_info']['file_size_mb']:.1f} MB |\n")
        f.write(f"| Parameters | {yolov5_results['benchmark'].get('total_params', 0):,} | {yolov9_results['benchmark'].get('total_params', 0):,} |\n")
        f.write(f"| Layers | {yolov5_results['benchmark'].get('num_layers', 'N/A')} | {yolov9_results['benchmark'].get('num_layers', 'N/A')} |\n")
        f.write(f"| Architecture | CSPDarknet + PANet | GELAN + ELAN |\n\n")
        
        # Training Results
        f.write("## 📈 Training Results\n\n")
        f.write("| Metric | YOLOv5s | YOLOv9 GELAN-c |\n")
        f.write("|--------|---------|----------------|\n")
        f.write(f"| Epochs Trained | {yolov5_results['training'].get('epochs', 'N/A')} | {yolov9_results['training'].get('epochs', 'N/A')} |\n")
        f.write(f"| Final Precision | {yolov5_results['training'].get('final_precision', 0):.4f} | {yolov9_results['training'].get('final_precision', 0):.4f} |\n")
        f.write(f"| Final Recall | {yolov5_results['training'].get('final_recall', 0):.4f} | {yolov9_results['training'].get('final_recall', 0):.4f} |\n")
        f.write(f"| mAP@0.5 | {yolov5_results['training'].get('final_mAP50', 0):.4f} | {yolov9_results['training'].get('final_mAP50', 0):.4f} |\n")
        f.write(f"| mAP@0.5:0.95 | {yolov5_results['training'].get('final_mAP50_95', 0):.4f} | {yolov9_results['training'].get('final_mAP50_95', 0):.4f} |\n\n")
        
        # Inference Performance
        f.write("## ⚡ Inference Performance\n\n")
        f.write("| Metric | YOLOv5s | YOLOv9 GELAN-c |\n")
        f.write("|--------|---------|----------------|\n")
        f.write(f"| Device | {yolov5_results['benchmark'].get('device', 'N/A')} | {yolov9_results['benchmark'].get('device', 'N/A')} |\n")
        f.write(f"| Avg Inference Time | {yolov5_results['benchmark'].get('avg_inference_ms', 0):.1f} ms | {yolov9_results['benchmark'].get('avg_inference_ms', 0):.1f} ms |\n")
        f.write(f"| FPS | {yolov5_results['benchmark'].get('fps', 0):.1f} | {yolov9_results['benchmark'].get('fps', 0):.1f} |\n")
        f.write(f"| Avg Detections/Image | {yolov5_results['benchmark'].get('avg_detections', 0):.2f} | {yolov9_results['benchmark'].get('avg_detections', 0):.2f} |\n\n")
        
        # Analysis
        f.write("## 🔍 Analysis\n\n")
        
        y5_fps = yolov5_results['benchmark'].get('fps', 0)
        y9_fps = yolov9_results['benchmark'].get('fps', 0)
        speed_ratio = y5_fps / y9_fps if y9_fps > 0 else 0
        
        y5_size = yolov5_results['file_info']['file_size_mb']
        y9_size = yolov9_results['file_info']['file_size_mb']
        size_ratio = y9_size / y5_size if y5_size > 0 else 0
        
        f.write("### Key Findings\n\n")
        f.write(f"1. **Speed:** YOLOv5s is **{speed_ratio:.1f}x faster** than YOLOv9 GELAN-c\n")
        f.write(f"2. **Size:** YOLOv5s is **{size_ratio:.1f}x smaller** than YOLOv9 GELAN-c\n")
        f.write(f"3. **Accuracy:** ")
        
        y5_map = yolov5_results['training'].get('final_mAP50', 0)
        y9_map = yolov9_results['training'].get('final_mAP50', 0)
        if y5_map > y9_map:
            f.write(f"YOLOv5s achieves **{((y5_map-y9_map)/y9_map*100):.1f}% higher mAP@0.5**\n")
        else:
            f.write(f"YOLOv9 achieves **{((y9_map-y5_map)/y5_map*100):.1f}% higher mAP@0.5**\n")
        
        f.write("\n### Recommendations\n\n")
        f.write("| Use Case | Recommended Model | Reason |\n")
        f.write("|----------|-------------------|--------|\n")
        f.write("| Real-time detection | YOLOv5s | Higher FPS |\n")
        f.write("| Edge/Mobile deployment | YOLOv5s | Smaller size |\n")
        f.write("| Accuracy-critical apps | Check mAP scores | Depends on training |\n")
        f.write("| Resource-constrained | YOLOv5s | Lower compute needs |\n\n")
        
        f.write("---\n\n")
        f.write("## 📁 Output Files\n\n")
        f.write("- `model_comparison_report.md` - This report\n")
        f.write("- `comparison_data.json` - Raw comparison data\n")
        f.write("- `comparison_charts.png` - Visual comparison charts\n\n")
        
        f.write("---\n")
        f.write("*Generated by ANPR Model Comparison Tool*\n")
    
    return report_path


def generate_charts(yolov5_results: Dict, yolov9_results: Dict, output_dir: Path) -> Path:
    """Generate comparison charts."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('YOLOv5 vs YOLOv9 - ANPR Model Comparison', fontsize=16, fontweight='bold')
        
        colors = ['#3498db', '#e74c3c']
        models = ['YOLOv5s', 'YOLOv9']
        
        # 1. Model Size
        ax1 = axes[0, 0]
        sizes = [yolov5_results['file_info']['file_size_mb'], 
                 yolov9_results['file_info']['file_size_mb']]
        bars1 = ax1.bar(models, sizes, color=colors, edgecolor='black', linewidth=1.2)
        ax1.set_ylabel('File Size (MB)', fontsize=11)
        ax1.set_title('Model Size Comparison', fontsize=12, fontweight='bold')
        for bar, size in zip(bars1, sizes):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2, 
                    f'{size:.1f} MB', ha='center', va='bottom', fontsize=10)
        ax1.set_ylim(0, max(sizes) * 1.2)
        
        # 2. Inference Speed
        ax2 = axes[0, 1]
        fps_vals = [yolov5_results['benchmark'].get('fps', 0), 
                    yolov9_results['benchmark'].get('fps', 0)]
        bars2 = ax2.bar(models, fps_vals, color=colors, edgecolor='black', linewidth=1.2)
        ax2.set_ylabel('Frames Per Second', fontsize=11)
        ax2.set_title('Inference Speed', fontsize=12, fontweight='bold')
        for bar, fps in zip(bars2, fps_vals):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                    f'{fps:.1f} FPS', ha='center', va='bottom', fontsize=10)
        ax2.set_ylim(0, max(fps_vals) * 1.2 if max(fps_vals) > 0 else 10)
        
        # 3. Accuracy Metrics
        ax3 = axes[1, 0]
        metrics = ['Precision', 'Recall', 'mAP@0.5', 'mAP@0.5:0.95']
        y5_vals = [
            yolov5_results['training'].get('final_precision', 0),
            yolov5_results['training'].get('final_recall', 0),
            yolov5_results['training'].get('final_mAP50', 0),
            yolov5_results['training'].get('final_mAP50_95', 0)
        ]
        y9_vals = [
            yolov9_results['training'].get('final_precision', 0),
            yolov9_results['training'].get('final_recall', 0),
            yolov9_results['training'].get('final_mAP50', 0),
            yolov9_results['training'].get('final_mAP50_95', 0)
        ]
        
        x = np.arange(len(metrics))
        width = 0.35
        bars3a = ax3.bar(x - width/2, y5_vals, width, label='YOLOv5s', color=colors[0], edgecolor='black')
        bars3b = ax3.bar(x + width/2, y9_vals, width, label='YOLOv9', color=colors[1], edgecolor='black')
        ax3.set_ylabel('Score', fontsize=11)
        ax3.set_title('Training Metrics', fontsize=12, fontweight='bold')
        ax3.set_xticks(x)
        ax3.set_xticklabels(metrics, fontsize=9)
        ax3.legend(loc='lower right')
        ax3.set_ylim(0, 1.15)
        ax3.axhline(y=1.0, color='gray', linestyle='--', alpha=0.3)
        
        # 4. Parameters
        ax4 = axes[1, 1]
        params = [yolov5_results['benchmark'].get('total_params', 0) / 1e6, 
                  yolov9_results['benchmark'].get('total_params', 0) / 1e6]
        bars4 = ax4.bar(models, params, color=colors, edgecolor='black', linewidth=1.2)
        ax4.set_ylabel('Parameters (Millions)', fontsize=11)
        ax4.set_title('Model Complexity', fontsize=12, fontweight='bold')
        for bar, p in zip(bars4, params):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                    f'{p:.1f}M', ha='center', va='bottom', fontsize=10)
        ax4.set_ylim(0, max(params) * 1.2 if max(params) > 0 else 10)
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        chart_path = output_dir / "comparison_charts.png"
        plt.savefig(chart_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        
        return chart_path
        
    except ImportError:
        console.log("[yellow]matplotlib not available, skipping charts")
        return None


def main():
    console.rule("[bold blue]🚗 YOLOv5 vs YOLOv9 - Model Comparison")
    
    # Paths
    yolov5_weights = ROOT / "runs" / "train_yolov5" / "exp" / "weights" / "best.pt"
    yolov9_weights = ROOT / "runs" / "train_yolov9" / "exp" / "weights" / "best.pt"
    test_images_dir = ROOT / "data" / "processed" / "test" / "images"
    output_dir = ROOT / "runs" / "comparison"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Check weights
    if not yolov5_weights.exists():
        console.log(f"[red]YOLOv5 weights not found: {yolov5_weights}")
        return
    if not yolov9_weights.exists():
        console.log(f"[red]YOLOv9 weights not found: {yolov9_weights}")
        return
    
    # Get test images
    test_images = sorted(test_images_dir.glob("*.jpg"))
    console.log(f"[cyan]Found {len(test_images)} test images")
    
    # Collect results
    results = {}
    
    # YOLOv5
    console.rule("[bold green]Evaluating YOLOv5")
    console.log("[cyan]Running YOLOv5 benchmark...")
    results['yolov5'] = {
        'file_info': get_model_file_info(yolov5_weights),
        'training': load_training_results("yolov5"),
        'benchmark': run_speed_benchmark("yolov5", yolov5_weights, test_images)
    }
    console.log(f"[green]✓ YOLOv5: {results['yolov5']['benchmark'].get('fps', 0):.1f} FPS")
    
    # YOLOv9
    console.rule("[bold green]Evaluating YOLOv9")
    console.log("[cyan]Running YOLOv9 benchmark...")
    results['yolov9'] = {
        'file_info': get_model_file_info(yolov9_weights),
        'training': load_training_results("yolov9"),
        'benchmark': run_speed_benchmark("yolov9", yolov9_weights, test_images)
    }
    console.log(f"[green]✓ YOLOv9: {results['yolov9']['benchmark'].get('fps', 0):.1f} FPS")
    
    # Save raw data
    with open(output_dir / "comparison_data.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    # Generate report
    console.rule("[bold blue]Generating Report")
    report_path = generate_report(results['yolov5'], results['yolov9'], output_dir)
    console.log(f"[green]✓ Report saved: {report_path}")
    
    # Generate charts
    chart_path = generate_charts(results['yolov5'], results['yolov9'], output_dir)
    if chart_path:
        console.log(f"[green]✓ Charts saved: {chart_path}")
    
    # Print summary table
    console.rule("[bold green]📊 Comparison Summary")
    
    table = Table(title="YOLOv5 vs YOLOv9 - ANPR Comparison")
    table.add_column("Metric", style="cyan", justify="left")
    table.add_column("YOLOv5s", style="blue", justify="center")
    table.add_column("YOLOv9", style="red", justify="center")
    table.add_column("Winner", style="green", justify="center")
    
    y5 = results['yolov5']
    y9 = results['yolov9']
    
    # Size
    y5_size = y5['file_info']['file_size_mb']
    y9_size = y9['file_info']['file_size_mb']
    table.add_row("Model Size", f"{y5_size:.1f} MB", f"{y9_size:.1f} MB", 
                  "YOLOv5 ✓" if y5_size < y9_size else "YOLOv9 ✓")
    
    # Parameters
    y5_params = y5['benchmark'].get('total_params', 0) / 1e6
    y9_params = y9['benchmark'].get('total_params', 0) / 1e6
    table.add_row("Parameters", f"{y5_params:.1f}M", f"{y9_params:.1f}M",
                  "YOLOv5 ✓" if y5_params < y9_params else "YOLOv9 ✓")
    
    # FPS
    y5_fps = y5['benchmark'].get('fps', 0)
    y9_fps = y9['benchmark'].get('fps', 0)
    table.add_row("Speed (FPS)", f"{y5_fps:.1f}", f"{y9_fps:.1f}",
                  "YOLOv5 ✓" if y5_fps > y9_fps else "YOLOv9 ✓")
    
    # Inference time
    y5_time = y5['benchmark'].get('avg_inference_ms', 0)
    y9_time = y9['benchmark'].get('avg_inference_ms', 0)
    table.add_row("Inference Time", f"{y5_time:.1f} ms", f"{y9_time:.1f} ms",
                  "YOLOv5 ✓" if y5_time < y9_time else "YOLOv9 ✓")
    
    # mAP
    y5_map = y5['training'].get('final_mAP50', 0)
    y9_map = y9['training'].get('final_mAP50', 0)
    table.add_row("mAP@0.5", f"{y5_map:.3f}", f"{y9_map:.3f}",
                  "YOLOv5 ✓" if y5_map > y9_map else "YOLOv9 ✓")
    
    # mAP 50-95
    y5_map95 = y5['training'].get('final_mAP50_95', 0)
    y9_map95 = y9['training'].get('final_mAP50_95', 0)
    table.add_row("mAP@0.5:0.95", f"{y5_map95:.3f}", f"{y9_map95:.3f}",
                  "YOLOv5 ✓" if y5_map95 > y9_map95 else "YOLOv9 ✓")
    
    # Precision
    y5_prec = y5['training'].get('final_precision', 0)
    y9_prec = y9['training'].get('final_precision', 0)
    table.add_row("Precision", f"{y5_prec:.3f}", f"{y9_prec:.3f}",
                  "YOLOv5 ✓" if y5_prec > y9_prec else "YOLOv9 ✓")
    
    # Recall
    y5_rec = y5['training'].get('final_recall', 0)
    y9_rec = y9['training'].get('final_recall', 0)
    table.add_row("Recall", f"{y5_rec:.3f}", f"{y9_rec:.3f}",
                  "YOLOv5 ✓" if y5_rec > y9_rec else "YOLOv9 ✓")
    
    console.print(table)
    
    console.print(f"\n[bold]📁 Output Files:[/bold]")
    console.print(f"  • Report: {report_path}")
    if chart_path:
        console.print(f"  • Charts: {chart_path}")
    console.print(f"  • Data: {output_dir / 'comparison_data.json'}")
    
    console.print(f"\n[cyan]Open report with:[/cyan] open \"{report_path}\"")


if __name__ == "__main__":
    main()

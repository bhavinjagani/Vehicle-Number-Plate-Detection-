#!/usr/bin/env python
"""
Thin orchestration layer that launches YOLOv5 or YOLOv9 training with a shared
dataset, logging, and argument surface.

Supports automatic device detection:
- CUDA (NVIDIA GPUs)
- MPS (Apple Silicon - M1/M2/M3/M4)
- CPU (fallback)
"""

from __future__ import annotations

import argparse
import platform
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional

from rich.console import Console
from rich.table import Table

console = Console()
ROOT = Path(__file__).resolve().parent.parent


def get_best_device() -> str:
    """
    Auto-detect the best available device for training.
    Priority: CUDA > MPS (Apple Silicon) > CPU
    
    Returns:
        str: Device string compatible with PyTorch/YOLO ('0' for CUDA, 'mps' for Apple, 'cpu' for fallback)
    """
    try:
        import torch
        
        # Check for CUDA (NVIDIA GPU)
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            console.log(f"[green]✓ CUDA available: {device_name}")
            return "0"
        
        # Check for MPS (Apple Silicon)
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            # Additional check for MPS build
            if torch.backends.mps.is_built():
                console.log(f"[green]✓ Apple Silicon (MPS) available")
                return "mps"
        
        # Fallback to CPU
        console.log("[yellow]⚠ No GPU detected, using CPU (training will be slow)")
        return "cpu"
        
    except ImportError:
        console.log("[yellow]⚠ PyTorch not found, defaulting to CPU")
        return "cpu"


def get_recommended_settings(device: str) -> Dict[str, int]:
    """
    Get recommended batch size and workers based on device.
    
    Args:
        device: The device string ('0', 'mps', or 'cpu')
    
    Returns:
        Dict with recommended 'batch_size' and 'workers'
    """
    if device == "mps":
        # Apple Silicon - moderate batch size, fewer workers to avoid memory issues
        return {"batch_size": 8, "workers": 4}
    elif device == "0" or device.isdigit():
        # CUDA - can handle larger batches
        return {"batch_size": 16, "workers": 8}
    else:
        # CPU - small batch size
        return {"batch_size": 4, "workers": 2}


def print_system_info() -> None:
    """Print system information for debugging."""
    console.log(f"[cyan]System: {platform.system()} {platform.machine()}")
    console.log(f"[cyan]Python: {sys.version.split()[0]}")
    
    try:
        import torch
        console.log(f"[cyan]PyTorch: {torch.__version__}")
    except ImportError:
        pass

MODEL_REGISTRY: Dict[str, Dict[str, str]] = {
    "yolov5": {
        "repo": "external/yolov5",
        "script": "train.py",
        "default_weights": "yolov5s.pt",
        "default_project": "runs/train_yolov5",
    },
    "yolov9": {
        "repo": "external/yolov9",
        "script": "train.py",
        "default_weights": "yolov9-c.pt",
        "default_project": "runs/train_yolov9",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train YOLOv5 or YOLOv9 on the unified dataset.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Device Options:
  --device auto    Auto-detect best device (CUDA > MPS > CPU) [default]
  --device 0       Use CUDA GPU 0
  --device mps     Use Apple Silicon GPU
  --device cpu     Use CPU only

Examples:
  python src/train.py --model yolov9 --data data/processed/data.yaml
  python src/train.py --model yolov5 --data data/processed/data.yaml --device mps
        """
    )
    parser.add_argument("--model", choices=MODEL_REGISTRY.keys(), required=True)
    parser.add_argument("--data", type=Path, required=True, help="Path to YOLO data.yaml")
    parser.add_argument("--weights", type=str, default=None, help="Initial weights (empty string '' for scratch)")
    parser.add_argument("--cfg", type=Path, help="Model config yaml (for YOLOv9)")
    parser.add_argument("--img-size", type=int, default=640)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size (auto if not specified)")
    parser.add_argument("--device", default="auto", help="Device: auto, 0 (CUDA), mps (Apple), cpu")
    parser.add_argument("--project", type=Path, help="Override project directory")
    parser.add_argument("--name", type=str, default=None, help="Experiment name")
    parser.add_argument("--hyp", type=Path, help="Custom hyp.yaml path")
    parser.add_argument("--workers", type=int, default=None, help="Dataloader workers (auto if not specified)")
    parser.add_argument("--freeze", type=int, default=0)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--exist-ok", action="store_true")
    parser.add_argument(
        "--opts",
        nargs=argparse.REMAINDER,
        default=[],
        help="Additional upstream args appended verbatim (e.g. --cos-lr).",
    )
    return parser.parse_args()


def ensure_repo_exists(repo_path: Path) -> None:
    if not repo_path.exists():
        raise FileNotFoundError(
            f"{repo_path} not found. Run bash scripts/bootstrap_models.sh first."
        )


def build_train_command(args: argparse.Namespace, meta: Dict[str, str]) -> List[str]:
    script_path = (ROOT / meta["repo"] / meta["script"]).resolve()
    ensure_repo_exists(script_path.parent)

    # Auto-detect device if set to 'auto'
    if args.device == "auto":
        device = get_best_device()
    else:
        device = args.device
        console.log(f"[cyan]Using specified device: {device}")
    
    # Get recommended settings based on device
    recommended = get_recommended_settings(device)
    
    # Use provided values or fall back to recommendations
    batch_size = args.batch_size if args.batch_size is not None else recommended["batch_size"]
    workers = args.workers if args.workers is not None else recommended["workers"]
    
    console.log(f"[cyan]Batch size: {batch_size}, Workers: {workers}")

    # Handle weights - empty string means train from scratch
    if args.weights is not None:
        weights_str = str(args.weights).strip()
        if weights_str == "":
            # Empty string means train from scratch
            weights = ""
            console.log("[cyan]Training from scratch (no pretrained weights)")
        else:
            weights = str(Path(weights_str).resolve())
    else:
        weights = meta["default_weights"]

    project = Path(args.project).resolve() if args.project else (ROOT / meta["default_project"])

    cmd = [
        sys.executable,
        str(script_path),
        "--data",
        str(args.data.resolve()),
        "--weights",
        str(weights),
        "--img",
        str(args.img_size),
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(batch_size),
        "--device",
        str(device),
        "--project",
        str(project),
        "--workers",
        str(workers),
        "--freeze",
        str(args.freeze),
        "--patience",
        str(args.patience),
    ]

    # Add model config for YOLOv9 (required for training from scratch or with GELAN)
    if args.cfg:
        cmd += ["--cfg", str(Path(args.cfg).resolve())]
    elif args.model == "yolov9" and (not args.weights or str(args.weights) == ""):
        # Default to gelan-c for YOLOv9 training from scratch (more stable)
        default_cfg = ROOT / meta["repo"] / "models" / "detect" / "gelan-c.yaml"
        if default_cfg.exists():
            cmd += ["--cfg", str(default_cfg)]
            console.log(f"[cyan]Using default config: gelan-c.yaml")

    if args.name:
        cmd += ["--name", args.name]
    if args.hyp:
        cmd += ["--hyp", str(Path(args.hyp).resolve())]
    if args.resume:
        cmd.append("--resume")
    if args.exist_ok:
        cmd.append("--exist-ok")

    cmd.extend(args.opts)
    return cmd


def pretty_print(cmd: List[str]) -> None:
    table = Table(title="Launching training job")
    table.add_column("Argument")
    table.add_column("Value", overflow="fold")
    skip_next = False
    for idx, token in enumerate(cmd):
        if skip_next:
            skip_next = False
            continue
        if not token.startswith("--"):
            continue
        value = ""
        if idx + 1 < len(cmd) and not cmd[idx + 1].startswith("--"):
            value = cmd[idx + 1]
            skip_next = True
        table.add_row(token, value)
    console.print(table)


def main() -> None:
    args = parse_args()
    meta = MODEL_REGISTRY[args.model]
    
    console.rule(f"[bold green]Training {args.model.upper()}")
    print_system_info()
    
    cmd = build_train_command(args, meta)
    console.log(" ".join(cmd))
    pretty_print(cmd)

    subprocess.run(cmd, check=True)
    console.log("[green]Training completed successfully.")


if __name__ == "__main__":
    main()

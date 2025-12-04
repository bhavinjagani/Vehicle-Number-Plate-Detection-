#!/usr/bin/env python
"""
Thin orchestration layer that launches YOLOv5 or YOLOv9 training with a shared
dataset, logging, and argument surface.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

from rich.console import Console
from rich.table import Table

console = Console()
ROOT = Path(__file__).resolve().parent.parent

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
        description="Train YOLOv5 or YOLOv9 on the unified dataset."
    )
    parser.add_argument("--model", choices=MODEL_REGISTRY.keys(), required=True)
    parser.add_argument("--data", type=Path, required=True, help="Path to YOLO data.yaml")
    parser.add_argument("--weights", type=Path, help="Initial weights checkpoint")
    parser.add_argument("--img-size", type=int, default=640)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="0")
    parser.add_argument("--project", type=Path, help="Override project directory")
    parser.add_argument("--name", type=str, default=None, help="Experiment name")
    parser.add_argument("--hyp", type=Path, help="Custom hyp.yaml path")
    parser.add_argument("--workers", type=int, default=8)
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

    if args.weights:
        weights = Path(args.weights).resolve()
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
        str(args.batch_size),
        "--device",
        str(args.device),
        "--project",
        str(project),
        "--workers",
        str(args.workers),
        "--freeze",
        str(args.freeze),
        "--patience",
        str(args.patience),
    ]

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
    cmd = build_train_command(args, meta)

    console.rule(f"[bold green]Training {args.model.upper()}")
    console.log(" ".join(cmd))
    pretty_print(cmd)

    subprocess.run(cmd, check=True)
    console.log("[green]Training completed successfully.")


if __name__ == "__main__":
    main()

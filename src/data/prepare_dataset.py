#!/usr/bin/env python
"""
Merge multiple ALPR datasets (already in YOLO txt format) into a single dataset
that can be shared by both YOLOv5 and YOLOv9 training runs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, Optional

import yaml
from rich.console import Console
from tqdm import tqdm

console = Console()

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge ALPR datasets into a single YOLO-ready directory."
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to configs/datasets.yaml",
    )
    parser.add_argument(
        "--strategy",
        choices=("copy", "symlink"),
        help="Override merge strategy (defaults to config value).",
    )
    parser.add_argument(
        "--limit-per-split",
        type=int,
        default=None,
        help="Optional cap per split for smoke tests.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse everything but do not copy/symlink files.",
    )
    return parser.parse_args()


def load_yaml(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def iter_images(root: Path) -> Iterable[Path]:
    for ext in IMAGE_EXTS:
        yield from sorted(root.rglob(f"*{ext}"))


def choose_strategy(cfg_strategy: str, arg_strategy: Optional[str]) -> str:
    strategy = (arg_strategy or cfg_strategy or "copy").lower()
    if strategy not in {"copy", "symlink"}:
        raise ValueError(f"Unsupported merge strategy: {strategy}")
    return strategy


def sanitize_name(dataset: str, index: int, original: Path) -> str:
    suffix = original.suffix.lower()
    hash_suffix = hashlib.sha1(f"{dataset}_{original.stem}_{index}".encode()).hexdigest()[:6]
    return f"{dataset}_{original.stem}_{index:06d}_{hash_suffix}{suffix}"


def copy_pair(
    image_path: Path,
    label_path: Path,
    dest_images: Path,
    dest_labels: Path,
    new_image_name: str,
    new_label_name: str,
    strategy: str,
    dry_run: bool,
) -> None:
    if dry_run:
        return

    ensure_dir(dest_images)
    ensure_dir(dest_labels)

    image_dest = dest_images / new_image_name
    label_dest = dest_labels / new_label_name

    if strategy == "copy":
        shutil.copy2(image_path, image_dest)
        shutil.copy2(label_path, label_dest)
    else:
        if image_dest.exists():
            image_dest.unlink()
        if label_dest.exists():
            label_dest.unlink()
        image_dest.symlink_to(image_path.resolve())
        label_dest.symlink_to(label_path.resolve())


def build_data_yaml(output_root: Path) -> str:
    content = {
        "path": str(output_root),
        "train": "train/images",
        "val": "val/images",
        "test": "test/images",
        "nc": 1,
        "names": ["license_plate"],
    }
    return yaml.dump(content, sort_keys=False)


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)

    output_root = Path(config["output_root"]).resolve()
    splits = config.get("splits", ["train", "val"])
    strategy = choose_strategy(config.get("merge_strategy", "copy"), args.strategy)
    limit_per_split = args.limit_per_split or config.get("limit_per_split")

    console.rule("[bold blue]Dataset preparation")
    console.log(f"Output root: {output_root}")
    console.log(f"Splits: {splits}")
    console.log(f"Strategy: {strategy}")
    if limit_per_split:
        console.log(f"Limit per split: {limit_per_split}")
    if args.dry_run:
        console.log("[yellow]Dry run enabled (no files will be written)")

    stats: Dict[str, Dict[str, int]] = {}
    missing_labels: Dict[str, int] = {}

    for dataset in config.get("datasets", []):
        if not dataset.get("enabled", True):
            console.log(f"[yellow]Skipping disabled dataset: {dataset['name']}")
            continue

        dataset_name = dataset["name"]
        stats[dataset_name] = {split: 0 for split in splits}
        missing_labels[dataset_name] = 0

        console.log(f"[bold green]Processing dataset: {dataset_name}")
        for split in splits:
            split_cfg = dataset.get("splits", {}).get(split)
            if not split_cfg:
                console.log(f"[red]Missing split '{split}' for {dataset_name}, skipping.")
                continue

            image_dir = Path(split_cfg["images"]).expanduser().resolve()
            label_dir = Path(split_cfg["labels"]).expanduser().resolve()

            if not image_dir.exists() or not label_dir.exists():
                console.log(
                    f"[red]Image or label directory missing for {dataset_name}/{split}: "
                    f"{image_dir} | {label_dir}"
                )
                continue

            dest_images = output_root / split / "images"
            dest_labels = output_root / split / "labels"

            images = list(iter_images(image_dir))
            for image_path in tqdm(images, desc=f"{dataset_name} {split}"):
                if limit_per_split and stats[dataset_name][split] >= limit_per_split:
                    break
                label_path = label_dir / (image_path.stem + ".txt")
                if not label_path.exists():
                    missing_labels[dataset_name] += 1
                    continue

                new_image_name = sanitize_name(dataset_name, stats[dataset_name][split], image_path)
                new_label_name = Path(new_image_name).with_suffix(".txt").name

                copy_pair(
                    image_path,
                    label_path,
                    dest_images,
                    dest_labels,
                    new_image_name,
                    new_label_name,
                    strategy,
                    args.dry_run,
                )

                stats[dataset_name][split] += 1

    ensure_dir(output_root)
    data_yaml_path = output_root / "data.yaml"
    if not args.dry_run:
        data_yaml_path.write_text(build_data_yaml(output_root), encoding="utf-8")
        console.log(f"[cyan]Wrote YOLO data file -> {data_yaml_path}")

        manifest = {
            "generated_at": datetime.utcnow().isoformat(),
            "output_root": str(output_root),
            "strategy": strategy,
            "limit_per_split": limit_per_split,
            "stats": stats,
            "missing_labels": missing_labels,
            "config_source": str(args.config.resolve()),
        }
        manifest_path = output_root / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        console.log(f"[cyan]Wrote manifest -> {manifest_path}")

    console.rule("[bold blue]Summary")
    for dataset_name, split_counts in stats.items():
        total = sum(split_counts.values())
        miss = missing_labels.get(dataset_name, 0)
        console.log(
            f"{dataset_name}: total={total} "
            + " ".join(f"{split}={count}" for split, count in split_counts.items())
            + f" missing_labels={miss}"
        )

    console.log("[green]Done.")


if __name__ == "__main__":
    main()

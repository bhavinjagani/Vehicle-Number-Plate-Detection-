#!/usr/bin/env python
"""
Standalone YOLOv9 detector - runs in isolation to avoid module conflicts.
Called by app.py via subprocess.
"""
import sys
import json
import argparse
from pathlib import Path

# Setup paths BEFORE any other imports
PROJECT_ROOT = Path(__file__).parent.parent
YOLOV9_PATH = PROJECT_ROOT / "external" / "yolov9"
sys.path.insert(0, str(YOLOV9_PATH))

import torch

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Image path")
    parser.add_argument("--weights", required=True, help="Weights path")
    parser.add_argument("--conf", type=float, default=0.15, help="Confidence threshold")
    args = parser.parse_args()
    
    try:
        # Load model
        model = torch.hub.load(
            str(YOLOV9_PATH),
            'custom',
            path=args.weights,
            source='local',
            force_reload=True
        )
        model.conf = args.conf
        model.iou = 0.45
        
        # Run inference
        results = model(args.image)
        preds = results.xyxy[0].cpu().numpy().tolist()
        
        # Output as JSON
        print("RESULT:" + json.dumps(preds))
        
    except Exception as e:
        print(f"ERROR:{str(e)}", file=sys.stderr)
        print("RESULT:[]")

if __name__ == "__main__":
    main()




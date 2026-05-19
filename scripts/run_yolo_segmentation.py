# -*- coding: utf-8 -*-
"""
Created on Mon May 18 23:55:12 2026

@author: betus
"""

from pathlib import Path
import argparse
import json
import cv2
import numpy as np
from ultralytics import YOLO


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def get_image_paths(input_path: Path):
    if input_path.is_dir():
        return sorted([p for p in input_path.iterdir() if p.suffix.lower() in IMAGE_EXTS])
    return [input_path]


def run_yolo_segmentation(input_path, output_dir, model_name, conf=0.15):
    input_path = Path(input_path)
    output_dir = Path(output_dir)

    mask_dir = output_dir / "segmentation_visualizations"
    json_dir = output_dir / "segmentation_json"
    mask_dir.mkdir(parents=True, exist_ok=True)
    json_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(model_name)
    image_paths = get_image_paths(input_path)

    summary = {}

    for img_path in image_paths:
        results = model.predict(str(img_path), conf=conf, verbose=False)
        image_summary = []

        for result in results:
            annotated = result.plot()
            out_img = mask_dir / f"{img_path.stem}_seg.jpg"
            cv2.imwrite(str(out_img), annotated)

            if result.masks is None:
                summary[img_path.name] = []
                continue

            names = result.names
            boxes = result.boxes

            for i, mask_xy in enumerate(result.masks.xy):
                cls_id = int(boxes.cls[i])
                label = names[cls_id]
                score = float(boxes.conf[i])

                polygon = np.array(mask_xy).tolist()

                image_summary.append({
                    "label": label,
                    "confidence": score,
                    "polygon_xy": polygon,
                })

        json_path = json_dir / f"{img_path.stem}_segments.json"
        with open(json_path, "w") as f:
            json.dump(image_summary, f, indent=2)

        summary[img_path.name] = image_summary
        print(f"{img_path.name}: {len(image_summary)} segments detected")

    with open(output_dir / "segmentation_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Saved segmentation outputs to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="outputs/segmentation")
    parser.add_argument("--model", required=True)
    parser.add_argument("--conf", type=float, default=0.15)
    args = parser.parse_args()

    run_yolo_segmentation(
        input_path=args.input,
        output_dir=args.output,
        model_name=args.model,
        conf=args.conf,
    )
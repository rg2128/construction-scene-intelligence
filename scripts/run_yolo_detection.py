from pathlib import Path
import argparse
import json
import cv2
from ultralytics import YOLO


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def get_image_paths(input_path: Path):
    if input_path.is_dir():
        return sorted([p for p in input_path.iterdir() if p.suffix.lower() in IMAGE_EXTS])
    return [input_path]


def run_yolo_detection(input_path, output_dir, model_name="yolov8n.pt", conf=0.25):
    input_path = Path(input_path)
    output_dir = Path(output_dir)

    det_dir = output_dir / "detections"
    vis_dir = output_dir / "visualizations"
    det_dir.mkdir(parents=True, exist_ok=True)
    vis_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(model_name)
    image_paths = get_image_paths(input_path)

    summary = {}

    for img_path in image_paths:
        results = model.predict(str(img_path), conf=conf, verbose=False)
        detections = []

        for result in results:
            if result.boxes is None:
                continue

            for box in result.boxes:
                cls_id = int(box.cls[0])
                label = result.names[cls_id]
                score = float(box.conf[0])
                x1, y1, x2, y2 = [float(v) for v in box.xyxy[0]]

                detections.append({
                    "label": label,
                    "confidence": score,
                    "bbox_xyxy": [x1, y1, x2, y2]
                })

            annotated = result.plot()
            vis_path = vis_dir / f"{img_path.stem}_yolo.jpg"
            cv2.imwrite(str(vis_path), annotated)

        json_path = det_dir / f"{img_path.stem}_detections.json"
        with open(json_path, "w") as f:
            json.dump(detections, f, indent=2)

        summary[img_path.name] = detections
        print(f"{img_path.name}: {len(detections)} objects detected")

    with open(output_dir / "detections_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Saved outputs to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Image file or folder")
    parser.add_argument("--output", default="outputs", help="Output folder")
    parser.add_argument("--model", default="yolov8n.pt", help="YOLO model name/path")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    args = parser.parse_args()

    run_yolo_detection(
        input_path=args.input,
        output_dir=args.output,
        model_name=args.model,
        conf=args.conf,
    )

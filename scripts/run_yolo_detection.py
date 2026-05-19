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


def draw_detections(image, detections):
    annotated = image.copy()

    for det in detections:
        x1, y1, x2, y2 = [int(v) for v in det["bbox_xyxy"]]
        label = det["label"]
        conf = det["confidence"]

        cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 0, 255), 2)

        text = f"{label} {conf:.2f}"
        cv2.putText(
            annotated,
            text,
            (x1, max(y1 - 8, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 0, 255),
            2,
            cv2.LINE_AA,
        )

    return annotated


def box_area(bbox):
    x1, y1, x2, y2 = bbox
    return max(0, x2 - x1) * max(0, y2 - y1)

def remove_nested_boxes(
    detections,
    overlap_thresh=1,
    max_area_ratio=0.25,
    confidence_margin=0.20,
):
    filtered = []

    for i, small_det in enumerate(detections):
        sx1, sy1, sx2, sy2 = small_det["bbox_xyxy"]
        small_area = small_det["area"]
        small_conf = small_det["confidence"]

        keep = True

        for j, large_det in enumerate(detections):
            if i == j:
                continue

            lx1, ly1, lx2, ly2 = large_det["bbox_xyxy"]
            large_area = large_det["area"]
            large_conf = large_det["confidence"]

            if large_area <= small_area:
                continue

            area_ratio = small_area / large_area

            xx1 = max(sx1, lx1)
            yy1 = max(sy1, ly1)
            xx2 = min(sx2, lx2)
            yy2 = min(sy2, ly2)

            inter_w = max(0, xx2 - xx1)
            inter_h = max(0, yy2 - yy1)
            inter_area = inter_w * inter_h

            overlap_fraction_of_small = inter_area / small_area

            nested = overlap_fraction_of_small > overlap_thresh
            much_smaller = area_ratio < max_area_ratio
            much_lower_conf = small_conf < (large_conf - confidence_margin)

            if nested and much_smaller and much_lower_conf:
            #if much_lower_conf:

                keep = False
                break

        if keep:
            filtered.append(small_det)

    return filtered
def run_yolo_detection(
    input_path,
    output_dir,
    model_name="yolov11n.pt",
    conf=0.25,
    keep_classes=None,
    min_box_area=5000,
):
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
        image = cv2.imread(str(img_path))
        if image is None:
            print(f"Skipping unreadable image: {img_path}")
            continue

        results = model.predict(str(img_path), conf=conf, verbose=False)
        detections = []

        for result in results:
            if result.boxes is None:
                continue

            for box in result.boxes:
                cls_id = int(box.cls[0])
                label = result.names[cls_id]
                score = float(box.conf[0])
                bbox = [float(v) for v in box.xyxy[0]]

                if keep_classes is not None and label not in keep_classes:
                    continue

                if box_area(bbox) < min_box_area:
                    continue

                detections.append(
                    {
                        "label": label,
                        "confidence": score,
                        "bbox_xyxy": bbox,
                        "area": box_area(bbox),
                    }
                )

        detections = remove_nested_boxes(detections)

        annotated = draw_detections(image, detections)
        vis_path = vis_dir / f"{img_path.stem}_filtered_yolo.jpg"
        cv2.imwrite(str(vis_path), annotated)

        json_path = det_dir / f"{img_path.stem}_detections.json"
        with open(json_path, "w") as f:
            json.dump(detections, f, indent=2)

        summary[img_path.name] = detections
        print(f"{img_path.name}: {len(detections)} filtered objects detected")

    with open(output_dir / "detections_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Saved outputs to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Image file or folder")
    parser.add_argument("--output", default="outputs", help="Output folder")
    parser.add_argument("--model", default="yolov11n.pt", help="YOLO model name/path")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument(
        "--keep-classes",
        nargs="+",
        default=["person"],
        help="Only keep these classes",
    )
    parser.add_argument(
        "--min-box-area",
        type=float,
        default=5000,
        help="Remove small boxes, useful for partial body detections",
    )

    args = parser.parse_args()

    run_yolo_detection(
        input_path=args.input,
        output_dir=args.output,
        model_name=args.model,
        conf=args.conf,
        keep_classes=args.keep_classes,
        min_box_area=args.min_box_area,
    )
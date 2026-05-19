from pathlib import Path
import json
import cv2
from ultralytics import YOLO


PERSON_MODEL = "yolo11n.pt"

PPE_MODEL = (
    "runs/segment/outputs/training/"
    "yolo11n_seg_construction_ppe/weights/best.pt"
)

INPUT_PATH = "data/sample_images"
OUTPUT_DIR = Path("outputs/person_ppe_pipeline")

PPE_CLASSES = ["helmet", "vest", "without_helmet", "without_vest"]


def compute_iou(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)

    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)

    union = area_a + area_b - inter

    if union == 0:
        return 0

    return inter / union


def get_adaptive_crop(image, current_box, all_boxes):
    img_h, img_w = image.shape[:2]

    x1, y1, x2, y2 = current_box
    box_w = x2 - x1
    box_h = y2 - y1

    base_pad = int(0.15 * max(box_w, box_h))
    base_pad = max(10, min(base_pad, 60))

    crowded = False

    for other_box in all_boxes:
        if other_box == current_box:
            continue

        iou = compute_iou(current_box, other_box)

        if iou > 0.05:
            crowded = True
            break

    if crowded:
        pad = max(5, int(base_pad * 0.35))
    else:
        pad = base_pad

    x1p = max(0, x1 - pad)
    y1p = max(0, y1 - pad)
    x2p = min(img_w, x2 + pad)
    y2p = min(img_h, y2 + pad)

    crop = image[y1p:y2p, x1p:x2p]

    return crop, [x1p, y1p, x2p, y2p], pad


def summarize_ppe(ppe_results):
    found = {cls: [] for cls in PPE_CLASSES}

    for result in ppe_results:
        if result.boxes is None:
            continue

        for box in result.boxes:
            cls_id = int(box.cls[0])
            label = result.names[cls_id]
            conf = float(box.conf[0])

            if label in found:
                found[label].append(conf)

    helmet = len(found["helmet"]) > 0
    vest = len(found["vest"]) > 0
    without_helmet = len(found["without_helmet"]) > 0
    without_vest = len(found["without_vest"]) > 0

    violations = []

    if without_helmet and not helmet:
        violations.append("missing_helmet")

    if without_vest and not vest:
        violations.append("missing_vest")

    return {
        "helmet_detected": helmet,
        "vest_detected": vest,
        "without_helmet_detected": without_helmet,
        "without_vest_detected": without_vest,
        "violations": violations,
        "raw_ppe_confidences": found,
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    person_model = YOLO(PERSON_MODEL)
    ppe_model = YOLO(PPE_MODEL)

    image_paths = [
        p for p in Path(INPUT_PATH).glob("*")
        if p.suffix.lower() in [".jpg", ".jpeg", ".png"]
    ]

    full_summary = {}

    for img_path in image_paths:

        image = cv2.imread(str(img_path))
        if image is None:
            continue

        person_results = person_model.predict(
            str(img_path),
            conf=0.25,
            classes=[0],   # COCO person class
            verbose=False,
        )

        annotated = image.copy()
        image_summary = []

        person_boxes = []

        for result in person_results:
            if result.boxes is None:
                continue

            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                person_conf = float(box.conf[0])

                person_boxes.append(([x1, y1, x2, y2], person_conf))

        person_count = 0

        for current_box, person_conf in person_boxes:

            person_count += 1

            x1, y1, x2, y2 = current_box

            crop, padded_box, pad_used = get_adaptive_crop(
                image=image,
                current_box=current_box,
                all_boxes=[b for b, _ in person_boxes],
            )

            if crop.size == 0:
                continue

            # -----------------------------------
            # Save raw crop
            # -----------------------------------

            raw_crop_path = (
                OUTPUT_DIR /
                f"{img_path.stem}_worker_{person_count:02d}_crop_raw.jpg"
            )

            cv2.imwrite(str(raw_crop_path), crop)

            # -----------------------------------
            # Resize small crops
            # -----------------------------------

            h, w = crop.shape[:2]

            target_h = 256

            scale = max(1.0, target_h / h)

            new_w = int(w * scale)
            new_h = int(h * scale)

            crop_resized = cv2.resize(
                crop,
                (new_w, new_h),
                interpolation=cv2.INTER_CUBIC,
            )

            crop_path = (
                OUTPUT_DIR /
                f"{img_path.stem}_worker_{person_count:02d}_crop_resized.jpg"
            )

            cv2.imwrite(str(crop_path), crop_resized)

            # -----------------------------------
            # PPE segmentation
            # -----------------------------------

            ppe_results = ppe_model.predict(
                str(crop_path),
                conf=0.05,
                imgsz=640,
                verbose=False,
            )

            seg_path = (
                OUTPUT_DIR /
                f"{img_path.stem}_worker_{person_count:02d}_ppe_seg.jpg"
            )

            for ppe_result in ppe_results:
                ppe_annotated = ppe_result.plot(boxes=False)
                cv2.imwrite(str(seg_path), ppe_annotated)

            ppe_summary = summarize_ppe(ppe_results)

            # -----------------------------------
            # Final visualization
            # -----------------------------------

            if ppe_summary["violations"]:
                box_color = (0, 0, 255)
                status = ",".join(ppe_summary["violations"])
            else:
                box_color = (0, 255, 0)
                status = "ppe_ok"

            cv2.rectangle(
                annotated,
                (x1, y1),
                (x2, y2),
                box_color,
                3,
            )

            label = f"worker_{person_count}: {status}"

            cv2.putText(
                annotated,
                label,
                (x1, max(y1 - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                box_color,
                2,
                cv2.LINE_AA,
            )

            image_summary.append(
                {
                    "worker_id": person_count,
                    "person_confidence": person_conf,
                    "person_bbox_xyxy": [x1, y1, x2, y2],
                    "padded_crop_box_xyxy": padded_box,
                    "padding_used_px": pad_used,
                    "raw_crop_path": str(raw_crop_path),
                    "resized_crop_path": str(crop_path),
                    "crop_original_size_hw": [h, w],
                    "crop_resized_size_hw": [new_h, new_w],
                    "resize_scale": scale,
                    "ppe_segmentation_path": str(seg_path),
                    "ppe_summary": ppe_summary,
                }
            )

        out_path = OUTPUT_DIR / f"{img_path.stem}_pipeline.jpg"

        cv2.imwrite(str(out_path), annotated)

        full_summary[img_path.name] = image_summary

        print(f"{img_path.name}: {len(image_summary)} workers analyzed")

    summary_path = OUTPUT_DIR / "person_ppe_summary.json"

    with open(summary_path, "w") as f:
        json.dump(full_summary, f, indent=2)

    print(f"Saved pipeline outputs to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
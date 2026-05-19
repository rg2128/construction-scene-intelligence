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

OUTPUT_DIR = Path("outputs/person_ppe_pipeline_fixed_padding")

PPE_CLASSES = ["helmet", "vest", "without_helmet", "without_vest"]


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

        person_count = 0

        for result in person_results:

            if result.boxes is None:
                continue

            for box in result.boxes:

                person_count += 1

                x1, y1, x2, y2 = map(int, box.xyxy[0])

                person_conf = float(box.conf[0])

                # -----------------------------------
                # Fixed equal padding
                # -----------------------------------

                img_h, img_w = image.shape[:2]

                pad = 40

                x1p = max(0, x1 - pad)
                y1p = max(0, y1 - pad)

                x2p = min(img_w, x2 + pad)
                y2p = min(img_h, y2 + pad)

                crop = image[y1p:y2p, x1p:x2p]

                if crop.size == 0:
                    continue

                # -----------------------------------
                # Save crop
                # -----------------------------------

                crop_path = (
                    OUTPUT_DIR /
                    f"{img_path.stem}_worker_{person_count:02d}_crop_fixedpad.jpg"
                )

                cv2.imwrite(str(crop_path), crop)

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

                    status = ",".join(
                        ppe_summary["violations"]
                    )

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
                        "padded_crop_box_xyxy": [x1p, y1p, x2p, y2p],
                        "padding_used_px": pad,
                        "crop_path": str(crop_path),
                        "ppe_segmentation_path": str(seg_path),
                        "ppe_summary": ppe_summary,
                    }
                )

        out_path = (
            OUTPUT_DIR /
            f"{img_path.stem}_pipeline.jpg"
        )

        cv2.imwrite(str(out_path), annotated)

        full_summary[img_path.name] = image_summary

        print(
            f"{img_path.name}: "
            f"{len(image_summary)} workers analyzed"
        )

    summary_path = (
        OUTPUT_DIR /
        "person_ppe_summary.json"
    )

    with open(summary_path, "w") as f:

        json.dump(
            full_summary,
            f,
            indent=2,
        )

    print(
        f"Saved pipeline outputs to: "
        f"{OUTPUT_DIR}"
    )


if __name__ == "__main__":

    main()
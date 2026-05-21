from pathlib import Path
import argparse
import json
import cv2
import yaml
from ultralytics import YOLO


def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def make_padded_crop(image, box, pad):
    img_h, img_w = image.shape[:2]

    x1, y1, x2, y2 = box

    x1p = max(0, x1 - pad)
    y1p = max(0, y1 - pad)
    x2p = min(img_w, x2 + pad)
    y2p = min(img_h, y2 + pad)

    crop = image[y1p:y2p, x1p:x2p]

    return crop, [x1p, y1p, x2p, y2p]


def resize_crop_if_needed(crop, target_height=256):
    h, w = crop.shape[:2]

    scale = max(1.0, target_height / h)

    new_w = int(w * scale)
    new_h = int(h * scale)

    if scale == 1.0:
        return crop, [h, w], [h, w], scale

    resized = cv2.resize(
        crop,
        (new_w, new_h),
        interpolation=cv2.INTER_CUBIC,
    )

    return resized, [h, w], [new_h, new_w], scale


def summarize_single_ppe_run(ppe_results, ppe_classes):
    found = {cls: [] for cls in ppe_classes}

    for result in ppe_results:

        if result.boxes is None:
            continue

        for box in result.boxes:

            cls_id = int(box.cls[0])
            label = result.names[cls_id]
            conf = float(box.conf[0])

            if label in found:
                found[label].append(conf)

    summary = {}

    for cls in ppe_classes:

        vals = found[cls]

        summary[cls] = {
            "detected": len(vals) > 0,
            "max_conf": max(vals) if vals else 0.0,
            "mean_conf": sum(vals) / len(vals) if vals else 0.0,
            "n_instances": len(vals),
        }

    return summary


def aggregate_padding_results(per_padding_results, ppe_classes):

    agg = {}

    for cls in ppe_classes:

        max_confs = []
        detected_count = 0

        for run in per_padding_results:

            cls_result = run["ppe_summary"][cls]

            max_confs.append(
                cls_result["max_conf"]
            )

            if cls_result["detected"]:
                detected_count += 1

        agg[cls] = {
            "mean_max_conf_across_paddings":
                sum(max_confs) / len(max_confs),

            "max_conf_across_paddings":
                max(max_confs),

            "detected_in_n_paddings":
                detected_count,

            "detected_fraction":
                detected_count / len(per_padding_results),
        }

    helmet_score = agg["helmet"]["mean_max_conf_across_paddings"]
    vest_score = agg["vest"]["mean_max_conf_across_paddings"]

    without_helmet_score = (
        agg["without_helmet"]["mean_max_conf_across_paddings"]
    )

    without_vest_score = (
        agg["without_vest"]["mean_max_conf_across_paddings"]
    )

    helmet_detected = (
        helmet_score >= 0.10
        or agg["helmet"]["detected_fraction"] >= 0.40
    )

    vest_detected = (
        vest_score >= 0.10
        or agg["vest"]["detected_fraction"] >= 0.40
    )

    without_helmet_detected = (
        without_helmet_score >= 0.10
        or agg["without_helmet"]["detected_fraction"] >= 0.40
    )

    without_vest_detected = (
        without_vest_score >= 0.10
        or agg["without_vest"]["detected_fraction"] >= 0.40
    )

    violations = []

    if without_helmet_detected and not helmet_detected:
        violations.append("missing_helmet")

    if without_vest_detected and not vest_detected:
        violations.append("missing_vest")

    final = {
        "helmet_detected": helmet_detected,
        "vest_detected": vest_detected,
        "without_helmet_detected": without_helmet_detected,
        "without_vest_detected": without_vest_detected,
        "violations": violations,
        "ensemble_scores": agg,
    }

    return final


def draw_worker_result(image, box, worker_id, final_summary):

    x1, y1, x2, y2 = box

    if final_summary["violations"]:

        color = (0, 0, 255)

        status = ",".join(
            final_summary["violations"]
        )

    else:

        color = (0, 255, 0)

        status = "ppe_ok"

    cv2.rectangle(
        image,
        (x1, y1),
        (x2, y2),
        color,
        3,
    )

    label = f"worker_{worker_id}: {status}"

    cv2.putText(
        image,
        label,
        (x1, max(y1 - 10, 20)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        color,
        2,
        cv2.LINE_AA,
    )


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        default="configs/ppe_pipeline.yaml",
        help="Path to config YAML",
    )

    args = parser.parse_args()

    cfg = load_config(args.config)

    output_dir = Path(cfg["output_dir"])

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    person_model = YOLO(
        cfg["person_model"]
    )

    ppe_model = YOLO(
        cfg["ppe_model"]
    )

    image_paths = [
        p for p in Path(cfg["input_path"]).glob("*")
        if p.suffix.lower() in [".jpg", ".jpeg", ".png"]
    ]

    full_summary = {}

    for img_path in image_paths:

        image = cv2.imread(str(img_path))

        if image is None:
            continue

        person_results = person_model.predict(
            str(img_path),
            conf=cfg["person_conf"],
            classes=[0],
            verbose=False,
        )

        annotated = image.copy()

        image_summary = []

        person_boxes = []

        for result in person_results:

            if result.boxes is None:
                continue

            for box in result.boxes:

                x1, y1, x2, y2 = map(
                    int,
                    box.xyxy[0]
                )

                person_conf = float(
                    box.conf[0]
                )

                person_boxes.append(
                    (
                        [x1, y1, x2, y2],
                        person_conf,
                    )
                )

        for worker_idx, (
            person_box,
            person_conf,
        ) in enumerate(
            person_boxes,
            start=1,
        ):

            per_padding_results = []

            for pad in cfg["padding_values"]:

                crop, padded_box = make_padded_crop(
                    image=image,
                    box=person_box,
                    pad=pad,
                )

                if crop.size == 0:
                    continue

                crop_for_model, original_hw, model_hw, scale = (
                    resize_crop_if_needed(
                        crop,
                        target_height=cfg[
                            "small_crop_target_height"
                        ],
                    )
                )

                crop_path = (
                    output_dir
                    / (
                        f"{img_path.stem}"
                        f"_worker_{worker_idx:02d}"
                        f"_pad_{pad:03d}_crop.jpg"
                    )
                )

                cv2.imwrite(
                    str(crop_path),
                    crop_for_model,
                )

                ppe_results = ppe_model.predict(
                    str(crop_path),
                    conf=cfg["ppe_conf"],
                    imgsz=cfg["ppe_imgsz"],
                    verbose=False,
                )

                seg_path = (
                    output_dir
                    / (
                        f"{img_path.stem}"
                        f"_worker_{worker_idx:02d}"
                        f"_pad_{pad:03d}_seg.jpg"
                    )
                )

                for ppe_result in ppe_results:

                    ppe_annotated = ppe_result.plot(
                        boxes=False
                    )

                    cv2.imwrite(
                        str(seg_path),
                        ppe_annotated,
                    )

                ppe_summary = summarize_single_ppe_run(
                    ppe_results,
                    cfg["ppe_classes"],
                )

                per_padding_results.append(
                    {
                        "padding_px": pad,
                        "padded_box_xyxy": padded_box,
                        "crop_path": str(crop_path),
                        "segmentation_path": str(seg_path),
                        "crop_original_size_hw": original_hw,
                        "crop_model_input_size_hw": model_hw,
                        "resize_scale": scale,
                        "ppe_summary": ppe_summary,
                    }
                )

            final_summary = aggregate_padding_results(
                per_padding_results,
                cfg["ppe_classes"],
            )

            draw_worker_result(
                annotated,
                person_box,
                worker_idx,
                final_summary,
            )

            image_summary.append(
                {
                    "worker_id": worker_idx,
                    "person_confidence": person_conf,
                    "person_bbox_xyxy": person_box,
                    "padding_values_tested":
                        cfg["padding_values"],
                    "per_padding_results":
                        per_padding_results,
                    "final_ppe_summary":
                        final_summary,
                }
            )

        out_path = (
            output_dir
            / f"{img_path.stem}_pipeline_ensemble.jpg"
        )

        cv2.imwrite(
            str(out_path),
            annotated,
        )

        full_summary[img_path.name] = image_summary

        print(
            f"{img_path.name}: "
            f"{len(image_summary)} workers analyzed"
        )

    summary_path = (
        output_dir
        / "person_ppe_padding_ensemble_summary.json"
    )

    with open(summary_path, "w") as f:

        json.dump(
            full_summary,
            f,
            indent=2,
        )

    # -----------------------------------
    # Human-readable report
    # -----------------------------------

    report_path = (
        output_dir
        / "ppe_ensemble_report.txt"
    )

    with open(report_path, "w") as f:

        f.write(
            "PPE ENSEMBLE ANALYSIS REPORT\n"
        )

        f.write("=" * 80 + "\n\n")

        for image_name, workers in full_summary.items():

            f.write(
                f"IMAGE: {image_name}\n"
            )

            f.write("-" * 80 + "\n")

            if len(workers) == 0:

                f.write(
                    "No workers detected.\n\n"
                )

                continue

            for worker in workers:

                worker_id = worker["worker_id"]

                bbox = worker[
                    "person_bbox_xyxy"
                ]

                f.write(
                    f"\nWorker {worker_id}\n"
                )

                f.write(
                    f"Bounding Box: {bbox}\n"
                )

                final_summary = worker[
                    "final_ppe_summary"
                ]

                f.write(
                    "\nFinal PPE Decision:\n"
                )

                f.write(
                    f"  Helmet Detected: "
                    f"{final_summary['helmet_detected']}\n"
                )

                f.write(
                    f"  Vest Detected: "
                    f"{final_summary['vest_detected']}\n"
                )

                f.write(
                    f"  Without Helmet: "
                    f"{final_summary['without_helmet_detected']}\n"
                )

                f.write(
                    f"  Without Vest: "
                    f"{final_summary['without_vest_detected']}\n"
                )

                if final_summary["violations"]:

                    f.write(
                        f"  Violations: "
                        f"{', '.join(final_summary['violations'])}\n"
                    )

                else:

                    f.write(
                        "  Violations: none\n"
                    )

                f.write(
                    "\nPadding Ensemble Details:\n"
                )

                scores = final_summary[
                    "ensemble_scores"
                ]

                for cls_name, cls_result in scores.items():

                    f.write(
                        f"\n  {cls_name}\n"
                    )

                    f.write(
                        f"    Mean Max Confidence: "
                        f"{cls_result['mean_max_conf_across_paddings']:.3f}\n"
                    )

                    f.write(
                        f"    Max Confidence: "
                        f"{cls_result['max_conf_across_paddings']:.3f}\n"
                    )

                    f.write(
                        f"    Detected Fraction: "
                        f"{cls_result['detected_fraction']:.2f}\n"
                    )

                    f.write(
                        f"    Detected In "
                        f"{cls_result['detected_in_n_paddings']} / "
                        f"{len(worker['padding_values_tested'])} paddings\n"
                    )

                f.write(
                    "\nPer-Padding Breakdown:\n"
                )

                for pad_result in worker[
                    "per_padding_results"
                ]:

                    pad = pad_result[
                        "padding_px"
                    ]

                    f.write(
                        f"\n    Padding = {pad}px\n"
                    )

                    for cls_name, cls_info in (
                        pad_result["ppe_summary"].items()
                    ):

                        f.write(
                            f"      {cls_name}: "
                            f"detected={cls_info['detected']} "
                            f"max_conf={cls_info['max_conf']:.3f} "
                            f"mean_conf={cls_info['mean_conf']:.3f}\n"
                        )

                f.write("\n")

            f.write("\n\n")

    print(
        f"Saved report to: {report_path}"
    )

    print(
        f"Saved ensemble outputs to: {output_dir}"
    )


if __name__ == "__main__":
    main()

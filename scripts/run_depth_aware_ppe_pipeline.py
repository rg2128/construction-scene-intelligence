from pathlib import Path
import argparse
import json
import cv2
import yaml
import torch
import numpy as np
from ultralytics import YOLO


def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def build_scene_summary(image_summary):
    n_workers = len(image_summary)

    workers_with_helmet = 0
    workers_with_vest = 0
    workers_with_violations = 0

    missing_helmet = 0
    missing_vest = 0

    for worker in image_summary:
        final = worker["final_ppe_summary"]

        if final["helmet_detected"]:
            workers_with_helmet += 1

        if final["vest_detected"]:
            workers_with_vest += 1

        if final["violations"]:
            workers_with_violations += 1

        if "missing_helmet" in final["violations"]:
            missing_helmet += 1

        if "missing_vest" in final["violations"]:
            missing_vest += 1

    if n_workers == 0:
        risk_level = "unknown"
    elif workers_with_violations == 0:
        risk_level = "low"
    elif workers_with_violations / n_workers < 0.5:
        risk_level = "medium"
    else:
        risk_level = "high"

    return {
        "n_workers": n_workers,
        "workers_with_helmet": workers_with_helmet,
        "workers_with_vest": workers_with_vest,
        "workers_with_violations": workers_with_violations,
        "missing_helmet": missing_helmet,
        "missing_vest": missing_vest,
        "risk_level": risk_level,
    }


def load_depth_model(model_type="DPT_Hybrid"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    midas = torch.hub.load(
        "intel-isl/MiDaS",
        model_type,
        trust_repo=True,
    )

    midas.to(device)
    midas.eval()

    transforms = torch.hub.load(
        "intel-isl/MiDaS",
        "transforms",
        trust_repo=True,
    )

    if model_type in ["DPT_Large", "DPT_Hybrid"]:
        transform = transforms.dpt_transform
    else:
        transform = transforms.small_transform

    return midas, transform, device


def estimate_depth(image_bgr, midas, transform, device):
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    input_batch = transform(image_rgb).to(device)

    with torch.no_grad():
        prediction = midas(input_batch)

        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=image_rgb.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()

    return prediction.cpu().numpy()


def median_depth_in_box(depth_map, box):
    x1, y1, x2, y2 = box

    crop = depth_map[y1:y2, x1:x2]

    if crop.size == 0:
        return None

    return float(np.median(crop))


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


def compute_depth_keep_indices(
    ppe_result,
    crop_depth_resized,
    worker_depth,
    depth_tolerance_std=1.0,
):
    if worker_depth is None:
        if ppe_result.boxes is None:
            return []

        return list(range(len(ppe_result.boxes)))

    if ppe_result.masks is None or ppe_result.boxes is None:
        return []

    depth_std = float(np.std(crop_depth_resized))

    if depth_std < 1e-6:
        return list(range(len(ppe_result.boxes)))

    keep_indices = []

    for i, mask_xy in enumerate(ppe_result.masks.xy):
        pts = np.array(mask_xy, dtype=np.int32)

        mask = np.zeros(crop_depth_resized.shape, dtype=np.uint8)
        cv2.fillPoly(mask, [pts], 1)

        mask_depth_values = crop_depth_resized[mask == 1]

        if mask_depth_values.size == 0:
            continue

        mask_depth = float(np.median(mask_depth_values))

        z_diff = abs(mask_depth - worker_depth) / depth_std

        if z_diff <= depth_tolerance_std:
            keep_indices.append(i)

    return keep_indices


def summarize_single_ppe_run(ppe_results, ppe_classes):
    found = {cls: [] for cls in ppe_classes}

    for result in ppe_results:
        if result.boxes is None:
            continue

        keep_indices = getattr(result, "keep_depth_indices", None)

        if keep_indices is None:
            keep_indices = list(range(len(result.boxes)))

        for i in keep_indices:
            cls_id = int(result.boxes.cls[i])
            label = result.names[cls_id]
            conf = float(result.boxes.conf[i])

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


def aggregate_padding_results(
    per_padding_results,
    ppe_classes,
    cfg,
):
    if len(per_padding_results) == 0:
        return {
            "helmet_detected": False,
            "vest_detected": False,
            "without_helmet_detected": False,
            "without_vest_detected": False,
            "violations": ["ppe_not_evaluated"],
            "ensemble_scores": {},
        }

    agg = {}

    for cls in ppe_classes:
        max_confs = []
        detected_count = 0

        for run in per_padding_results:
            cls_result = run["ppe_summary"][cls]
            max_confs.append(cls_result["max_conf"])

            if cls_result["detected"]:
                detected_count += 1

        agg[cls] = {
            "mean_max_conf_across_paddings": (
                sum(max_confs) / len(max_confs)
            ),
            "max_conf_across_paddings": max(max_confs),
            "detected_in_n_paddings": detected_count,
            "detected_fraction": (
                detected_count / len(per_padding_results)
            ),
        }

    min_mean_conf = cfg.get("ensemble_min_mean_conf", 0.10)
    min_detected_fraction = cfg.get("ensemble_min_detected_fraction", 0.40)
    exclusive_margin = cfg.get("exclusive_class_margin", 0.15)

    helmet_score = agg["helmet"]["mean_max_conf_across_paddings"]
    vest_score = agg["vest"]["mean_max_conf_across_paddings"]

    without_helmet_score = (
        agg["without_helmet"]["mean_max_conf_across_paddings"]
    )

    without_vest_score = (
        agg["without_vest"]["mean_max_conf_across_paddings"]
    )

    helmet_fraction = agg["helmet"]["detected_fraction"]
    vest_fraction = agg["vest"]["detected_fraction"]

    without_helmet_fraction = agg["without_helmet"]["detected_fraction"]
    without_vest_fraction = agg["without_vest"]["detected_fraction"]

    helmet_supported = (
        helmet_score >= min_mean_conf
        or helmet_fraction >= min_detected_fraction
    )

    vest_supported = (
        vest_score >= min_mean_conf
        or vest_fraction >= min_detected_fraction
    )

    without_helmet_supported = (
        without_helmet_score >= min_mean_conf
        or without_helmet_fraction >= min_detected_fraction
    )

    without_vest_supported = (
        without_vest_score >= min_mean_conf
        or without_vest_fraction >= min_detected_fraction
    )

    # -----------------------------
    # Mutually exclusive decisions
    # -----------------------------
    # helmet vs without_helmet
    # vest vs without_vest
    #
    # A negative class only wins if it beats
    # the positive class by a configurable margin.
    # This prevents false violations when both classes
    # are weakly or moderately detected.
    # -----------------------------

    helmet_detected = (
        helmet_supported
        and helmet_score >= without_helmet_score - exclusive_margin
    )

    without_helmet_detected = (
        without_helmet_supported
        and without_helmet_score > helmet_score + exclusive_margin
    )

    vest_detected = (
        vest_supported
        and vest_score >= without_vest_score - exclusive_margin
    )

    without_vest_detected = (
        without_vest_supported
        and without_vest_score > vest_score + exclusive_margin
    )

    violations = []

    if without_helmet_detected:
        violations.append("missing_helmet")

    if without_vest_detected:
        violations.append("missing_vest")

    if not helmet_detected and not without_helmet_detected:
        violations.append("uncertain_helmet")

    if not vest_detected and not without_vest_detected:
        violations.append("uncertain_vest")

    return {
        "helmet_detected": helmet_detected,
        "vest_detected": vest_detected,
        "without_helmet_detected": without_helmet_detected,
        "without_vest_detected": without_vest_detected,
        "violations": violations,
        "ensemble_scores": agg,
        "decision_thresholds": {
            "ensemble_min_mean_conf": min_mean_conf,
            "ensemble_min_detected_fraction": min_detected_fraction,
            "exclusive_class_margin": exclusive_margin,
        },
    }




def draw_filtered_segmentation(
    image,
    ppe_result,
    keep_indices,
):
    vis = image.copy()

    if ppe_result.masks is None or ppe_result.boxes is None:
        return vis

    overlay = vis.copy()

    for i in keep_indices:
        mask_xy = ppe_result.masks.xy[i]

        pts = np.array(mask_xy, dtype=np.int32)

        cls_id = int(ppe_result.boxes.cls[i])
        label = ppe_result.names[cls_id]
        conf = float(ppe_result.boxes.conf[i])

        cv2.fillPoly(overlay, [pts], (255, 0, 255))

        cv2.polylines(
            vis,
            [pts],
            isClosed=True,
            color=(255, 0, 255),
            thickness=2,
        )

        x_text = int(np.min(pts[:, 0]))
        y_text = int(np.min(pts[:, 1]))

        cv2.putText(
            vis,
            f"{label} {conf:.2f}",
            (x_text, max(y_text - 8, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 0, 255),
            2,
            cv2.LINE_AA,
        )

    vis = cv2.addWeighted(
        overlay,
        0.35,
        vis,
        0.65,
        0,
    )

    return vis


def draw_worker_result(
    image,
    box,
    worker_id,
    final_summary,
):
    x1, y1, x2, y2 = box

    if final_summary["violations"]:
        color = (0, 0, 255)
        status = ",".join(final_summary["violations"])
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


def write_human_report(report_path, full_summary):
    with open(report_path, "w") as f:
        f.write("PPE ENSEMBLE ANALYSIS REPORT\n")
        f.write("=" * 80 + "\n\n")

        for image_name, image_data in full_summary.items():
            scene_summary = image_data["scene_summary"]
            workers = image_data["workers"]

            f.write(f"IMAGE: {image_name}\n")
            f.write("-" * 80 + "\n")

            f.write("\nScene Summary:\n")

            f.write(
                f"  Workers detected: "
                f"{scene_summary['n_workers']}\n"
            )

            f.write(
                f"  Workers with helmet: "
                f"{scene_summary['workers_with_helmet']}\n"
            )

            f.write(
                f"  Workers with vest: "
                f"{scene_summary['workers_with_vest']}\n"
            )

            f.write(
                f"  Workers with violations: "
                f"{scene_summary['workers_with_violations']}\n"
            )

            f.write(
                f"  Missing helmet: "
                f"{scene_summary['missing_helmet']}\n"
            )

            f.write(
                f"  Missing vest: "
                f"{scene_summary['missing_vest']}\n"
            )

            f.write(
                f"  Risk level: "
                f"{scene_summary['risk_level']}\n\n"
            )

            if len(workers) == 0:
                f.write("No workers detected.\n\n")
                continue

            for worker in workers:
                f.write(f"\nWorker {worker['worker_id']}\n")

                f.write(
                    f"Bounding Box: "
                    f"{worker['person_bbox_xyxy']}\n"
                )

                final_summary = worker["final_ppe_summary"]

                f.write("\nFinal PPE Decision:\n")

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
                    f.write("  Violations: none\n")

                f.write("\nPadding Ensemble Details:\n")

                scores = final_summary["ensemble_scores"]

                for cls_name, cls_result in scores.items():
                    f.write(f"\n  {cls_name}\n")

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

                f.write("\n")


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
    output_dir.mkdir(parents=True, exist_ok=True)

    person_model = YOLO(cfg["person_model"])
    ppe_model = YOLO(cfg["ppe_model"])

    use_depth = cfg.get("use_depth_filtering", False)

    if use_depth:
        depth_model, depth_transform, depth_device = load_depth_model(
            cfg.get("depth_model", "DPT_Hybrid")
        )
    else:
        depth_model = None
        depth_transform = None
        depth_device = None

    image_paths = [
        p for p in Path(cfg["input_path"]).glob("*")
        if p.suffix.lower() in [".jpg", ".jpeg", ".png"]
    ]

    full_summary = {}

    for img_path in image_paths:
        image = cv2.imread(str(img_path))

        if image is None:
            continue

        image_output_dir = output_dir / img_path.stem
        image_output_dir.mkdir(parents=True, exist_ok=True)

        if use_depth:
            full_depth = estimate_depth(
                image,
                depth_model,
                depth_transform,
                depth_device,
            )
        else:
            full_depth = None

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
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                person_conf = float(box.conf[0])

                person_boxes.append(
                    ([x1, y1, x2, y2], person_conf)
                )

        for worker_idx, (person_box, person_conf) in enumerate(
            person_boxes,
            start=1,
        ):
            worker_depth = None

            if use_depth:
                worker_depth = median_depth_in_box(
                    full_depth,
                    person_box,
                )

            per_padding_results = []

            for pad in cfg["padding_values"]:
                crop, padded_box = make_padded_crop(
                    image=image,
                    box=person_box,
                    pad=pad,
                )

                if crop.size == 0:
                    continue

                (
                    crop_for_model,
                    original_hw,
                    model_hw,
                    scale,
                ) = resize_crop_if_needed(
                    crop,
                    target_height=cfg["small_crop_target_height"],
                )

                crop_path = (
                    image_output_dir
                    / f"worker_{worker_idx:02d}_pad_{pad:03d}_crop.jpg"
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

                depth_filtering_used = False

                if use_depth:
                    crop_depth = full_depth[
                        padded_box[1]:padded_box[3],
                        padded_box[0]:padded_box[2],
                    ]

                    crop_depth_resized = cv2.resize(
                        crop_depth,
                        (
                            crop_for_model.shape[1],
                            crop_for_model.shape[0],
                        ),
                        interpolation=cv2.INTER_CUBIC,
                    )

                    depth_norm = cv2.normalize(
                        crop_depth_resized,
                        None,
                        0,
                        255,
                        cv2.NORM_MINMAX,
                    ).astype(np.uint8)

                    depth_color = cv2.applyColorMap(
                        depth_norm,
                        cv2.COLORMAP_INFERNO,
                    )

                    depth_vis_path = (
                        image_output_dir
                        / f"worker_{worker_idx:02d}_pad_{pad:03d}_depth.jpg"
                    )

                    cv2.imwrite(
                        str(depth_vis_path),
                        depth_color,
                    )

                    for ppe_result in ppe_results:
                        keep_indices = compute_depth_keep_indices(
                            ppe_result=ppe_result,
                            crop_depth_resized=crop_depth_resized,
                            worker_depth=worker_depth,
                            depth_tolerance_std=cfg.get(
                                "depth_tolerance_std",
                                1.0,
                            ),
                        )

                        ppe_result.keep_depth_indices = keep_indices

                    depth_filtering_used = True

                else:
                    for ppe_result in ppe_results:
                        if ppe_result.boxes is not None:
                            ppe_result.keep_depth_indices = list(
                                range(len(ppe_result.boxes))
                            )
                        else:
                            ppe_result.keep_depth_indices = []

                    depth_vis_path = None
                    crop_depth_resized = None

                for i, ppe_result in enumerate(ppe_results):
                    seg_path = (
                        image_output_dir
                        / (
                            f"worker_{worker_idx:02d}"
                            f"_pad_{pad:03d}"
                            f"_seg_{i:02d}.jpg"
                        )
                    )

                    ppe_annotated = draw_filtered_segmentation(
                        crop_for_model,
                        ppe_result,
                        ppe_result.keep_depth_indices,
                    )

                    cv2.imwrite(
                        str(seg_path),
                        ppe_annotated,
                    )

                    if use_depth:
                        depth_overlay = draw_filtered_segmentation(
                            depth_color,
                            ppe_result,
                            ppe_result.keep_depth_indices,
                        )

                        depth_overlay_path = (
                            image_output_dir
                            / (
                                f"worker_{worker_idx:02d}"
                                f"_pad_{pad:03d}"
                                f"_depth_overlay_{i:02d}.jpg"
                            )
                        )

                        cv2.imwrite(
                            str(depth_overlay_path),
                            depth_overlay,
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
                        "crop_original_size_hw": original_hw,
                        "crop_model_input_size_hw": model_hw,
                        "resize_scale": scale,
                        "worker_depth_median": worker_depth,
                        "depth_filtering_used": depth_filtering_used,
                        "depth_visualization_path": (
                            str(depth_vis_path)
                            if depth_vis_path is not None
                            else None
                        ),
                        "crop_depth_mean": (
                            float(np.mean(crop_depth_resized))
                            if crop_depth_resized is not None
                            else None
                        ),
                        "crop_depth_std": (
                            float(np.std(crop_depth_resized))
                            if crop_depth_resized is not None
                            else None
                        ),
                        "ppe_summary": ppe_summary,
                    }
                )

            final_summary = aggregate_padding_results(
                per_padding_results,
                cfg["ppe_classes"],
                cfg,
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
                    "padding_values_tested": cfg["padding_values"],
                    "worker_depth_median": worker_depth,
                    "per_padding_results": per_padding_results,
                    "final_ppe_summary": final_summary,
                }
            )

        out_path = image_output_dir / "pipeline_ensemble.jpg"

        cv2.imwrite(
            str(out_path),
            annotated,
        )

        scene_summary = build_scene_summary(image_summary)

        image_result = {
            "scene_summary": scene_summary,
            "workers": image_summary,
        }

        full_summary[img_path.name] = image_result

        image_json_path = image_output_dir / "summary.json"

        with open(image_json_path, "w") as f:
            json.dump(
                image_result,
                f,
                indent=2,
            )

        image_report_path = image_output_dir / "report.txt"

        write_human_report(
            image_report_path,
            {img_path.name: image_result},
        )

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

    report_path = output_dir / "ppe_ensemble_report.txt"

    write_human_report(
        report_path,
        full_summary,
    )

    print(f"Saved report to: {report_path}")
    print(f"Saved ensemble outputs to: {output_dir}")


if __name__ == "__main__":
    main()
from pathlib import Path
import argparse
import json
import cv2
import numpy as np
import torch


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def get_image_paths(input_path: Path):
    if input_path.is_dir():
        return sorted([p for p in input_path.iterdir() if p.suffix.lower() in IMAGE_EXTS])
    return [input_path]


def normalize_depth(depth):
    depth_min = depth.min()
    depth_max = depth.max()

    if depth_max - depth_min < 1e-6:
        return np.zeros_like(depth, dtype=np.uint8)

    depth_norm = (depth - depth_min) / (depth_max - depth_min)
    depth_img = (depth_norm * 255).astype(np.uint8)

    return depth_img


def run_depth(input_path, output_dir, model_type="DPT_Hybrid"):
    input_path = Path(input_path)
    output_dir = Path(output_dir)

    depth_dir = output_dir / "depth_maps"
    json_dir = output_dir / "depth_json"

    depth_dir.mkdir(parents=True, exist_ok=True)
    json_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Using device: {device}")
    print(f"Loading MiDaS model: {model_type}")

    midas = torch.hub.load("intel-isl/MiDaS", model_type)
    midas.to(device)
    midas.eval()

    transforms = torch.hub.load("intel-isl/MiDaS", "transforms")

    if model_type in ["DPT_Large", "DPT_Hybrid"]:
        transform = transforms.dpt_transform
    else:
        transform = transforms.small_transform

    image_paths = get_image_paths(input_path)

    summary = {}

    for img_path in image_paths:
        image_bgr = cv2.imread(str(img_path))

        if image_bgr is None:
            print(f"Skipping unreadable image: {img_path}")
            continue

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

        depth = prediction.cpu().numpy()
        depth_img = normalize_depth(depth)

        color_depth = cv2.applyColorMap(depth_img, cv2.COLORMAP_INFERNO)

        depth_path = depth_dir / f"{img_path.stem}_depth.jpg"
        raw_depth_path = depth_dir / f"{img_path.stem}_depth_raw.npy"

        cv2.imwrite(str(depth_path), color_depth)
        np.save(str(raw_depth_path), depth)

        depth_stats = {
            "image": img_path.name,
            "model": model_type,
            "depth_min": float(depth.min()),
            "depth_max": float(depth.max()),
            "depth_mean": float(depth.mean()),
            "depth_std": float(depth.std()),
            "depth_map_path": str(depth_path),
            "raw_depth_path": str(raw_depth_path),
            "note": "MiDaS depth is relative, not metric distance.",
        }

        json_path = json_dir / f"{img_path.stem}_depth_summary.json"

        with open(json_path, "w") as f:
            json.dump(depth_stats, f, indent=2)

        summary[img_path.name] = depth_stats

        print(f"Saved depth map: {depth_path}")

    with open(output_dir / "depth_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Saved all depth outputs to: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Image or image folder")
    parser.add_argument("--output", default="outputs/depth", help="Output directory")
    parser.add_argument(
        "--model",
        default="DPT_Hybrid",
        choices=["DPT_Large", "DPT_Hybrid", "MiDaS_small"],
        help="MiDaS model type",
    )

    args = parser.parse_args()

    run_depth(
        input_path=args.input,
        output_dir=args.output,
        model_type=args.model,
    )

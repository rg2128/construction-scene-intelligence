from pathlib import Path
import argparse
import cv2
import numpy as np


CLASS_NAMES = {
    0: "helmet",
    1: "vest",
    2: "without_helmet",
    3: "without_vest",
}


def draw_text_with_background(image, text, x, y):
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8
    thickness = 2

    text_w, text_h = cv2.getTextSize(text, font, font_scale, thickness)[0]
    y = max(y, text_h + 8)

    cv2.rectangle(
        image,
        (x, y - text_h - 8),
        (x + text_w + 8, y + 4),
        (255, 0, 255),
        -1,
    )

    cv2.putText(
        image,
        text,
        (x + 4, y - 4),
        font,
        font_scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )


def draw_labels(image_path, label_path, output_path):
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    img_h, img_w = image.shape[:2]

    with open(label_path, "r") as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]

    print(f"Image size: {img_w} x {img_h}")
    print(f"Labels found: {len(lines)}")

    for line in lines:
        parts = line.split()
        cls_id = int(float(parts[0]))
        coords = list(map(float, parts[1:]))

        label = CLASS_NAMES.get(cls_id, f"class_{cls_id}")

        if len(coords) == 4:
            xc, yc, w, h = coords
            x1 = int((xc - w / 2) * img_w)
            y1 = int((yc - h / 2) * img_h)
            x2 = int((xc + w / 2) * img_w)
            y2 = int((yc + h / 2) * img_h)

            cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 255), 4)
            draw_text_with_background(image, label, x1, y1)

        elif len(coords) >= 6 and len(coords) % 2 == 0:
            points = []
            for i in range(0, len(coords), 2):
                x = int(coords[i] * img_w)
                y = int(coords[i + 1] * img_h)
                points.append([x, y])

            pts = np.array(points, dtype=np.int32)

            cv2.polylines(
                image,
                [pts],
                isClosed=True,
                color=(255, 0, 255),
                thickness=4,
            )

            x_text = min(p[0] for p in points)
            y_text = min(p[1] for p in points)
            draw_text_with_background(image, label, x_text, y_text)

            print(f"{label}: polygon with {len(points)} points")

        else:
            print(f"Skipping unsupported label line: {line}")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), image)
    print(f"Saved labeled image to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--output", default="outputs/label_visualizations/labels.jpg")
    args = parser.parse_args()

    draw_labels(
        image_path=Path(args.image),
        label_path=Path(args.label),
        output_path=Path(args.output),
    )
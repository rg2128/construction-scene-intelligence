from ultralytics import YOLO


def main():
    model = YOLO("yolo11n.pt")

    model.train(
        data="data/construction_person/data.yaml",
        epochs=50,
        imgsz=640,
        batch=8,
        project="outputs/training",
        name="yolo11n_construction_person",
    )


if __name__ == "__main__":
    main()

from ultralytics import YOLO

def main():
    model = YOLO("yolo11n.pt")

    model.train(
        data="C:/anpr15k/License Plate Detection/data.yaml",
        epochs=100,
        imgsz=640,
        batch=16,
        patience=15,
        device=0,
        project="runs/detect",
        name="anpr_plate_model_15k"
    )

if __name__ == "__main__":
    main()
from ultralytics import YOLO

def main():
    model = YOLO(r"runs/detect/runs/detect/anpr_plate_model_15k/weights/last.pt")
    model.train(resume=True, batch=8)

if __name__ == "__main__":
    main()
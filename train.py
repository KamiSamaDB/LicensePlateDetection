from ultralytics import YOLO

def main():
    model = YOLO('yolov8n.pt')
    
    results = model.train(
        data='dataset/dataset.yaml',
        epochs=50,
        imgsz=640,
        batch=8,
        name='license_plate_detector',
        patience=10,
        save=True,
        plots=True,
        workers=0
    )
    
    print("\nTraining complete!")
    print(f"Best model saved to: runs/detect/license_plate_detector/weights/best.pt")

if __name__ == "__main__":
    main()

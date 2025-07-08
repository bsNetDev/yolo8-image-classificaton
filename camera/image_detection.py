# camera/image_detection.py
import torch
import cv2
import numpy as np
from ultralytics import YOLO

def run_detection():
    # Check CUDA availability
    if not torch.cuda.is_available():
        print("CUDA is not available. Running on CPU.")
        device = torch.device("cpu")
    else:
        device = torch.device("cuda")
        print("CUDA Available:", torch.cuda.is_available())
        print("Device Name:", torch.cuda.get_device_name(0))

    # Load YOLOv8 model
    model = YOLO("camera/model/yolov8n.pt", task='detect').to(device)
    print(f"Model is using: {model.device}")

    # Start video capture
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open webcam")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Run detection
        results = model(rgb_frame, device=device, verbose=False)[0]

        # Draw bounding boxes
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = box.conf[0].item()
            cls = int(box.cls[0])
            label = model.names[cls]

            if conf > 0.5:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cv2.imshow("Webcam - Object Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

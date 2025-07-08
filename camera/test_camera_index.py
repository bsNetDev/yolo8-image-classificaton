import cv2

for index in range(5):
    print(f"Testing camera index {index}")
    cap = cv2.VideoCapture(index)
    if cap.isOpened():
        print(f"Camera index {index} opened successfully.")
        cap.release()
    else:
        print(f"Camera index {index} failed.")

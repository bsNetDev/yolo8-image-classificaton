import cv2

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Camera not detected.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break
    cv2.imshow("Camera", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

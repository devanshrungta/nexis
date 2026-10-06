import cv2
import time
from ultralytics import YOLO

# -----------------------------
# YOLO model
# -----------------------------
model = YOLO("best.pt")
# model = YOLO("yolo26m.engine")

# -----------------------------
# Camera
# -----------------------------
cap = cv2.VideoCapture(0, cv2.CAP_V4L2)

if not cap.isOpened():
    print("Error: Could not open the ELP camera.")
    exit()

# Camera settings
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FPS, 60)
cap.set(cv2.CAP_PROP_FOURCC,
        cv2.VideoWriter_fourcc(*"MJPG"))

time.sleep(0.5)

print("Camera started")
print("Press Q to quit")

# -----------------------------
# FPS
# -----------------------------
prev_time = time.time()

while True:

    ret, frame = cap.read()

    if not ret:
        print("Failed to read frame")
        continue

    # -----------------------------
    # YOLO detection
    # -----------------------------
    results = model(
        frame,
        imgsz=640,
        conf=0.5,
        verbose=False
    )

    # Draw detections
    annotated_frame = results[0].plot()

    # -----------------------------
    # FPS
    # -----------------------------
    current_time = time.time()
    fps = 1.0 / (current_time - prev_time)
    prev_time = current_time

    cv2.putText(
        annotated_frame,
        f"FPS: {fps:.1f}",
        (30, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    # -----------------------------
    # Display
    # -----------------------------
    cv2.imshow("YOLO + ELP Camera", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()


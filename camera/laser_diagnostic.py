import cv2
import numpy as np
import time

# bgr 255 255 255
# hsv 90 2 255

cap = cv2.VideoCapture(0, cv2.CAP_V4L2)

if not cap.isOpened():
    raise RuntimeError("Could not open camera")

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FPS, 60)
cap.set(
    cv2.CAP_PROP_FOURCC,
    cv2.VideoWriter_fourcc(*"MJPG")
)

time.sleep(0.5)

print("Camera started")
print("Point the laser at a target")
print("Press Q to quit")


while True:

    ret, frame = cap.read()

    if not ret:
        continue

    # Show the raw camera image
    output = frame.copy()

    cv2.putText(
        output,
        "Point laser at target",
        (30, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.imshow("Camera", output)

    # -----------------------------------------
    # Find the brightest pixel
    # -----------------------------------------

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(gray)

    # Mark brightest pixel
    cv2.circle(
        output,
        max_loc,
        15,
        (0, 0, 255),
        2
    )

    cv2.putText(
        output,
        f"Brightest: {max_val:.0f}  Pos: {max_loc}",
        (30, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2
    )

    # -----------------------------------------
    # Print RGB/BGR/HSV of brightest pixel
    # -----------------------------------------

    x, y = max_loc

    b, g, r = frame[y, x]

    hsv = cv2.cvtColor(
        np.uint8([[[b, g, r]]]),
        cv2.COLOR_BGR2HSV
    )[0, 0]

    cv2.putText(
        output,
        f"BGR: {b},{g},{r}",
        (30, 120),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    cv2.putText(
        output,
        f"HSV: {hsv[0]},{hsv[1]},{hsv[2]}",
        (30, 155),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    cv2.imshow("Camera", output)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


cap.release()
cv2.destroyAllWindows()

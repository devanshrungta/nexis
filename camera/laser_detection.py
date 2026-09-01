import cv2
import numpy as np
import time


# ============================================================
# Camera
# ============================================================

cap = cv2.VideoCapture(0, cv2.CAP_V4L2)

if not cap.isOpened():
    raise RuntimeError("Could not open ELP camera")

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FPS, 60)
cap.set(
    cv2.CAP_PROP_FOURCC,
    cv2.VideoWriter_fourcc(*"MJPG")
)

time.sleep(0.5)

print("Camera started")
print("Press Q to quit")


# ============================================================
# Main loop
# ============================================================

while True:

    ret, frame = cap.read()

    if not ret:
        continue

    # --------------------------------------------------------
    # Convert to grayscale
    # --------------------------------------------------------

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # --------------------------------------------------------
    # Find very bright pixels
    # --------------------------------------------------------

    mask = cv2.inRange(
        gray,
        245,       # minimum brightness
        255
    )

    # --------------------------------------------------------
    # Remove tiny noise
    # --------------------------------------------------------

    kernel = np.ones((3, 3), np.uint8)

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel
    )

    # --------------------------------------------------------
    # Find contours
    # --------------------------------------------------------

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    output = frame.copy()

    best = None
    best_area = 0

    for contour in contours:

        area = cv2.contourArea(contour)

        # Ignore very small regions
        if area < 2:
            continue

        if area > best_area:
            best = contour
            best_area = area

    # --------------------------------------------------------
    # Laser detection
    # --------------------------------------------------------

    if best is not None:

        M = cv2.moments(best)

        if M["m00"] != 0:

            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])

            # Draw detected blob
            cv2.drawContours(
                output,
                [best],
                -1,
                (0, 255, 0),
                2
            )

            # Draw laser center
            cv2.circle(
                output,
                (cx, cy),
                8,
                (0, 0, 255),
                -1
            )

            # Crosshair
            cv2.drawMarker(
                output,
                (cx, cy),
                (0, 0, 255),
                cv2.MARKER_CROSS,
                30,
                2
            )

            # Coordinates
            cv2.putText(
                output,
                f"Laser: ({cx}, {cy})",
                (cx + 15, cy),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2
            )

    else:

        cv2.putText(
            output,
            "LASER NOT DETECTED",
            (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2
        )

    # --------------------------------------------------------
    # Display
    # --------------------------------------------------------

    cv2.imshow(
        "Camera + Laser",
        output
    )

    cv2.imshow(
        "Brightness Mask",
        mask
    )

    # --------------------------------------------------------
    # Quit
    # --------------------------------------------------------

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# ============================================================
# Cleanup
# ============================================================

cap.release()
cv2.destroyAllWindows()

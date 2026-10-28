#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="/home/sunoj-orin/devansh/laser_position_calibrated_control_package"
PYTHON="/home/sunoj-orin/venvs/weed-yolo/bin/python"
SCRIPT="${PROJECT_DIR}/laser_servo_tracker_yolo_stable_auto_visit.py"

if [[ ! -x "$PYTHON" ]]; then
  echo "ERROR: expected YOLO virtual-environment Python at: $PYTHON" >&2
  exit 1
fi
if [[ ! -f "$SCRIPT" ]]; then
  echo "ERROR: controller not found at: $SCRIPT" >&2
  echo "Copy laser_servo_tracker_yolo_stable_auto_visit.py into $PROJECT_DIR first." >&2
  exit 1
fi

cd "$PROJECT_DIR"
if [[ "$EUID" -ne 0 ]]; then
  exec sudo "$PYTHON" "$SCRIPT" "$@"
else
  exec "$PYTHON" "$SCRIPT" "$@"
fi

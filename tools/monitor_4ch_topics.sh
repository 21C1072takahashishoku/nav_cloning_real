#!/usr/bin/env bash
set -euo pipefail

echo "[1] Camera topics"
rostopic hz /camera_center/usb_cam/image_raw -w 3 || true
rostopic hz /camera_left/usb_cam/image_raw -w 3 || true
rostopic hz /camera_right/usb_cam/image_raw -w 3 || true

echo "[2] Mask topics"
rostopic hz /segmentation/ground_mask_center -w 3 || true
rostopic hz /segmentation/ground_mask_left -w 3 || true
rostopic hz /segmentation/ground_mask_right -w 3 || true

echo "[3] Mask encoding"
rostopic echo -n 1 /segmentation/ground_mask_center/encoding || true

echo "[4] GPU"
nvidia-smi || true

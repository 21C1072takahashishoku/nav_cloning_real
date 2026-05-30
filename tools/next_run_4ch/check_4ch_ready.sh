#!/usr/bin/env bash

echo "============================================================"
echo "[1] ROS package check"
echo "============================================================"
source /opt/ros/noetic/setup.bash
source ~/shoku_ws/devel/setup.bash
export ROS_PACKAGE_PATH=$ROS_PACKAGE_PATH:$HOME/shoku_ws/src

rospack find nav_cloning || exit 1

echo "============================================================"
echo "[2] Required files"
echo "============================================================"
FILES=(
  "$HOME/shoku_ws/src/nav_cloning/scripts/start_mmseg.sh"
  "$HOME/shoku_ws/src/nav_cloning/scripts/segmentation_colorserver_real.py"
  "$HOME/shoku_ws/src/nav_cloning/scripts/nav_cloning_4ch_node_pytorch_online_mask.py"
  "$HOME/shoku_ws/src/nav_cloning/experiments/experiment_4ch_mask2former_offline_collect.sh"
  "$HOME/shoku_ws/src/nav_cloning/launch/nav_cloning_4ch_all.launch"
)

for f in "${FILES[@]}"; do
  if [ -e "$f" ]; then
    echo "OK: $f"
  else
    echo "NG: missing $f"
  fi
done

echo "============================================================"
echo "[3] Executable permission"
echo "============================================================"
ls -l ~/shoku_ws/src/nav_cloning/scripts/start_mmseg.sh
ls -l ~/shoku_ws/src/nav_cloning/scripts/nav_cloning_4ch_node_pytorch_online_mask.py
ls -l ~/shoku_ws/src/nav_cloning/experiments/experiment_4ch_mask2former_offline_collect.sh

echo "============================================================"
echo "[4] /home/orne_beta remains"
echo "============================================================"
grep -R "/home/orne_beta" -n ~/shoku_ws/src/nav_cloning || echo "OK: no /home/orne_beta"

echo "============================================================"
echo "[5] cv_bridge remains in segmentation_colorserver"
echo "============================================================"
grep -n "imgmsg_to_cv2\|cv2_to_imgmsg" \
  ~/shoku_ws/src/nav_cloning/scripts/segmentation_colorserver_real.py \
  || echo "OK: no cv_bridge conversion remains"

echo "============================================================"
echo "[6] 4ch debug log check"
echo "============================================================"
grep -n "4ch save check\|4ch save DO" \
  ~/shoku_ws/src/nav_cloning/scripts/nav_cloning_4ch_node_pytorch_online_mask.py \
  || echo "NG: save debug logs not found"

echo "============================================================"
echo "[7] Mask2Former conda env check"
echo "============================================================"
/home/orne_beta/miniconda3/envs/mask2former_mmseg/bin/python - <<'PY'
import torch
import mmseg
import mmdet
from mmseg.apis import init_model, inference_model
print("torch:", torch.__version__)
print("cuda:", torch.cuda.is_available())
print("mmseg:", mmseg.__version__)
print("mmdet:", mmdet.__version__)
print("Mask2Former env OK")
PY

echo "============================================================"
echo "CHECK DONE"
echo "============================================================"

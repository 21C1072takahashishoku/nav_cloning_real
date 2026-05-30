#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# 4ch Mask2Former offline data collection for real robot
# ============================================================
# Important:
# - Run this from a normal ROS terminal.
# - Do NOT `conda activate mask2former_mmseg` in this terminal.
# - Mask2Former is executed inside start_mmseg.sh using conda python directly.

NODE_SCRIPT="nav_cloning_4ch_node_pytorch_online_mask.py"

RUNS="${RUNS:-1}"
DATASET_ID="${1:-real_4ch_short_test}"
DATASET_STRIDE="${DATASET_STRIDE:-1}"
DATASET_ACTION_OFFSET="${DATASET_ACTION_OFFSET:-0.2}"
SAVE_OFFLINE_DATASET="${SAVE_OFFLINE_DATASET:-true}"
SAVE_RAW_IMAGES="${SAVE_RAW_IMAGES:-true}"
TRAIN_STEPS="${TRAIN_STEPS:-5000}"
TEST_STEPS="${TEST_STEPS:-2000}"
AUTO_STOP_TEST="${AUTO_STOP_TEST:-true}"
AUTO_SAVE_MODEL="${AUTO_SAVE_MODEL:-true}"
AUTO_LOAD_MODEL="${AUTO_LOAD_MODEL:-true}"
WAYPOINT_CONFIG="$(rospack find waypoint_server)/config/waypoint_server_tsudanuma_2-3.yaml"

# Safety defaults.
CMD_VEL_LINEAR_FIXED="${CMD_VEL_LINEAR_FIXED:-0.2}"
CMD_VEL_LINEAR_SOURCE="${CMD_VEL_LINEAR_SOURCE:-fixed}"
CMD_VEL_MODE="${CMD_VEL_MODE:-mixed}"
MASK_MAX_AGE_SEC="${MASK_MAX_AGE_SEC:-1.5}"
REQUIRE_FRESH_MASK="${REQUIRE_FRESH_MASK:-true}"

# Same base settings as the 3ch offline collection script.
MAP_FILE="${MAP_FILE:-cit_3f_map}"
USE_WAYPOINT_NAV="${USE_WAYPOINT_NAV:-false}"
DIST_ERR="${DIST_ERR:-1.0}"
INITIAL_POSE_X="${INITIAL_POSE_X:--9.44}"
INITIAL_POSE_Y="${INITIAL_POSE_Y:-28.83}"
INITIAL_POSE_A="${INITIAL_POSE_A:-3.14}"
USE_INITPOSE="${USE_INITPOSE:-false}"

printf '%s\n' "============================================================"
printf '%s\n' "Run 4ch Mask2Former offline collection"
printf '%s\n' "============================================================"
printf 'NODE_SCRIPT=%s\n' "$NODE_SCRIPT"
printf 'DATASET_ID=%s\n' "$DATASET_ID"
printf 'TRAIN_STEPS=%s TEST_STEPS=%s\n' "$TRAIN_STEPS" "$TEST_STEPS"
printf 'DATASET_STRIDE=%s\n' "$DATASET_STRIDE"
printf 'MAP_FILE=%s\n' "$MAP_FILE"
printf 'WAYPOINT_CONFIG=%s\n' "$WAYPOINT_CONFIG"
printf 'CMD_VEL_LINEAR_FIXED=%s\n' "$CMD_VEL_LINEAR_FIXED"
printf 'MASK_MAX_AGE_SEC=%s\n' "$MASK_MAX_AGE_SEC"
printf '%s\n' "============================================================"

for _ in $(seq "${RUNS}"); do
  roslaunch nav_cloning nav_cloning_4ch_all.launch \
    script:="${NODE_SCRIPT}" \
    mode:=change_dataset_balance \
    map_file:="${MAP_FILE}" \
    use_waypoint_nav:="${USE_WAYPOINT_NAV}" \
    waypoint_server_config:="${WAYPOINT_CONFIG}" \
    dist_err:="${DIST_ERR}" \
    initial_pose_x:="${INITIAL_POSE_X}" \
    initial_pose_y:="${INITIAL_POSE_Y}" \
    initial_pose_a:="${INITIAL_POSE_A}" \
    use_initpose:="${USE_INITPOSE}" \
    save_offline_dataset:="${SAVE_OFFLINE_DATASET}" \
    save_raw_images:="${SAVE_RAW_IMAGES}" \
    dataset_id:="${DATASET_ID}" \
    dataset_stride:="${DATASET_STRIDE}" \
    dataset_action_offset:="${DATASET_ACTION_OFFSET}" \
    train_steps:="${TRAIN_STEPS}" \
    test_steps:="${TEST_STEPS}" \
    auto_stop_test:="${AUTO_STOP_TEST}" \
    auto_save_model:="${AUTO_SAVE_MODEL}" \
    auto_load_model:="${AUTO_LOAD_MODEL}" \
    mask_max_age_sec:="${MASK_MAX_AGE_SEC}" \
    require_fresh_mask:="${REQUIRE_FRESH_MASK}" \
    cmd_vel_linear_fixed:="${CMD_VEL_LINEAR_FIXED}" \
    cmd_vel_linear_source:="${CMD_VEL_LINEAR_SOURCE}" \
    cmd_vel_mode:="${CMD_VEL_MODE}"

  sleep 10
done

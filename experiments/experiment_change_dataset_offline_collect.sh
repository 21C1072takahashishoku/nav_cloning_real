#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# 使用プログラム
# ============================================================
# リサイズ前の元画像 raw_img も保存する版
NODE_SCRIPT="pytorch/nav_cloning_fullpic_node_pytorch_offline_collect.py"

# ============================================================
# 実験ID・保存設定
# ============================================================
RUNS="${RUNS:-1}"
DATASET_ID="${1:-$(date +%Y%m%d_%H%M%S)}"

# 1なら全step保存、5なら5stepに1回保存
DATASET_STRIDE="${DATASET_STRIDE:-1}"

# left/right仮想ラベル用オフセット
DATASET_ACTION_OFFSET="${DATASET_ACTION_OFFSET:-0.2}"

# 既存の48x64 npy保存
SAVE_OFFLINE_DATASET="${SAVE_OFFLINE_DATASET:-true}"

# 今回追加：リサイズ前480x640 png保存
SAVE_RAW_IMAGES="${SAVE_RAW_IMAGES:-true}"

# ============================================================
# 学習・テストstep設定
# ============================================================
TRAIN_STEPS="${TRAIN_STEPS:-10000}"
TEST_STEPS="${TEST_STEPS:-3000}"

# 短時間確認用にする場合は、実行時に環境変数で上書きする
# 例：
# TRAIN_STEPS=300 TEST_STEPS=300 DATASET_STRIDE=5 ./run_collect_raw_offline.sh rawtest_001

AUTO_STOP_TEST="${AUTO_STOP_TEST:-true}"
AUTO_SAVE_MODEL="${AUTO_SAVE_MODEL:-true}"
AUTO_LOAD_MODEL="${AUTO_LOAD_MODEL:-true}"

# ============================================================
# waypoint設定
# ============================================================
WAYPOINT_CONFIG="$(rospack find waypoint_server)/config/waypoint_server_tsudanuma_2-3.yaml"

# ============================================================
# 実行前確認
# ============================================================
echo "============================================================"
echo "Run nav_cloning offline collection"
echo "============================================================"
echo "NODE_SCRIPT=${NODE_SCRIPT}"
echo "DATASET_ID=${DATASET_ID}"
echo "RUNS=${RUNS}"
echo "TRAIN_STEPS=${TRAIN_STEPS}"
echo "TEST_STEPS=${TEST_STEPS}"
echo "DATASET_STRIDE=${DATASET_STRIDE}"
echo "DATASET_ACTION_OFFSET=${DATASET_ACTION_OFFSET}"
echo "SAVE_OFFLINE_DATASET=${SAVE_OFFLINE_DATASET}"
echo "SAVE_RAW_IMAGES=${SAVE_RAW_IMAGES}"
echo "AUTO_STOP_TEST=${AUTO_STOP_TEST}"
echo "AUTO_SAVE_MODEL=${AUTO_SAVE_MODEL}"
echo "AUTO_LOAD_MODEL=${AUTO_LOAD_MODEL}"
echo "WAYPOINT_CONFIG=${WAYPOINT_CONFIG}"
echo "============================================================"

for _ in $(seq "${RUNS}")
do
  roslaunch nav_cloning nav_cloning_all.launch \
    script:="${NODE_SCRIPT}" \
    mode:=change_dataset_balance \
    map_file:=cit_3f_map \
    use_waypoint_nav:=false \
    waypoint_server_config:="${WAYPOINT_CONFIG}" \
    dist_err:=1.0 \
    initial_pose_x:=-9.44 \
    initial_pose_y:=28.83 \
    initial_pose_a:=3.14 \
    robot_x:=-9.3 \
    robot_y:=28.6 \
    robot_Y:=3.14 \
    use_initpose:=false \
    robot_name:=gamma \
    use_dynamic_inflation:=false \
    inflation_mode:=identity \
    inflation_index_large:=0 \
    inflation_index_small:=10 \
    inflation_large:=0.6 \
    inflation_small:=0.3 \
    save_offline_dataset:="${SAVE_OFFLINE_DATASET}" \
    save_raw_images:="${SAVE_RAW_IMAGES}" \
    dataset_id:="${DATASET_ID}" \
    dataset_stride:="${DATASET_STRIDE}" \
    dataset_action_offset:="${DATASET_ACTION_OFFSET}" \
    train_steps:="${TRAIN_STEPS}" \
    test_steps:="${TEST_STEPS}" \
    auto_stop_test:="${AUTO_STOP_TEST}" \
    auto_save_model:="${AUTO_SAVE_MODEL}" \
    auto_load_model:="${AUTO_LOAD_MODEL}"

  sleep 10
done

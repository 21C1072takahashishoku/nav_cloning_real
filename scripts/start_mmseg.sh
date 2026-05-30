#!/usr/bin/env bash
set -e

SCRIPT_PATH="$1"
shift

# ROS environment for rospy/message types.
source /opt/ros/noetic/setup.bash
export ROS_PACKAGE_PATH="${ROS_PACKAGE_PATH}:/home/orne_beta/shoku_ws/src"

# Use conda python directly. Do not rely on interactive `conda activate` from roslaunch.
CONDA_PYTHON="/home/orne_beta/miniconda3/envs/mask2former_mmseg/bin/python"

if [ ! -x "$CONDA_PYTHON" ]; then
  echo "ERROR: conda python not found: $CONDA_PYTHON"
  echo "Check: /home/orne_beta/miniconda3/bin/conda env list"
  exit 1
fi

exec "$CONDA_PYTHON" "$SCRIPT_PATH" "$@"

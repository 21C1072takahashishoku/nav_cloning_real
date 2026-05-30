Next run package for real robot 4ch Mask2Former test.

Files:
1. start_mmseg.sh
   Copy to: ~/shoku_ws/src/nav_cloning/scripts/start_mmseg.sh
   Role: launch Mask2Former with conda python without activating conda in ROS terminal.

2. experiment_4ch_mask2former_offline_collect.sh
   Copy to: ~/shoku_ws/src/nav_cloning/experiments/
   Role: 4ch version of the 3ch offline collection script.

3. patch_segmentation_no_cvbridge.py
   Run once if segmentation_colorserver_real.py still uses cv_bridge.
   Command: python3 patch_segmentation_no_cvbridge.py

4. patch_nav4ch_import_debug.py
   Run once if nav_cloning_4ch_node_pytorch_online_mask.py is in scripts/ and cannot import nav_cloning_pytorch.
   Command: python3 patch_nav4ch_import_debug.py

5. nav_cloning_4ch_all_next.launch
   Reference launch file with safer defaults. Review before replacing existing launch.

Minimum next-run command:
cd ~/shoku_ws/src/nav_cloning/experiments
source /opt/ros/noetic/setup.bash
source ~/shoku_ws/devel/setup.bash
TRAIN_STEPS=300 TEST_STEPS=100 ./experiment_4ch_mask2former_offline_collect.sh real_4ch_short_test

for i in `seq 1`
do
  roslaunch nav_cloning nav_cloning_all.launch script:=nav_cloning_node_pytorch.py mode:=change_dataset_balance  map_file:=cit_3f_map use_waypoint_nav:=false waypoint_server_config:=$(rospack find waypoint_server)/config/waypoint_server_tsudanuma_2-3.yaml dist_err:=1.0 initial_pose_x:=-9.44 initial_pose_y:=28.83 initial_pose_a:=3.14 robot_x:=-9.3 robot_y:=28.6 robot_Y:=3.14 use_initpose:="false" robot_name:="gamma" use_dynamic_inflation:=false inflation_mode:=identity inflation_index_large:=0 inflation_index_small:=10 inflation_large:=0.6 inflation_small:=0.3
  sleep 10
done

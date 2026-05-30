for i in `seq 1`
do
  roslaunch nav_cloning nav_cloning_all.launch script:=nav_cloning_node_pytorch.py mode:=use_dl_output  map_file:=cit_3f_map use_waypoint_nav:=false waypoint_server_config:=$(rospack find waypoint_server)/config/waypoint_server_tsudanuma_2-3.yaml dist_err:=1.0 initial_pose_x:=-9.44 initial_pose_y:=28.83 initial_pose_a:=3.14 robot_x:=-9.3 robot_y:=28.6 robot_Y:=3.14 use_initpose:="true" robot_name:="gamma"
  sleep 10
done

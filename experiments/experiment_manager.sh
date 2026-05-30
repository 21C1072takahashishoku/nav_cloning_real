for i in `seq 1`
do
  roslaunch nav_cloning nav_cloning_all.launch script:=nav_cloning_node_pytorch_test.py map_file:=/imai/maps/tsudanuma/map_test/map_test dist_err:=1.0 initial_pose_x:=-9.44 initial_pose_y:=28.83 initial_pose_a:=3.14 robot_x:=-9.3 robot_y:=28.6 robot_Y:=3.14 use_initpose:="false" robot_name:="gamma"
  sleep 10
done

for i in `seq 1`
do
  roslaunch nav_cloning nav_cloning_all.launch script:=nav_cloning_node_pytorch_EBV_day.py mode:=use_dl_output  map_file:=cit_3_map robot_name:="gamma"
  sleep 10
done

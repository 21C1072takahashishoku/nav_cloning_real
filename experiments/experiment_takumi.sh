for i in `seq 1`
do
  roslaunch nav_cloning nav_cloning_all.launch script:=nav_cloning_node_pytorch_plot1_willow_50_corner.py script2:=nav_cloning_node_pytorch_plot2_willow_50_corner.py mode:=change_dataset_balance  map_file:=cit_3_map robot_name:="gamma"
  sleep 10
done

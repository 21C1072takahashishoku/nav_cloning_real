for i in `seq 1`
do
  roslaunch nav_cloning nav_cloning_all.launch script:=nav_cloning_node_pytorch_plot1_conventional.py script2:=nav_cloning_node_pytorch_plot2_conventional.py mode:=change_dataset_balance  map_file:=cit_3_map robot_name:="gamma"
  sleep 10
done

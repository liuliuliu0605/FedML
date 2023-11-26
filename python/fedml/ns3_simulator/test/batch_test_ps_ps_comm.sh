#!/usr/bin/env bash

pattern=pfl
underlay=geantdistance
ps_num=9
model_size=2401488
#topo_name=complete
#access_link_capacity=1000000
#core_link_capacity=1000000000

topo_name_list=(complete 2d_torus ring star)
access_link_capacity_list=(1000000 10000000 100000000 1000000000)
core_link_capacity_list=(1000000 10000000 100000000 1000000000)
#access_link_capacity_list=(1000000)
#core_link_capacity_list=(1000000)
for topo_name in ${topo_name_list[@]};
do
  for access_link_capacity in ${access_link_capacity_list[@]};
  do
    for core_link_capacity in ${core_link_capacity_list[@]};
    do
      mpirun -np 9 python test_ps_ps_comm.py --pattern $pattern \
      --underlay $underlay --ps_num $ps_num \
      --topo_name $topo_name --model_size=$model_size \
      --access_link_capacity $access_link_capacity \
      --core_link_capacity $core_link_capacity
    done
  done
done



CONFIG_PATH=config/cifar10_resnet56/fedml_config.yaml
RANDOM_SEED=10
GROUP_NUM=9
COMM_ROUND=500
TIME_BUDGET=0
GROUP_METHOD=hetero
GROUP_ALPHA=1.0
TOPO_NAME=complete
GROUP_COMM_PATTERN=decentralized
GROUP_COMM_ROUND=1
ACCESS_LINK_CAPACITY=1e7
CORE_LINK_CAPACITY=1e9
LAN_CAPACITY=1e11
LOCAL_UPDATE_TIME=0.07576
WORKER_NUM=$(($GROUP_NUM+1))
GPU_UTIL_PARSE=localhost:3,2,3,2

random_seed_list=()
time_budget_list=()
access_link_capacity_list=()
core_link_capacity=()
group_alpha_list=(10.0 1.0 0.1)
group_method_list=(random hetero)
group_comm_pattern_list=(decentralized centralized async-centralized)


group_method_list=(hetero)
group_comm_pattern_list=(decentralized)


for GROUP_METHOD in ${group_method_list[@]};
do

  for GROUP_ALPHA in ${group_alpha_list[@]};
  do

    for GROUP_COMM_PATTERN in ${group_comm_pattern_list[@]};
    do

      if [ "${GROUP_COMM_PATTERN}" = "decentralized" ]; then
#          group_comm_round_list=(0 1)
#          topo_name_list=(complete 2d_torus ring star)
          group_comm_round_list=(1)
          topo_name_list=(ring)
        elif [ "${GROUP_COMM_PATTERN}" = "centralized" ];then
          group_comm_round_list=(4)
          topo_name_list=(complete)
        else
          group_comm_round_list=(4)
          topo_name_list=(complete)
      fi

      for TOPO_NAME in ${topo_name_list[@]};
      do

        for GROUP_COMM_ROUND in ${group_comm_round_list[@]};
        do

            log_file="group_method=$GROUP_METHOD-group_alpha=$GROUP_ALPHA-topo=$TOPO_NAME-group_comm_pattern=$GROUP_COMM_PATTERN-group_comm_round=$GROUP_COMM_ROUND.log"
            echo $log_file
            nohup mpirun -np $WORKER_NUM \
                -hostfile mpi_host_file \
                python main.py --cf $CONFIG_PATH --random_seed $RANDOM_SEED\
                --worker_num $WORKER_NUM --gpu_util_parse $GPU_UTIL_PARSE\
                --group_num $GROUP_NUM --group_method $GROUP_METHOD --group_alpha $GROUP_ALPHA\
                --topo_name $TOPO_NAME --group_comm_pattern $GROUP_COMM_PATTERN\
                --comm_round $COMM_ROUND --time_budget $TIME_BUDGET\
                --group_comm_round $GROUP_COMM_ROUND\
                --access_link_capacity $ACCESS_LINK_CAPACITY --core_link_capacity $CORE_LINK_CAPACITY\
                --lan_capacity $LAN_CAPACITY --local_update_time $LOCAL_UPDATE_TIME\
                > batch_log/$log_file 2>&1 \
                & echo $! >> batch_log/process.pid
            #    --enable_ns3
            #    --enable_dynamic_topo
            sleep 10

        done

      done

    done

  done

done


#mpirun -np $WORKER_NUM \
#    -hostfile mpi_host_file \
#    python main.py --cf $CONFIG_PATH --random_seed $RANDOM_SEED\
#    --worker_num $WORKER_NUM --gpu_util_parse $GPU_UTIL_PARSE\
#    --group_num $GROUP_NUM --group_method $GROUP_METHOD --group_alpha $GROUP_ALPHA\
#    --topo_name $TOPO_NAME --group_comm_pattern $GROUP_COMM_PATTERN\
#    --comm_round $COMM_ROUND --time_budget $TIME_BUDGET\
#    --group_comm_round $GROUP_COMM_ROUND\
#    --access_link_capacity $ACCESS_LINK_CAPACITY --core_link_capacity $CORE_LINK_CAPACITY\
#    --lan_capacity $LAN_CAPACITY --local_update_time $LOCAL_UPDATE_TIME\
#    --enable_ns3
##    --enable_dynamic_topo
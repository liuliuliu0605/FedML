CONFIG_PATH=config/cifar100_mobilenet_v3/fedml_config.yaml
RANDOM_SEED=0
GROUP_NUM=9
BASE_COMM_ROUND=0
TIME_BUDGET=20000
PARTITION_ALPHA=0.5
GROUP_METHOD=hetero
GROUP_ALPHA=10
TOPO_NAME=complete
GROUP_COMM_PATTERN=decentralized
GROUP_COMM_ROUND=1
ACCESS_LINK_CAPACITY=1e7
CORE_LINK_CAPACITY=1e10
LAN_CAPACITY=1e11
LOCAL_UPDATE_TIME=0.9654
WORKER_NUM=$(($GROUP_NUM+1))
#GPU_UTIL_PARSE_LIST=(localhost:7,7,6,6 localhost:6,6,7,7)
GPU_UTIL_PARSE_LIST=(localhost:3,2,2,3 localhost:2,3,3,2)
#GPU_UTIL_PARSE_LIST=(localhost:4,1,4,1 localhost:4,1,4,1)
lan_capacity_ratio_list=0.0001,0.001,0.001,0.001,0.001,0.01,0.01,0.01,0.01
random_seed_list=()
time_budget_list=()
access_link_capacity_list=()
core_link_capacity=()
group_alpha_list=(0.05 0.1 5)
partition_alpha_list=(50 25 0.5)
alpha_index_list=(0 1 2)
group_method_list=(random hetero)
group_comm_pattern_list=(decentralized centralized async-centralized)

########################################################################
group_method_list=(hetero)
group_comm_pattern_list=(decentralized centralized async-centralized)
alpha_index_list=(0)

experiment_num=0
for GROUP_METHOD in ${group_method_list[@]};
do

  for index in ${alpha_index_list[@]};
  do

    PARTITION_ALPHA=${partition_alpha_list[index]}
    GROUP_ALPHA=${group_alpha_list[index]}
    for GROUP_COMM_PATTERN in ${group_comm_pattern_list[@]};
    do

      if [ "${GROUP_COMM_PATTERN}" = "decentralized" ]; then
#          group_comm_round_list=(1 10 3 30 5 50 7 70 100 200)
#          topo_name_list=(complete 2d_torus ring star)
#          group_comm_round_list=(0 1 10 3 30)
#          group_comm_round_list=(5 50 7 70 100)
#          topo_name_list=(complete)
          group_comm_round_list=(0 1)
          topo_name_list=(complete)
        elif [ "${GROUP_COMM_PATTERN}" = "centralized" ];then
          group_comm_round_list=(4)
          topo_name_list=(complete)
        elif [ "${GROUP_COMM_PATTERN}" = "async-centralized" ];then
          group_comm_round_list=(4)
          topo_name_list=(complete)
      fi

      for TOPO_NAME in ${topo_name_list[@]};
      do

        for GROUP_COMM_ROUND in ${group_comm_round_list[@]};
        do
            if [ ${GROUP_COMM_ROUND} = 0 ]; then
                COMM_ROUND=0
              else
                COMM_ROUND=$((BASE_COMM_ROUND/GROUP_COMM_ROUND))
            fi

            GPU_UTIL_PARSE=${GPU_UTIL_PARSE_LIST[experiment_num%2]}
            log_file="cifar100-group_method=$GROUP_METHOD-group_alpha=$GROUP_ALPHA-partition_alpha=$PARTITION_ALPHA-topo=$TOPO_NAME-group_comm_pattern=$GROUP_COMM_PATTERN-group_comm_round=$GROUP_COMM_ROUND.log"
            echo ${experiment_num}_${log_file}

            # same tau
            nohup mpirun -np $WORKER_NUM \
                -hostfile mpi_host_file \
                python main.py --cf $CONFIG_PATH --random_seed $RANDOM_SEED\
                --worker_num $WORKER_NUM --gpu_util_parse $GPU_UTIL_PARSE\
                --partition_alpha $PARTITION_ALPHA\
                --group_num $GROUP_NUM --group_method $GROUP_METHOD --group_alpha $GROUP_ALPHA\
                --topo_name $TOPO_NAME --group_comm_pattern $GROUP_COMM_PATTERN\
                --comm_round $COMM_ROUND --time_budget $TIME_BUDGET\
                --group_comm_round $GROUP_COMM_ROUND\
                --access_link_capacity $ACCESS_LINK_CAPACITY --core_link_capacity $CORE_LINK_CAPACITY\
                --lan_capacity $LAN_CAPACITY --local_update_time $LOCAL_UPDATE_TIME\
                --enable_ns3\
                --lan_capacity_ratio_list $lan_capacity_ratio_list\
                > batch_log/$log_file 2>&1 &\
                echo $! >> batch_log/process.pid
            sleep 60

            # diff tau, only enabled when GROUP_COMM_ROUND=0
            if [ ${GROUP_COMM_ROUND} = 0 ]; then
              echo ${experiment_num}_diff_tau_${log_file}
              nohup mpirun -np $WORKER_NUM \
                  -hostfile mpi_host_file \
                  python main.py --cf $CONFIG_PATH --random_seed $RANDOM_SEED\
                  --worker_num $WORKER_NUM --gpu_util_parse $GPU_UTIL_PARSE\
                  --partition_alpha $PARTITION_ALPHA\
                  --group_num $GROUP_NUM --group_method $GROUP_METHOD --group_alpha $GROUP_ALPHA\
                  --topo_name $TOPO_NAME --group_comm_pattern $GROUP_COMM_PATTERN\
                  --comm_round $COMM_ROUND --time_budget $TIME_BUDGET\
                  --group_comm_round $GROUP_COMM_ROUND\
                  --access_link_capacity $ACCESS_LINK_CAPACITY --core_link_capacity $CORE_LINK_CAPACITY\
                  --lan_capacity $LAN_CAPACITY --local_update_time $LOCAL_UPDATE_TIME\
                  --enable_ns3\
                  --lan_capacity_ratio_list $lan_capacity_ratio_list\
                  --enable_diff_tau \
                  > batch_log/diff_tau_$log_file 2>&1 &\
                  echo $! >> batch_log/process.pid
            sleep 60
            #    --enable_ns3
            #    --enable_dynamic_topo
            experiment_num=$((experiment_num+1))
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
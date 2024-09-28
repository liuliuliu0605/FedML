CONFIG_PATH=config/fedshakespeare_rnn/fedml_config.yaml
RANDOM_SEED=0
GROUP_NUM=9
COMM_ROUND=3
TIME_BUDGET=0
GROUP_COMM_PATTERN=decentralized
TOPO_NAME=complete
GROUP_METHOD=random
GROUP_COMM_ROUND=10
WORKER_NUM=$(($GROUP_NUM+1))
GPU_UTIL_PARSE_LIST=(localhost:0,0,0,3,2,2,3,0 localhost:0,0,0,2,3,3,2,0)
#GPU_UTIL_PARSE_LIST=(localhost:7,7,6,6 localhost:6,6,7,7)
#GPU_UTIL_PARSE_LIST=(localhost:2,3,3,2 localhost:3,2,2,3)
#GPU_UTIL_PARSE_LIST=(localhost:4,1,4,1 localhost:4,1,4,1)
group_alpha_list=(0.05 0.1 5)
partition_alpha_list=(50 25 0.5)
alpha_index_list=(2)

experiment_num=0
for index in ${alpha_index_list[@]};
do
  PARTITION_ALPHA=${partition_alpha_list[index]}
  GROUP_ALPHA=${group_alpha_list[index]}
  GPU_UTIL_PARSE=${GPU_UTIL_PARSE_LIST[experiment_num%2]}
  log_file="[estimate]fedshakespeare-group_method=$GROUP_METHOD-group_alpha=$GROUP_ALPHA-partition_alpha=$PARTITION_ALPHA-topo=$TOPO_NAME-group_comm_pattern=$GROUP_COMM_PATTERN-group_comm_round=$GROUP_COMM_ROUND.log"
  echo ${experiment_num}_${log_file}
  nohup mpirun -np $WORKER_NUM \
      -hostfile mpi_host_file \
      python main.py --cf $CONFIG_PATH --random_seed $RANDOM_SEED\
      --worker_num $WORKER_NUM --gpu_util_parse $GPU_UTIL_PARSE\
      --partition_alpha $PARTITION_ALPHA\
      --group_num $GROUP_NUM --group_method $GROUP_METHOD --group_alpha $GROUP_ALPHA\
      --topo_name $TOPO_NAME --group_comm_pattern $GROUP_COMM_PATTERN\
      --comm_round $COMM_ROUND --time_budget $TIME_BUDGET\
      --group_comm_round $GROUP_COMM_ROUND\
      --enable_parameter_estimation\
      > batch_log/$log_file 2>&1 \
      & echo $! >> batch_log/process.pid
  sleep 60
  experiment_num=$((experiment_num+1))

done

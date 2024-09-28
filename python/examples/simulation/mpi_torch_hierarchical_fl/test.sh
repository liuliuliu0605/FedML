CONFIG_PATH=config/fedshakespeare_rnn/fedml_config.yaml
#CONFIG_PATH=config/cifar10_resnet20/fedml_config.yaml
RANDOM_SEED=0
GROUP_NUM=9
COMM_ROUND=100
TIME_BUDGET=1000
PARTITION_ALPHA=0.5
GROUP_METHOD=random
GROUP_ALPHA=0
TOPO_NAME=complete
GROUP_COMM_PATTERN=decentralized
GROUP_COMM_ROUND=100
ACCESS_LINK_CAPACITY=1e7
CORE_LINK_CAPACITY=1e10
LAN_CAPACITY=1e11
LOCAL_UPDATE_TIME=0.275
WORKER_NUM=$(($GROUP_NUM+1))
GPU_UTIL_PARSE=localhost:0,0,0,3,2,2,3,0
#GPU_UTIL_PARSE=localhost:0,0,0,1,1,1,0,0
#GPU_UTIL_PARSE_LIST=(localhost:3,1,3,3 localhost:3,1,3,3)
#GPU_UTIL_PARSE_LIST=(localhost:4,1,4,1 localhost:4,1,4,1)



mpirun -np $WORKER_NUM \
        -hostfile mpi_host_file \
        python main.py --cf $CONFIG_PATH --random_seed $RANDOM_SEED\
        --worker_num $WORKER_NUM --gpu_util_parse $GPU_UTIL_PARSE\
        --partition_alpha $PARTITION_ALPHA\
        --group_num $GROUP_NUM --group_method $GROUP_METHOD --group_alpha $GROUP_ALPHA\
        --topo_name $TOPO_NAME --group_comm_pattern $GROUP_COMM_PATTERN\
        --comm_round $COMM_ROUND --time_budget $TIME_BUDGET\
        --group_comm_round $GROUP_COMM_ROUND\
        --access_link_capacity $ACCESS_LINK_CAPACITY --core_link_capacity $CORE_LINK_CAPACITY\
        --lan_capacity $LAN_CAPACITY --local_update_time $LOCAL_UPDATE_TIME

#--enable_ns3
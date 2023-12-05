#!/usr/bin/env bash
RANDOM_SEED=0
GROUP_NUM=9
GROUP_METHOD=hetero
COMM_ROUND=0
TIME_BUDGET=5000
#COMM_ROUND=1000
#TIME_BUDGET=0
TOPO_NAME=ring
ENABLE_DYNAMIC_TOPO=false
ENABLE_NS3=true
GROUP_COMM_PATTERN=decentralized
CONFIG_PATH=config/cifar10_resnet56/fedml_config.yaml

######################################################################
# group_comm_round_list=(0 1 5 10 50 100) # decentralized
# group_comm_round_list=(1 4 10) # centralized
# group_alpha_list=(0.01 0.1 1.0)
group_comm_round_list=(0 1 5 10 50 100)
#group_comm_round_list=(2 3 7 20 30 70)
group_alpha_list=(0.01)
######################################################################

WORKER_NUM=$(($GROUP_NUM+1))
hostname > mpi_host_file
mkdir -p batch_log
# we need to install yq (https://github.com/mikefarah/yq)
# sudo wget https://github.com/mikefarah/yq/releases/latest/download/yq_linux_amd64 -O /usr/bin/yq && sudo chmod +x /usr/bin/yq

yq -i ".common_args.random_seed = ${RANDOM_SEED}" $CONFIG_PATH
yq -i ".device_args.worker_num = ${WORKER_NUM}" $CONFIG_PATH
yq -i ".device_args.gpu_mapping_key = \"mapping_config1_${WORKER_NUM}\"" $CONFIG_PATH
yq -i ".train_args.group_num = ${GROUP_NUM}" $CONFIG_PATH
yq -i ".train_args.comm_round = ${COMM_ROUND}" $CONFIG_PATH
yq -i ".train_args.time_budget = ${TIME_BUDGET}" $CONFIG_PATH
yq -i ".train_args.group_method = \"${GROUP_METHOD}\"" $CONFIG_PATH
yq -i ".train_args.topo_name = \"${TOPO_NAME}\"" $CONFIG_PATH
yq -i ".train_args.group_comm_pattern = \"${GROUP_COMM_PATTERN}\"" $CONFIG_PATH
yq -i ".train_args.enable_dynamic_topo = ${ENABLE_DYNAMIC_TOPO}" $CONFIG_PATH
yq -i ".ns3_args.enable_ns3 = ${ENABLE_NS3}" $CONFIG_PATH

if [ "${TOPO_NAME}" != "random" ]; then
  yq -i ".train_args.topo_edge_probability = 1.0" $CONFIG_PATH
fi

for group_comm_round in ${group_comm_round_list[@]};
do
  echo "group_comm_round=$group_comm_round"
  yq -i ".train_args.group_comm_round = ${group_comm_round}" $CONFIG_PATH

  if [ ${TIME_BUDGET} = 0 ]; then
    yq -i ".train_args.comm_round = $((COMM_ROUND/group_comm_round))" $CONFIG_PATH
  fi

  for group_alpha in ${group_alpha_list[@]};
  do
    echo "group_alpha=$group_alpha"
    yq -i ".train_args.group_alpha = ${group_alpha}" $CONFIG_PATH

    if [ "${GROUP_METHOD}" = "random" ]; then
      yq -i ".train_args.group_alpha = 0" $CONFIG_PATH
    fi

    nohup mpirun -np $WORKER_NUM \
    -hostfile mpi_host_file \
    python torch_step_by_step_example.py --cf $CONFIG_PATH \
    > batch_log/"group_comm_pattern=$GROUP_COMM_PATTERN-group_comm_round=$group_comm_round-topo=$TOPO_NAME-group_alpha=$group_alpha.log" 2>&1 \
    & echo $! >> batch_log/process.pid
    sleep 60
  done
  
done

echo "Finished!"

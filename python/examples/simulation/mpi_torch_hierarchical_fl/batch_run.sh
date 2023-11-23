#!/usr/bin/env bash

GROUP_NUM=9
GROUP_METHOD="hetero"
COMM_ROUND=100000
TIME_BUDGET=2000
TOPO_NAME="complete"
GROUP_COMM_PATTERN="centralized"
CONFIG_PATH=config/cifar10_resnet56/fedml_config.yaml

group_comm_round_list=(1 5 10 50 100) # decentralized
group_comm_round_list=(1 4 10) # centralized
group_alpha_list=(0.01 0.1 1.0)

# test
group_comm_round_list=(1 4 10)
group_alpha_list=(0.01)

WORKER_NUM=$(($GROUP_NUM+1))
hostname > mpi_host_file
mkdir -p batch_log
# we need to install yq (https://github.com/mikefarah/yq)
# sudo wget https://github.com/mikefarah/yq/releases/latest/download/yq_linux_amd64 -O /usr/bin/yq && sudo chmod +x /usr/bin/yq

yq -i ".device_args.worker_num = ${WORKER_NUM}" $CONFIG_PATH
yq -i ".device_args.gpu_mapping_key = \"mapping_config1_${WORKER_NUM}\"" $CONFIG_PATH
yq -i ".train_args.group_num = ${GROUP_NUM}" $CONFIG_PATH
yq -i ".train_args.comm_round = ${COMM_ROUND}" $CONFIG_PATH
yq -i ".train_args.time_budget = ${TIME_BUDGET}" $CONFIG_PATH
yq -i ".train_args.group_method = \"${GROUP_METHOD}\"" $CONFIG_PATH
yq -i ".train_args.topo_name = \"${TOPO_NAME}\"" $CONFIG_PATH
yq -i ".train_args.group_comm_pattern = \"${GROUP_COMM_PATTERN}\"" $CONFIG_PATH

if [ "${GROUP_METHOD}" = "random" ]; then
  yq -i ".train_args.group_alpha = 0" $CONFIG_PATH
fi

if [ "${TOPO_NAME}" != "random" ]; then
  yq -i ".train_args.topo_edge_probability = 1.0" $CONFIG_PATH
fi
for group_comm_round in ${group_comm_round_list[@]};
do
  echo "group_comm_round=$group_comm_round"
  yq -i ".train_args.group_comm_round = ${group_comm_round}" $CONFIG_PATH

  for group_alpha in ${group_alpha_list[@]};
  do
    echo "group_alpha=$group_alpha"
    yq -i ".train_args.group_alpha = ${group_alpha}" $CONFIG_PATH

    nohup mpirun -np $WORKER_NUM \
    -hostfile mpi_host_file \
    python torch_step_by_step_example.py --cf $CONFIG_PATH \
    > batch_log/"group_comm_pattern=$GROUP_COMM_PATTERN-group_comm_round=$group_comm_round-topo=$TOPO_NAME-group_alpha=$group_alpha.log" 2>&1 \
    & echo $! >> batch_log/process.pid
    sleep 3600
  done

done

echo "Finished!"
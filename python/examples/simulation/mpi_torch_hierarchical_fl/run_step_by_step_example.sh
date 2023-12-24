#!/usr/bin/env bash

# run the following command:
# ./run_step_by_step_example.sh 10 config/mnist_lr/fedml_config_topo.yaml
# ./run_step_by_step_example.sh 10 config/cifar10_resnet20/fedml_config.yaml
# ./run_step_by_step_example.sh 10 config/femnist_cnn/fedml_config.yaml
# ./run_step_by_step_example.sh 10 config/cifar100_mobilenet_v3/fedml_config.yaml

WORKER_NUM=$1
CONFIG_PATH=$2

hostname > mpi_host_file

mpirun -np $WORKER_NUM \
-hostfile mpi_host_file \
python torch_step_by_step_example.py --cf $CONFIG_PATH
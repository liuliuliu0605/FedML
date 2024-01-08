import argparse
import logging
import fedml
import os
from fedml import FedMLRunner
from fedml.arguments import Arguments
from fedml.model.cv.resnet import resnet20


def add_args():
    parser = argparse.ArgumentParser(description="FedML")
    parser.add_argument(
        "--yaml_config_file", "--cf", help="yaml configuration file", type=str, default="",
    )

    # default arguments
    parser.add_argument("--run_id", type=str, default="0")

    # default arguments
    parser.add_argument("--rank", type=int, default=0)

    # default arguments
    parser.add_argument("--local_rank", type=int, default=0)

    # For hierarchical scenario
    parser.add_argument("--node_rank", type=int, default=0)

    # default arguments
    parser.add_argument("--role", type=str, default="client")

    # default arguments
    parser.add_argument("--run_device_id", type=str, default="0")

    # cluster arguments
    parser.add_argument("--worker_num", type=int, default=10)
    parser.add_argument("--gpu_util_parse", type=str, default="localhost:2,1,1,1,1,1,1,1")

    # Training arguments
    parser.add_argument("--random_seed", type=int, default=0)
    # parser.add_argument("--federated_optimizer", type=str, default="HierarchicalFL")
    # parser.add_argument("--client_num_in_total", type=int, default=1000)
    # parser.add_argument("--client_num_per_round", type=int, default=1000)
    parser.add_argument("--partition_alpha", type=float, default=0.5)
    parser.add_argument("--comm_round", type=int, default=1)
    parser.add_argument("--time_budget", type=int, default=1)
    # parser.add_argument("--epochs", type=int, default=1)
    # parser.add_argument("--client_optimizer", type=str, default="sgd")
    # parser.add_argument("--learning_rate", type=float, default=0.03)
    # parser.add_argument("--momentum", type=float, default=0.0)
    # parser.add_argument("--server_optimizer", type=str, default="sgd")
    # parser.add_argument("--server_lr", type=float, default=1.0)
    # parser.add_argument("--server_momentum", type=float, default=0.9)
    # args, unknown = parser.parse_known_args()

    # hierarchical arguments
    parser.add_argument("--group_num", type=int, default=9)
    parser.add_argument("--group_method", type=str, default="hetero")
    parser.add_argument("--group_alpha", type=float, default=10.0)
    parser.add_argument("--topo_name", type=str, default="complete")
    parser.add_argument("--group_comm_pattern", type=str, default="decentralized")
    parser.add_argument("--group_comm_round", type=int, default=1)
    parser.add_argument("--enable_ns3", action="store_true")
    parser.add_argument("--enable_dynamic_topo", action="store_true")
    parser.add_argument("--enable_parameter_estimation", action="store_true")

    # ns3 arguments
    parser.add_argument("--access_link_capacity", type=float, default=1.0e+8)
    parser.add_argument("--core_link_capacity", type=float, default=1.0e+10)
    parser.add_argument("--lan_capacity", type=float, default=1.0e+11)
    parser.add_argument("--local_update_time", type=float, default=0.07576)

    parser.add_argument("--override_cmd_args", action="store_true")

    args = parser.parse_args()
    return args


def load_arguments(training_type=None, comm_backend=None):
    cmd_args = add_args()
    logging.info(f"cmd_args: {cmd_args}")

    # Load all arguments from YAML config file
    args = Arguments(cmd_args, training_type, comm_backend, override_cmd_args=cmd_args.override_cmd_args)

    # os.path.expanduser() method in Python is used
    # to expand an initial path component ~( tilde symbol)
    # or ~user in the given path to user’s home directory.
    if hasattr(args, "data_cache_dir"):
        args.data_cache_dir = os.path.expanduser(args.data_cache_dir)
    if hasattr(args, "data_file_path"):
        args.data_file_path = os.path.expanduser(args.data_file_path)
    if hasattr(args, "partition_file_path"):
        args.partition_file_path = os.path.expanduser(args.partition_file_path)
    if hasattr(args, "part_file"):
        args.part_file = os.path.expanduser(args.part_file)

    args.rank = int(args.rank)
    return args


if __name__ == "__main__":
    # init FedML framework
    # args = fedml.init()
    args = load_arguments(fedml._global_training_type, fedml._global_comm_backend)
    logging.info(f"args: {args}")
    args = fedml.init(args)
    logging.info(f"args: {args}")

    # init device
    device = fedml.device.get_device(args)

    # load data
    dataset, output_dim = fedml.data.load(args)

    # load model

    # load model (the size of MNIST image is 28 x 28)
    if args.model == "resnet20":
        model = resnet20(class_num=output_dim)
    else:
        model = fedml.model.create(args, output_dim)

    # start training
    fedml_runner = FedMLRunner(args, device, dataset, model)
    fedml_runner.run()
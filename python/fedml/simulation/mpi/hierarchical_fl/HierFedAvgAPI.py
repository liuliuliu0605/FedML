from .HierFedAvgCloudAggregator import HierFedAVGCloudAggregator
from .HierFedAvgCloudManager import HierFedAVGCloudManager
from .HierFedAvgEdgeManager import HierFedAVGEdgeManager
from .HierGroup import HierGroup
from .utils import stats_group
from ....core import ClientTrainer, ServerAggregator
from ....core.dp.fedml_differential_privacy import FedMLDifferentialPrivacy
from ....core.security.fedml_attacker import FedMLAttacker
from ....core.security.fedml_defender import FedMLDefender
from ....ml.aggregator.aggregator_creator import create_server_aggregator
from ....ml.trainer.trainer_creator import create_model_trainer
from ....core.distributed.topology.symmetric_topology_manager import SymmetricTopologyManager

from ns import ns
from ....ns3_simulator.network import Network

from PIL import Image

import numpy as np
import wandb

def FedML_HierFedAvg_distributed(
    args,
    process_id,
    worker_number,
    comm,
    device,
    dataset,
    model,
    client_trainer: ClientTrainer = None,
    server_aggregator: ServerAggregator = None,
):
    [
        train_data_num,
        test_data_num,
        train_data_global,
        test_data_global,
        train_data_local_num_dict,
        train_data_local_dict,
        test_data_local_dict,
        class_num,
    ] = dataset

    FedMLAttacker.get_instance().init(args)
    FedMLDefender.get_instance().init(args)
    FedMLDifferentialPrivacy.get_instance().init(args)

    if process_id == 0:
        init_cloud_server(
            args,
            device,
            comm,
            process_id,
            worker_number,
            model,
            train_data_num,
            train_data_global,
            test_data_global,
            train_data_local_dict,
            test_data_local_dict,
            train_data_local_num_dict,
            server_aggregator,
            class_num
        )
    else:
        init_edge_server_clients(
            args,
            device,
            comm,
            process_id,
            worker_number,
            model,
            train_data_num,
            train_data_local_num_dict,
            train_data_local_dict,
            test_data_local_dict,
            client_trainer,
            server_aggregator,
        )


def init_cloud_server(
    args,
    device,
    comm,
    rank,
    size,
    model,
    train_data_num,
    train_data_global,
    test_data_global,
    train_data_local_dict,
    test_data_local_dict,
    train_data_local_num_dict,
    server_aggregator,
    class_num
):
    if server_aggregator is None:
        server_aggregator = create_server_aggregator(model, args)
    server_aggregator.set_id(-1)

    worker_num = size - 1

    # set up topology
    topology_manager = None
    if args.group_comm_pattern == 'decentralized':
        topology_manager = SymmetricTopologyManager(worker_num)
        topology_manager.generate_custom_topology(args)
    elif args.group_comm_pattern == 'allreduce':
        assert args.topo_name == 'ring'
        topology_manager = SymmetricTopologyManager(worker_num)
        topology_manager.generate_custom_topology(args)
    elif args.group_comm_pattern in ['centralized', 'decentralized']:
        topology_manager = None

    # setup ns3 simulator
    network = setup_ns3_simulator(args, rank, comm, topology_manager)

    # aggregator
    aggregator = HierFedAVGCloudAggregator(
        train_data_global,
        test_data_global,
        train_data_num,
        train_data_local_dict,
        test_data_local_dict,
        train_data_local_num_dict,
        worker_num,
        device,
        args,
        server_aggregator
    )

    # start the distributed training
    backend = args.backend
    group_indexes, group_to_client_indexes = setup_clients(args, train_data_local_dict, class_num)

    # print group detail
    stats_group(group_to_client_indexes, train_data_local_dict, train_data_local_num_dict, class_num, args)

    server_manager = HierFedAVGCloudManager(args, aggregator, group_indexes, group_to_client_indexes,
                                            comm, rank, size, backend, topology_manager, network)
    server_manager.send_init_msg()
    server_manager.run()


def init_edge_server_clients(
    args,
    device,
    comm,
    process_id,
    size,
    model,
    train_data_num,
    train_data_local_num_dict,
    train_data_local_dict,
    test_data_local_dict,
    group,
    model_trainer=None,
):

    if model_trainer is None:
        model_trainer = create_model_trainer(model, args)

    edge_index = process_id - 1
    backend = args.backend

    # Client assignment is decided on cloud server and the information will be communicated later
    group = HierGroup(
        edge_index,
        train_data_local_dict,
        test_data_local_dict,
        train_data_local_num_dict,
        args,
        device,
        model,
        model_trainer
    )

    # setup ns3 simulator
    network = setup_ns3_simulator(args, process_id, comm)

    edge_manager = HierFedAVGEdgeManager(group, args, comm, process_id, size, backend, network)
    edge_manager.run()


def setup_clients(
    args,
    train_data_local_dict,
    class_num
    ):

    if args.group_method == "random":
        group_indexes = np.random.randint(
            0, args.group_num, args.client_num_in_total
        )
    elif args.group_method == "hetero":
        group_indexes = args.group_indexes

    group_to_client_indexes = {}
    for client_idx, group_idx in enumerate(group_indexes):
        if not group_idx in group_to_client_indexes:
            group_to_client_indexes[group_idx] = []
        group_to_client_indexes[group_idx].append(client_idx)

    return group_indexes, group_to_client_indexes


def setup_ns3_simulator(
    args,
    process_id,
    mpi_comm,
    topology_manager=None
    ):
    ns.core.GlobalValue.Bind("SimulatorImplementationType", ns.core.StringValue("ns3::DistributedSimulatorImpl"))
    # initialize network
    network = Network(access_link_capacity=args.access_link_capacity,
                      core_link_capacity=args.core_link_capacity,
                      lan_capacity=args.lan_capacity,
                      verbose=False,
                      mpi_comm=mpi_comm,
                      seed=args.random_seed)

    network.read_underlay_graph(underlay_name=args.underlay)
    network.select_edge_pses(ps_num=args.group_num, method='mhrw')
    network.set_local_update_config(low=args.local_update_time*0.5, high=args.local_update_time*1.5)

    if args.group_comm_pattern in ['centralized', 'async-centralized']:
        network.select_cloud_ps(method='centroid')

    network.connect_pses(topology_manager, enable_optimization=True)

    if process_id == 0 and args.enable_wandb:
        import io
        buf = io.BytesIO()
        network.plot_underlay_graph(save_path=buf)
        buf.seek(0)
        img1 = Image.open(buf)
        wandb.log({"Underlay": wandb.Image(img1)})
        buf = io.BytesIO()
        network.plot_ps_connectivity_graph(save_path=buf)
        buf.seek(0)
        img2 = Image.open(buf)
        wandb.log({"Connectivity": wandb.Image(img2)})

    return network

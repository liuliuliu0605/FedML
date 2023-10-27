from ns import ns
from fedml.ns3_simulator.network import Network
from mpi4py import MPI
from fedml.core.distributed.topology.symmetric_topology_manager import SymmetricTopologyManager
from argument import parser

import numpy as np
import time

args = parser.parse_args()
print(args)

seed=1
enable_mpi = True if MPI.COMM_WORLD.Get_size() > 1 else False

if enable_mpi:
    ns.core.GlobalValue.Bind("SimulatorImplementationType", ns.core.StringValue("ns3::DistributedSimulatorImpl"))
    py_comm = MPI.COMM_WORLD
else:
    py_comm = None

pattern = args.pattern
model_size = args.model_size
ps_num = args.ps_num
client_num = args.client_num
group_comm_round = args.group_comm_round
mix_comm_round = args.mix_comm_round
client_num_list = [client_num for i in range(ps_num)]
underlay = args.underlay
access_link_capacity = int(args.access_link_capacity)
core_link_capacity = int(args.core_link_capacity)
local_update_config = {
    'low': 0,
    'high': 0
}

# initialize network
network = Network(access_link_capacity=access_link_capacity, core_link_capacity=core_link_capacity,
                  lan_capacity=1e11, verbose=False, mpi_comm=py_comm, seed=seed)
topology_manager = SymmetricTopologyManager(ps_num)
topology_manager.generate_custom_topology(args)

# TODO: how to make sure the order of ps and clients is right ?
network.read_underlay_graph(underlay_name=underlay)
network.select_edge_pses(ps_num=ps_num, method='mhrw')
network.select_cloud_ps(method='centroid')
network.connect_pses(topology_manager, enable_optimization=True)
network.select_clients(client_num_list, method='near_edge_ps')

for _ in range(5):
    # sampled_client_indexes = {0: [90, 652, 591], 1: [220, 528, 24], 2: [817, 133], 3: [906, 273], 4: [81, 644], 5: [239, 827], 6: [897, 73],
    #  7: [228, 516], 8: [808, 120]}
    # client_num_list = [len(sampled_client_indexes[i]) for i in range(ps_num)]
    if enable_mpi:
        comm = ns.cppyy.gbl.Convert2MPIComm(py_comm)
        ns.mpi.MpiInterface.Enable(comm)

    network.construct_network(graph_partition_method='girvan_newman',
                              system_id_list=list(range(1, MPI.COMM_WORLD.Get_size())))

    # network.plot_underlay_graph(save_path="underlay.png")
    # network.plot_ps_connectivity_graph(save_path="connectivity.png")
    # network.plot_ps_overlay_topology(save_path="overlay.png")

    t_a = time.time()
    if pattern == 'pfl':
        ps_ps_delay_matrix, ps_agg_delay, ps_mix_delay = network.run_fl_pfl(model_size=model_size,
                                                                            group_comm_round=group_comm_round,
                                                                            mix_comm_round=group_comm_round,
                                                                            local_update_config=local_update_config,
                                                                            start_time=0, stop_time=10000000)
    elif pattern == 'hfl':
        ps_ps_delay_matrix, ps_agg_delay, ps_mix_delay = network.run_fl_hfl(model_size=model_size,
                                                                            group_comm_round=group_comm_round,
                                                                            local_update_config=local_update_config,
                                                                            start_time=0, stop_time=10000000)
    elif pattern == 'rar':
        ps_ps_delay_matrix, ps_agg_delay, ps_mix_delay = network.run_fl_rar(model_size=model_size,
                                                                            group_comm_round=group_comm_round,
                                                                            local_update_config=local_update_config,
                                                                            start_time=0, stop_time=10000000)
    t_b = time.time()

    time_consuming_matrix = ps_ps_delay_matrix
    if time_consuming_matrix is not None:
        # print(ps_agg_delay)
        # print(ps_mix_delay)
        print("%.2f MB: total=%.5fs, agg=%.5f, mix=%.5f (%ds)" %
              (model_size / 1e6,
               np.max(time_consuming_matrix),
               ps_agg_delay.mean(),
               ps_mix_delay.mean(),
               t_b - t_a))
    print("-" * 50)

if enable_mpi:
    # destroy mpi if mpi mode
    ns.mpi.MpiInterface.Disable()
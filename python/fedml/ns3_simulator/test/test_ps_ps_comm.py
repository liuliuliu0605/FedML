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
verbose=False

if enable_mpi:
    ns.core.GlobalValue.Bind("SimulatorImplementationType", ns.core.StringValue("ns3::DistributedSimulatorImpl"))
    py_comm = MPI.COMM_WORLD
    comm = ns.cppyy.gbl.Convert2MPIComm(py_comm)
    ns.mpi.MpiInterface.Enable(comm)
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
access_link_capacity = args.access_link_capacity
core_link_capacity = args.core_link_capacity

# initialize network
network = Network(access_link_capacity=access_link_capacity,
                  core_link_capacity=core_link_capacity,
                  lan_capacity=1e11,
                  verbose=verbose,
                  mpi_comm=py_comm,
                  seed=seed)
topology_manager = SymmetricTopologyManager(ps_num)
topology_manager.generate_custom_topology(args)

network.read_underlay_graph(underlay_name=underlay)

network.select_edge_pses(ps_num=ps_num, method='mhrw')
network.select_cloud_ps(method='centroid')
network.connect_pses(topology_manager, enable_optimization=True)
# network.add_edge(0, 2)
# network.add_edge(0, 3)

# network.construct_network(graph_partition_method='girvan_newman')
network.construct_network(graph_partition_method='random')

# network.plot_underlay_graph(save_path="underlay.pdf")
# network.plot_ps_connectivity_graph(save_path="connectivity.pdf")
network.plot_ps_overlay_topology(save_path="overlay.pdf")

if pattern == 'pfl':
    app = network.set_pfl_step(model_size, start_time=0, stop_time=10000000, phases=1, initial_message='0')
elif pattern == 'hfl':
    _, app = network.set_hfl_step(model_size, start_time=0, stop_time=10000000, initial_message='0')
elif pattern == 'rar':
    app = network.set_pfl_step(model_size//ps_num, start_time=0, stop_time=10000000, phases=2*(ps_num-1), initial_message='0')
elif pattern == 'async-hfl':
    _, app_list = network.set_async_hfl_step(model_size, start_time=0, stop_time=10000000, initial_message='0')

start_of_simulation = ns.core.Simulator.Now().GetSeconds()

t_a = time.time()
ns.core.Simulator.Run()
t_b = time.time()

ns.core.Simulator.Destroy()

if pattern in ['pfl', 'hfl', 'rar']:
    app.gather_time_consuming_matrix(start_of_simulation=start_of_simulation)
    time_consuming_matrix = app.time_consuming_matrix
elif pattern in ['async-hfl']:
    time_consuming_matrix = np.zeros((ps_num+1, ps_num+1))
    for i, app in enumerate(app_list):
        app.gather_time_consuming_matrix(start_of_simulation=start_of_simulation)
        time_consuming_matrix[0, i+1] = app.time_consuming_matrix[0, 1]
        print(app.time_consuming_matrix[0, 1])

# aa.gather_time_consuming_matrix(start_of_simulation=start_of_simulation)
# print(aa.time_consuming_matrix)
# print(time_consuming_matrix)

if time_consuming_matrix is not None:
    print("%.2f MB: total=%.5fs (%ds)" %
          (model_size / 1e6,
           np.max(time_consuming_matrix),
           t_b - t_a))
print("-" * 50)

if enable_mpi:
    # destroy mpi if mpi mode
    ns.mpi.MpiInterface.Disable()

# if network.system_id == 0:
#     if pattern == 'pfl' or pattern == 'rar':
#         recv_time_list = []
#         for i in range(ps_num):
#             neighbour_list = topology_manager.get_in_neighbor_idx_list(i)
#             recv_time_list.append(time_consuming_matrix[:, i][neighbour_list].max())
#         file_name = '%s-%s-ps_%d-topo_%s-model_%d.txt' % (pattern, underlay, ps_num, args.topo_name, model_size)
#     elif pattern == 'hfl':
#         recv_time_list = time_consuming_matrix[0, 1:]
#         file_name = '%s-%s-ps_%d-model_%d.txt' % (pattern, underlay, ps_num, model_size)
#
#     with open(file_name, 'a') as f:
#         f.write("%d, %d, %.3f, %.3f, %.3f, %d\n" %
#                 (access_link_capacity,
#                  core_link_capacity,
#                  np.min(recv_time_list),
#                  np.max(recv_time_list),
#                  np.mean(recv_time_list),
#                  t_b-t_a))
#
#     if app.finished_or_not():
#         print("done!")
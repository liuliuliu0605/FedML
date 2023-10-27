from network import Network
from mpi4py import MPI
from ns import ns

import mpi4py
import numpy as np
import time

ns.cppyy.add_include_path(mpi4py.get_include())
ns.cppyy.cppdef("""
        #include <mpi4py/mpi4py.h>

        MPI_Comm * Convert2MPIComm(PyObject* py_src){
            if (!PyMPIComm_Get){
                if (import_mpi4py() < 0)
                  {
                  return NULL;
                  }
            }
            if (!PyObject_TypeCheck(py_src, &PyMPIComm_Type)){
                return NULL;
            }
            MPI_Comm *mpiComm = PyMPIComm_Get(py_src);
            return mpiComm;
        }
    """)

ns.core.GlobalValue.Bind("SimulatorImplementationType", ns.core.StringValue("ns3::DistributedSimulatorImpl"))
py_comm = MPI.COMM_WORLD
comm = ns.cppyy.gbl.Convert2MPIComm(py_comm)
ns.mpi.MpiInterface.Enable(comm)

np.random.seed(123456)
ps_num = 9
model_size = int(1e8)
network = Network(ps_num=ps_num, node_capacity=1e9, link_capacity=1e10, verbose=False, mpi_comm=py_comm)
network.read_underlay_graph(underlay_name='geantdistance')
# network.read_underlay_graph(underlay_name='gaia')
# network.plot_underlay_graph(save_path="underlay.png")

network.generate_connectivity_graph()
network.generate_ps_connectivity_graph()
# network.plot_ps_connectivity_graph(save_path="connectivity.png")

from fedml.core.distributed.topology.symmetric_topology_manager import SymmetricTopologyManager
class ARGS:
    topo_name = 'complete'
topology_manager = SymmetricTopologyManager(ps_num)
topology_manager.generate_custom_topology(ARGS())
# print(topology_manager.topology)
network.connect_pses(topology_manager, enable_optimization=True)
# network.plot_ps_overlay_topology(save_path="overlay.png")

network.construct_network()
network.add_communicator()

t_a = time.time()
time_consuming_matrix = network.test(model_size=model_size, start_time=0, stop_time=10000000, phases=1)
t_b = time.time()

if py_comm.rank == 0:
    print("[MPI]Simulation time used: %.3f seconds" % (t_b - t_a))
    print("%.2f MB, time consuming: %.5f seconds" % (model_size/1e6, np.max(time_consuming_matrix)))
    print("-" * 50)

ns.mpi.MpiInterface.Disable()
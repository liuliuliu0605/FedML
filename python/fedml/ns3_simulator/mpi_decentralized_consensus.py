from ns import ns
from decentralized_consensus import DecentralizedConsensus
from communicator import Communicator
from mpi4py import MPI

import sys
import ctypes
import time
import mpi4py


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

protocol = 'tcp'
port = 1080
verbose = True
model_size = int(1e3)
phases = 2
start_time = 0
end_time = 1000

# mpi initialization
ns.core.GlobalValue.Bind("SimulatorImplementationType", ns.core.StringValue("ns3::DistributedSimulatorImpl"))
py_comm = MPI.COMM_WORLD
comm = ns.cppyy.gbl.Convert2MPIComm(py_comm)
ns.mpi.MpiInterface.Enable(comm)
systemId = ns.mpi.MpiInterface.GetSystemId()
systemCount = ns.mpi.MpiInterface.GetSize()

assert systemCount == 2
system_a = 0
system_b = 1

# create nodes
nodes = ns.network.NodeContainer()
p2pNode1 = ns.network.CreateObject("Node")
p2pNode1.SetAttribute("SystemId", ns.core.UintegerValue(system_a))
p2pNode2 = ns.network.CreateObject("Node")
p2pNode2.SetAttribute("SystemId", ns.core.UintegerValue(system_b))
nodes.Add(p2pNode1)
nodes.Add(p2pNode2)

pointToPoint = ns.point_to_point.PointToPointHelper()
pointToPoint.SetDeviceAttribute("DataRate", ns.core.StringValue("10Mbps"))
pointToPoint.SetChannelAttribute("Delay", ns.core.StringValue("2ms"))  # delay should not be zero, otherwise it will suspend
devices = pointToPoint.Install(nodes)

stack = ns.internet.InternetStackHelper()
stack.Install(nodes)

address = ns.internet.Ipv4AddressHelper()
address.SetBase(ns.network.Ipv4Address("10.1.1.0"),
                ns.network.Ipv4Mask("255.255.255.0"))
interfaces = address.Assign(devices)

global_ip_dict = {
    123: interfaces.GetAddress(0),
    456: interfaces.GetAddress(1)
}

comm_a = Communicator(nodes.Get(system_a), _id=123, global_ip_dict=global_ip_dict, protocol=protocol, verbose=verbose)
comm_b = Communicator(nodes.Get(system_b), _id=456, global_ip_dict=global_ip_dict, protocol=protocol, verbose=verbose)
communicator_list = [comm_a, comm_b]
mix_matrix = [
    [1, 1],
    [1, 1]
]

dc = DecentralizedConsensus(model_size, communicator_list, mix_matrix, packet_size=1448, protocol=protocol,
                            verbose=verbose, mpi_comm=py_comm)
dc.run(start_time, end_time, phases=phases)
print(systemId, dc.time_consuming_matrix)
dc.gather_time_consuming_matrix()
print(systemId, dc.time_consuming_matrix)

ns.mpi.MpiInterface.Disable()
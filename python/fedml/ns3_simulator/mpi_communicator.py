from ns import ns
from communicator import Communicator
from sender import Sender
from receiver import Receiver
import sys
from ctypes import c_int
import time

protocol = 'tcp'
port = 1080
verbose = False
phase_size = int(1e6)
phases = 1
start_time = 0
end_time = 1000

# mpi initialization
argc = c_int(len(sys.argv))
argv = sys.argv
ns.core.GlobalValue.Bind("SimulatorImplementationType", ns.core.StringValue("ns3::NullMessageSimulatorImpl"))
ns.mpi.MpiInterface.Enable(argc, argv)
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

if systemId == system_b:
    comm_b.add_app_receiver(comm_a, phase_size, phases, port, start_time=0, stop_time=None)
    comm_b.add_app_sender(comm_a, phase_size, phases, port, 1448, start_time=0, stop_time=None)

if systemId == system_a:
    comm_a.add_app_sender(comm_b, phase_size, phases, port, 1448, start_time=0, stop_time=None)
    comm_a.add_app_receiver(comm_b, phase_size, phases, port, start_time=0, stop_time=None)

ns.core.Simulator.Stop(ns.core.Seconds(end_time))

t_a = time.time()
ns.core.Simulator.Run()
t_b = time.time()
ns.core.Simulator.Destroy()

# finalize mpi
ns.mpi.MpiInterface.Disable()

# print
if systemId == system_a:
    print("comm_a: %d %.5f" % (comm_a.finished_or_not(), comm_a.get_current_time()))
if systemId == system_b:
    print("comm_b: %d %.5f" % (comm_b.finished_or_not(), comm_b.get_current_time()))
print("Simulation time used: %.3f seconds" % (t_b - t_a))

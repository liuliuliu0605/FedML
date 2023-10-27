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
phaseTxSize = int(1e8)
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
systemSender = 0
systemReceiver = 1

# create nodes
nodes = ns.network.NodeContainer()
p2pNode1 = ns.network.CreateObject("Node")
p2pNode1.SetAttribute("SystemId", ns.core.UintegerValue(systemSender))
p2pNode2 = ns.network.CreateObject("Node")
p2pNode2.SetAttribute("SystemId", ns.core.UintegerValue(systemReceiver))
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


if systemId == systemReceiver:
    local_address = ns.network.InetSocketAddress(ns.network.Ipv4Address.GetAny(), port)
    communicator = Communicator(nodes.Get(systemReceiver))
    app_receiver = Receiver(communicator)
    app_receiver.Setup(local_address, phaseTxSize, phases=1, _id=-1, offline_or_online=1,
                       protocol=protocol, verbose=verbose)
    nodes.Get(systemReceiver).AddApplication(app_receiver)
    app_receiver.SetStartTime(ns.core.Seconds(start_time))
    app_receiver.SetStopTime(ns.core.Seconds(end_time))

if systemId == systemSender:
    sinkAddress = ns.network.InetSocketAddress(interfaces.GetAddress(systemReceiver), port)
    communicator = Communicator(nodes.Get(systemSender))
    app_sender = Sender(communicator)
    app_sender.Setup(sinkAddress, phaseTxSize, phases=1, packetSize=1448, _id=-1, offline_or_online=1,
              protocol=protocol, verbose=verbose)
    nodes.Get(systemSender).AddApplication(app_sender)
    app_sender.SetStartTime(ns.core.Seconds(start_time))
    app_sender.SetStopTime(ns.core.Seconds(end_time))

ns.core.Simulator.Stop(ns.core.Seconds(end_time))

t_a = time.time()
ns.core.Simulator.Run()
t_b = time.time()
ns.core.Simulator.Destroy()

# finalize mpi
ns.mpi.MpiInterface.Disable()

# print
if systemId == systemSender:
    print("Sender finished time: %.5f" % app_sender.get_current_time())
if systemId == systemReceiver:
    print("Receiver finished time: %.5f" % app_receiver.get_current_time())
print("Simulation time used: %.3f seconds" % (t_b - t_a))

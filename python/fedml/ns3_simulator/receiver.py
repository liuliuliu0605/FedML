from ns import ns
import cppdef

# normal_close_calssback = ns.cppyy.gbl.make_normal_close_callback(Sender.normal_close_wrapper)
# error_close_calssback = ns.cppyy.gbl.make_error_close_callback(Sender.error_close_wrapper)
# self.socket.SetCloseCallbacks(normal_close_calssback, error_close_calssback)


class Receiver(ns.network.Application):

    object_dict = {}    # str(listening_socket) -> object
    comm_socket_dict = {}   # str(listening_socket) -> [comm_socket, str(peer_socket_address)]
    address_key_dict = {}   # str(peer_socket_address) -> obj_key, i.e., str(listening_socket)

    def __init__(self, communicator):
        super(Receiver, self).__init__()
        self.communicator = communicator
        self.socket = None
        self.protocol = 'tcp'
        self.local_socket_addr = None
        self.phaseRxSize = 0
        self.phases = 1
        self.currentRxSize = 0
        self.verbose = False
        self.message_queue = []
        self.offline_or_online = 1
        self.current_time = 0
        self.current_phase = 0
        self.is_finished = False
        self._id = -1

    def __del__(self):
        self._clean_class_variable()
        self.socket = None

    def _clean_class_variable(self):
        if self.socket is not None and str(self.socket) in Receiver.object_dict:
            if str(self.socket) in Receiver.comm_socket_dict:
                for _, address_str in Receiver.comm_socket_dict[str(self.socket)]:
                    del Receiver.address_key_dict[address_str]
                del Receiver.comm_socket_dict[str(self.socket)]
            del Receiver.object_dict[str(self.socket)]
        # print(Receiver.object_dict, Receiver.address_key_dict)

    def _map_socket_to_receiver(self, socket):
        self._clean_class_variable()
        Receiver.object_dict[str(socket)] = self
        Receiver.comm_socket_dict[str(socket)] = []
        self.socket = socket

    def _create_socket(self):
        node = self.communicator.get_ns_node()
        socket = ns.network.Socket.CreateSocket(
            node,
            ns.core.TypeId.LookupByName(
                "ns3::{:s}SocketFactory".format(self.protocol.capitalize())
            )
        )
        self._map_socket_to_receiver(socket)

        if self.protocol == 'tcp':
            connection_request_callback = ns.cppyy.gbl.make_connection_request(Receiver.connection_request)
            new_connection_created_callback = ns.cppyy.gbl.make_new_connection_created(Receiver.new_connection_created)
            self.socket.SetAcceptCallback(connection_request_callback, new_connection_created_callback)
        else:
            rcv_wrapper_callback = ns.cppyy.gbl.make_rcv_wrapper(Receiver.rcv_wrapper)
            self.socket.SetRecvCallback(rcv_wrapper_callback)

        normal_close_calssback = ns.cppyy.gbl.make_normal_close_callback(Receiver.normal_close_wrapper)
        error_close_calssback = ns.cppyy.gbl.make_error_close_callback(Receiver.error_close_wrapper)
        socket.SetCloseCallbacks(normal_close_calssback, error_close_calssback)

        return self.socket

    def _listen(self):
        if ns.network.InetSocketAddress.IsMatchingType(self.local_socket_addr.ConvertTo()):
            self.socket.Bind(self.local_socket_addr.ConvertTo())
        elif ns.network.Inet6SocketAddress.IsMatchingType(self.local_socket_addr.ConvertTo()):
            self.socket.Bind6(self.local_socket_addr.ConvertTo())
        self.socket.Listen()

    @staticmethod
    def connection_request(socket: ns.Socket, address: ns.Address) -> bool:
        # map peer socket address to the listening socket
        Receiver.address_key_dict[str(address)] = str(socket)
        return True

    @staticmethod
    def normal_close_wrapper(socket: ns.Socket) -> None:
        pass
        # print("receiver normal close:", ns.core.Simulator.Now().GetSeconds())

    @staticmethod
    def error_close_wrapper(socket: ns.Socket) -> None:
        pass
        # print("receiver error close:", ns.core.Simulator.Now().GetSeconds())

    @staticmethod
    def new_connection_created(socket: ns.Socket, address: ns.Address) -> None:
        # map peer socket address to the comm socket
        obj_key = Receiver.address_key_dict[str(address)]
        Receiver.comm_socket_dict[obj_key].append((socket, str(address)))
        rcv_wrapper_callback = ns.cppyy.gbl.make_rcv_wrapper(Receiver.rcv_wrapper)
        socket.SetRecvCallback(rcv_wrapper_callback)

    @staticmethod
    def rcv_wrapper(socket: ns.Socket) -> None:
        address = ns.Address()
        socket.GetPeerName(address)
        obj_key = Receiver.address_key_dict[str(address)]
        Receiver.object_dict[obj_key].rcv_packet(socket)

    @staticmethod
    def go_offline_wrapper(socket: ns.Socket) -> None:
        Receiver.object_dict[str(socket)].go_offline()

    def rcv_packet(self, socket):
        src = ns.network.Address()

        while self.offline_or_online == 1:
            packet = socket.RecvFrom(self.phaseRxSize, 0, src)
            if not packet:
                break

            if ns.network.InetSocketAddress.IsMatchingType(src):
                address = ns.network.InetSocketAddress.ConvertFrom(src)
            elif ns.network.Inet6SocketAddress.IsMatchingType(src):
                address = ns.network.Inet6SocketAddress.ConvertFrom(src)

            self.currentRxSize += packet.GetSize()
            rcv_time = ns.core.Simulator.Now().GetSeconds()
            self.current_time = max(rcv_time, self.current_time)

            if self.verbose:
                print("+ At time %.6f packet sink %d (receiver %d) received %d (%d) bytes from %s port %d" %
                      (rcv_time, self.communicator.get_id(), self._id, packet.GetSize(), self.currentRxSize,
                       address.GetIpv4(), address.GetPort()))

            # receives enough data in this phase and generate data for downstream node
            if self.currentRxSize >= self.phaseRxSize * (self.current_phase + 1):
                self.complete_one_phase(socket, address)

    def complete_one_phase(self, socket, address):
        self.current_phase += 1
        self.message_queue.append(self.current_phase-1)

        if self.verbose:
            print("+ In %d-th phase packet sink %d (receiver %d) received %d total bytes" %
                  (self.current_phase - 1, self.communicator.get_id(), self._id, self.currentRxSize))

        # finish all phases
        if self.current_phase >= self.phases:
            if self.verbose:
                print("[Reception Finished] At time %.6f packet sink %d (receiver %d) received %d total bytes from %s port %d" %
                    (self.current_time, self.communicator.get_id(), self._id, self.currentRxSize, address.GetIpv4(),
                     address.GetPort()))
            self.is_finished = True
            self.StopApplication()
            return

        if self.communicator is not None:
            self.communicator.update_phase()

    def _disconnect(self):
        # Todo: how to make receiver offline and sender be aware of this ?
        pass
        # print(str(self.socket), Receiver.comm_socket_dict)
        # self.socket.ShutdownRecv()
        # self.socket.Close()
        # for comm_socket, _ in Receiver.comm_socket_dict[str(self.socket)]:
        #     del comm_socket
        #     # comm_socket.ShutdownRecv()
        #     comm_socket.Close()

    def go_offline(self):
        self.current_time = ns.core.Simulator.Now().GetSeconds()
        if self.verbose:
            print("# At time %.6f packet sink %d (receiver %d) is offline" %
                  (self.current_time, self.communicator.get_id(), self._id))
        self.offline_or_online = 0
        self._disconnect()

    def schedule_offline_online_event(self, delay=0, offline_interval=1):
        # the receiver will be offline for interval seconds after delay seconds

        if self.is_finished:
            return

        offline_time = ns.core.Time(ns.core.Seconds(delay))
        online_time = ns.core.Time(ns.core.Seconds(delay + offline_interval))

        offline_event = ns.cppyy.gbl.make_offline_online_event(Receiver.go_offline_wrapper, self.socket)
        # online_event = ns.cppyy.gbl.make_offline_online_event(Receiver.go_online_wrapper, self.socket)

        event1 = ns.core.Simulator.Schedule(offline_time, offline_event)
        # event2 = ns.core.Simulator.Schedule(online_time, online_event)
        # self.record_events(event1, event2)

    # def go_online(self):
    #     self.offline_or_online = 1
    #     self.currentTxSize = self.current_phase * self.phaseTxSize  # TODO: breakpoint resume ?
    #     self._create_socket()
    #     self._connect()
    #     self.current_time = ns.core.Simulator.Now().GetSeconds()
    #     if self.verbose:
    #         print("@ At time %.6f packet source %d (sender %d) is online" %
    #               (self.current_time, self.communicator.get_id(), self._id))
    #     self.write_until_buffer_full()

    def get_id(self):
        return self._id

    def get_current_phase(self):
        return self.current_phase

    def get_current_time(self):
        return self.current_time

    def fast_forward(self):
        if self.verbose:
            print("Fake: node %d received data from node %d" % (self.communicator_id, self._id))
        self.currentRxSize = self.phaseRxSize * (self.current_phase + 1)
        ignoring_phase = self.current_phase
        self.current_phase += 1
        if self.current_phase >= self.phases:
            self.is_finished = True
            self.StopApplication()
            return ignoring_phase
        if self.communicator is not None:
            self.communicator.update_phase()
        return ignoring_phase

    def Setup(self, local_socket_addr, phaseTxSize, phases=1, _id=-1, offline_or_online=1, protocol='tcp', verbose=False):
        self.protocol = protocol
        self.local_socket_addr = local_socket_addr
        self._create_socket()
        self.phaseRxSize = phaseTxSize
        self.phases = phases

        self._id = _id
        self.offline_or_online = offline_or_online
        self.verbose = verbose

    def StartApplication(self):
        self.currentRxSize = 0
        self.current_time = 0
        self.current_phase = 0
        self.is_finished = False
        self.message_queue = []
        if self.offline_or_online == 1:
            self._listen()

    def StopApplication(self):
        if self.socket:
            self.socket.ShutdownRecv()
            # self.socket.SetRecvCallback(ns.MakeNullCallback())
            self.socket.Close()


if __name__ == "__main__":
    from communicator import Communicator
    from sender import Sender
    import time

    protocol = 'tcp'
    port = 1080
    verbose = False
    phaseTxSize = int(1e6)
    start_time = 0
    end_time = 10000

    # if verbose:
    # ns.core.LogComponentEnable("OnOffApplication", ns.core.LOG_LEVEL_INFO)
    # ns.core.LogComponentEnable("TcpSocketBase", ns.core.LOG_LEVEL_INFO)
    # ns.core.LogComponentEnable("PacketSink", ns.core.LOG_LEVEL_INFO)
    # ns.core.Config.SetDefault("ns3::TcpSocket::SndBufSize", ns.core.UintegerValue(1 << 20))
    # ns.core.Config.SetDefault("ns3::TcpSocket::RcvBufSize", ns.core.UintegerValue(1 << 20))

    nodes = ns.network.NodeContainer()
    nodes.Create(2)

    pointToPoint = ns.point_to_point.PointToPointHelper()
    pointToPoint.SetDeviceAttribute("DataRate", ns.core.StringValue("10Mbps"))
    pointToPoint.SetChannelAttribute("Delay", ns.core.StringValue("2ms"))
    devices = pointToPoint.Install(nodes)

    stack = ns.internet.InternetStackHelper()
    stack.Install(nodes)

    address = ns.internet.Ipv4AddressHelper()
    address.SetBase(ns.network.Ipv4Address("10.1.1.0"),
                    ns.network.Ipv4Mask("255.255.255.0"))
    interfaces = address.Assign(devices)

    # receiving app
    local_address = ns.network.InetSocketAddress(ns.network.Ipv4Address.GetAny(), port)
    communicator = Communicator(nodes.Get(1))
    app_receiver = Receiver(communicator)
    app_receiver.Setup(local_address, phaseTxSize, phases=1, _id=-1, offline_or_online=1,
                       protocol=protocol, verbose=verbose)
    nodes.Get(1).AddApplication(app_receiver)
    app_receiver.SetStartTime(ns.core.Seconds(start_time))
    app_receiver.SetStopTime(ns.core.Seconds(end_time))

    # sending app
    sinkAddress = ns.network.InetSocketAddress(interfaces.GetAddress(1), port)
    communicator = Communicator(nodes.Get(0))
    app_sender = Sender(communicator)
    app_sender.Setup(sinkAddress, phaseTxSize, phases=1, packetSize=1448, _id=-1, offline_or_online=1,
              protocol=protocol, verbose=verbose)
    nodes.Get(0).AddApplication(app_sender)
    app_sender.SetStartTime(ns.core.Seconds(start_time))
    app_sender.SetStopTime(ns.core.Seconds(end_time))

    # start simulating
    t_a = time.time()
    ns.core.Simulator.Run()
    t_b = time.time()
    print("Sender finished time: %.5f" % app_sender.get_current_time())
    print("Receiver finished time: %.5f" % app_receiver.get_current_time())
    print("Simulation time used: %.3f seconds" % (t_b - t_a))
    ns.core.Simulator.Destroy()
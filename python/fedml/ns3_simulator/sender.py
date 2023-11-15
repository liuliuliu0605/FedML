from ns import ns
from . import cppdef

class Sender(ns.network.Application):
    # the size of data to be transmitted = m_phaseTxSize * m_phases
    # sending procedure will be finished when entering the m_phases-th phase
    # entering each phase needs message from message_queue, which will be populated by communicator
    # sender will be idle if sending buffer is null or message queue is null or offline
    # a sender corresponds to a socket and a communicator may contain multiple senders

    object_dict = {}    # str(socket) -> object

    def __init__(self, communicator):
        super(Sender, self).__init__()
        self.communicator = communicator
        self.socket = None
        self.protocol = 'tcp'
        self.peer = None
        self.phaseTxSize = 0  # data size in each phase
        self.phases = 1       # number of phases
        self.packetSize = 1448  # packet size
        self.currentTxSize = 0
        self.message_queue = []
        self.verbose = False
        self.offline_or_online = 1
        self.current_time = 0
        self.current_phase = 0
        self.is_idle = True
        self.is_finished = False
        self._id = '-1'
        self.event_list = []

    def __del__(self):
        self._clean_class_variable()
        self.socket = None

    def _clean_class_variable(self):
        # str(socket) represents its unique address
        if self.socket is not None and str(self.socket) in Sender.object_dict:
            del Sender.object_dict[str(self.socket)]
        # print(Sender.object_dict)

    @staticmethod
    def event_test():
        print("this is an event!!!!!")

    @staticmethod
    def write_wrapper(socket: ns.Socket, tx_available: int) -> None:
        Sender.object_dict[str(socket)].write_until_buffer_full()

    @staticmethod
    def go_online_wrapper(socket: ns.Socket) -> None:
        Sender.object_dict[str(socket)].go_online()

    @staticmethod
    def go_offline_wrapper(socket: ns.Socket) -> None:
        Sender.object_dict[str(socket)].go_offline()

    @staticmethod
    def add_message_wrapper(socket: ns.Socket, message: str) -> None:
        Sender.object_dict[str(socket)].add_message(message)

    @staticmethod
    def connection_succeeded(socket: ns.Socket) -> None:
        write_until_buffer_full_callback = ns.cppyy.gbl.make_write_callback(Sender.write_wrapper)
        socket.SetSendCallback(write_until_buffer_full_callback)
        # print(ns.core.Simulator.Now().GetSeconds(), str(socket))
        # Sender.object_dict[str(socket)].write_until_buffer_full()

    @staticmethod
    def connection_failed(socket: ns.Socket) -> None:
        raise ConnectionError

    @staticmethod
    def normal_close_wrapper(socket: ns.Socket) -> None:
        print("sender normal close:", ns.core.Simulator.Now().GetSeconds())

    @staticmethod
    def error_close_wrapper(socket: ns.Socket) -> None:
        print("sender error close:", ns.core.Simulator.Now().GetSeconds())

    def _create_socket(self):
        node = self.communicator.get_ns_node()
        socket = ns.network.Socket.CreateSocket(
            node,
            ns.core.TypeId.LookupByName(
                "ns3::{:s}SocketFactory".format(self.protocol.capitalize())
            )
        )
        self._map_socket_to_sender(socket)

        connection_succeeded_callback = ns.cppyy.gbl.make_connection_succeeded_callback(Sender.connection_succeeded)
        connection_failed_callback = ns.cppyy.gbl.make_connection_failed_callback(Sender.connection_failed)
        socket.SetConnectCallback(connection_succeeded_callback, connection_failed_callback)

        normal_close_calssback = ns.cppyy.gbl.make_normal_close_callback(Sender.normal_close_wrapper)
        error_close_calssback = ns.cppyy.gbl.make_error_close_callback(Sender.error_close_wrapper)
        socket.SetCloseCallbacks(normal_close_calssback, error_close_calssback)

        return socket

    def _connect(self):
        assert self.offline_or_online == 1
        if ns.network.InetSocketAddress.IsMatchingType(self.peer.ConvertTo()):
            self.socket.Bind()
        elif ns.network.Inet6SocketAddress.IsMatchingType(self.peer.ConvertTo()):
            self.socket.Bind6()
        self.socket.Connect(self.peer.ConvertTo())

    def _disconnect(self):
        assert self.offline_or_online == 0
        self.socket.ShutdownSend()
        self.socket.Close()

    def _map_socket_to_sender(self, socket):
        # one socket corresponds to one sender and delete the map if socket is changed
        if str(self.socket) in Sender.object_dict:
            del Sender.object_dict[str(self.socket)]
        Sender.object_dict[str(socket)] = self
        self.socket = socket

    def make_packet(self, size):

        packet = ns.network.Packet(int(size))
        return packet

    def write_until_buffer_full(self):
        self.is_idle = False

        while self.check_before_sending() and self.currentTxSize < self.phaseTxSize * (self.current_phase + 1):

            left = self.phaseTxSize * (self.current_phase + 1) - self.currentTxSize
            data_offset = self.currentTxSize % self.packetSize
            to_write = self.packetSize - data_offset
            to_write = min(to_write, left)
            to_write = min(to_write, self.socket.GetTxAvailable())
            packet = self.make_packet(to_write)
            amount_sent = self.socket.Send(packet, 0)

            if amount_sent <= 0:
                self.is_idle = True
                return

            self.current_time = ns.core.Simulator.Now().GetSeconds()
            self.currentTxSize += amount_sent
            if self.verbose:
                print("- At time %.6f packet source %s (sender %s) sent %d (%d) bytes to %s port %d" %
                      (self.current_time, self.communicator.get_id(), self._id, amount_sent, self.currentTxSize,
                       self.peer.GetIpv4(), self.peer.GetPort()))

        self.is_idle = True

        if self.currentTxSize >= self.phaseTxSize * (self.current_phase + 1):
            self.complete_one_phase()

    def complete_one_phase(self):
        self.current_phase += 1
        self.message_queue.pop(0)
        # self.get_communicator().update_global_comm_matrix(self.current_phase - 1, self._id) //TODO

        if self.verbose:
            print("- In %d-th phase packet source %s (sender %s) sent %d total bytes" %
                  (self.current_phase - 1, self.communicator.get_id(), self._id, self.currentTxSize))

        if self.current_phase >= self.phases:
            if self.verbose:
                print("[Transmission Finished] At time %.6f packet source %s (sender %s) sent %d total bytes to %s port %d" %
                      (self.current_time, self.communicator.get_id(), self._id, self.currentTxSize,
                       self.peer.GetIpv4(), self.peer.GetPort()))
            self.is_finished = True
            self.StopApplication()
            return

        # check message queue to send the next message
        if len(self.message_queue) > 0:
            self.write_until_buffer_full()

    def check_before_sending(self):
        ready = True
        warning_message = None
        if len(self.message_queue) == 0:
            ready = False
            warning_message = "Warning: node %s (sender %s) message queue was empty!" % \
                              (self.communicator.get_id(), self._id)
        elif self.socket.GetTxAvailable() <= 0:
            ready = False
            warning_message = "Warning: not available sending buffer!"
        elif self.offline_or_online == 0:
            ready = False
            warning_message = "Warning: node %s (sender %s) was offline!" % (self.communicator.get_id(), self._id)
        if self.verbose and warning_message is not None:
            print(warning_message)
        return ready

    def go_offline(self):
        self.current_time = ns.core.Simulator.Now().GetSeconds()
        if self.verbose:
            print("# At time %.6f packet source %s (sender %s) is offline" %
                  (self.current_time, self.communicator.get_id(), self._id))
        self.offline_or_online = 0
        self._disconnect()

    def go_online(self):
        self.current_time = ns.core.Simulator.Now().GetSeconds()
        if self.verbose:
            print("@ At time %.6f packet source %s (sender %s) is online" %
                  (self.current_time, self.communicator.get_id(), self._id))
        self.offline_or_online = 1
        self.currentTxSize = self.current_phase * self.phaseTxSize  # TODO: breakpoint resume ?
        self._create_socket()
        self._connect()
        self.write_until_buffer_full()

    def record_events(self, *events):
        # event is recorded and deleted if it expires
        self.event_list = [event for event in self.event_list if not event.IsExpired()]
        if not self.is_finished:
            self.event_list.extend(events)

    def schedule_offline_online_event(self, delay=0, offline_interval=1):
        # the sender will be offline for interval seconds after delay seconds

        if self.is_finished:
            return

        offline_time = ns.core.Time(ns.core.Seconds(delay))
        online_time = ns.core.Time(ns.core.Seconds(delay + offline_interval))

        offline_event = ns.cppyy.gbl.make_offline_online_event(Sender.go_offline_wrapper, self.socket)
        online_event = ns.cppyy.gbl.make_offline_online_event(Sender.go_online_wrapper, self.socket)

        event1 = ns.core.Simulator.Schedule(offline_time, offline_event)
        event2 = ns.core.Simulator.Schedule(online_time, online_event)
        self.record_events(event1, event2)

    def schedule_add_message(self, msg='', delay=0):
        start_time = ns.core.Time(ns.core.Seconds(delay))
        add_message_event = ns.cppyy.gbl.make_add_message_event(Sender.add_message_wrapper, self.socket, msg)
        event = ns.core.Simulator.Schedule(start_time, add_message_event)
        self.record_events(event)

    def add_message(self, msg=''):
        if self.verbose:
            print("Node %s (sender %s) receives new message" % (self.communicator.get_id(), self._id))
        self.message_queue.append(msg)
        if len(self.message_queue) == 1 and self.is_idle:
            # this case indicates sender has sending all the messages in the queue before
            # message must be sent in order
            self.write_until_buffer_full()

    def get_current_time(self):
        return self.current_time

    def get_current_phase(self):
        return self.current_phase

    def is_finished(self):
        return self.is_finished

    def get_id(self):
        return self._id

    def get_communicator(self):
        return self.communicator

    def fast_forward(self):
        assert len(self.message_queue) > 0
        self.current_time = max(self.current_time, ns.core.Simulator.Now().GetSeconds())
        self.currentTxSize = self.phaseTxSize * (self.current_phase + 1)
        ignoring_phase = self.current_phase
        self.current_phase += 1
        self.message_queue.pop(0)
        if self.current_phase >= self.phases:
            self.is_finished = True

            if self.communicator is not None:
                self.communicator.after_finish()

            self.StopApplication()

        if self.verbose:
            print("Fake: node %s sending data to node %s" % (self.communicator.get_id(), self._id))

        return ignoring_phase

    def Setup(self, peer_socket_addr, phaseTxSize, phases=1, packetSize=1448, _id='-1', offline_or_online=1,
              protocol='tcp', initial_message=None, verbose=False):
        self.protocol = protocol
        self.peer = peer_socket_addr
        self._create_socket()
        self.phaseTxSize = phaseTxSize
        self.phases = phases
        self.packetSize = packetSize

        self._id = _id
        self.offline_or_online = offline_or_online
        self.message_queue = [initial_message] if initial_message is not None else []
        self.verbose = verbose

    def StartApplication(self):
        self.currentTxSize = 0
        self.current_time = 0
        self.current_phase = 0
        self.is_idle = True
        self.is_finished = False
        # self.message_queue = [0]
        if self.offline_or_online == 1:
            self._connect()

    def StopApplication(self):
        if self.socket:
            self.socket.ShutdownSend()
            self.socket.Close()

        for event in self.event_list:
            event.Cancel()


if __name__ == "__main__":
    import time
    from .communicator import Communicator
    import numpy as np

    np.random.seed(123456)

    protocol = 'tcp'
    port = 1080
    verbose = False

    if verbose:
        ns.core.LogComponentEnable("PacketSink", ns.core.LOG_LEVEL_INFO)
        # ns.core.LogComponentEnable("TcpSocketBase", ns.core.LOG_LEVEL_INFO)
    ns.core.Config.SetDefault("ns3::TcpSocket::SndBufSize", ns.core.UintegerValue(1 << 20))
    ns.core.Config.SetDefault("ns3::TcpSocket::RcvBufSize", ns.core.UintegerValue(1 << 20))
    # ns.core.Config.SetDefault("ns3::TcpSocket::DelAckTimeout", ns.core.UintegerValue(2))

    # create nodes
    nodes = ns.network.NodeContainer()
    nodes.Create(2)

    # create p2p link
    pointToPoint = ns.point_to_point.PointToPointHelper()
    pointToPoint.SetDeviceAttribute("DataRate", ns.core.StringValue("10Mbps"))
    pointToPoint.SetChannelAttribute("Delay", ns.core.StringValue("5ms"))
    devices = pointToPoint.Install(nodes)

    stack = ns.internet.InternetStackHelper()
    stack.Install(nodes)

    # assign addresses
    address = ns.internet.Ipv4AddressHelper()
    address.SetBase(ns.network.Ipv4Address("10.1.1.0"),
                    ns.network.Ipv4Mask("255.255.255.0"))
    interfaces = address.Assign(devices)

    # receiving app
    sink_address = ns.network.InetSocketAddress(interfaces.GetAddress(1), port)
    sink_helper = ns.applications.PacketSinkHelper("ns3::{:s}SocketFactory".format(protocol.capitalize()),
                                                   sink_address.ConvertTo())
    sink_app = sink_helper.Install(nodes.Get(1))
    sink_app.Start(ns.core.Seconds(0))
    sink_app.Stop(ns.core.Seconds(20))

    # sending app
    communicator = Communicator(nodes.Get(0), _id=0)
    app = Sender(communicator)
    app.Setup(sink_address, int(1e5), phases=1, packetSize=1448, _id='1', offline_or_online=1,
              protocol=protocol, initial_message=1, verbose=verbose)
    nodes.Get(0).AddApplication(app)
    app.SetStartTime(ns.core.Seconds(0))
    app.SetStopTime(ns.core.Seconds(20))

    # sender goes offline after 1 second for 1 second
    # app.schedule_offline_online_event(0, 10)

    # start simulating
    t_a = time.time()
    ns.core.Simulator.Run()
    t_b = time.time()
    print("Sender finished time: %.5f" % app.get_current_time())
    print("Simulation time used: %.3f" % (t_b - t_a))
    if app.is_finished:
        print("done!")
    ns.core.Simulator.Destroy()
from ns import ns
from .sender import Sender
from .receiver import Receiver

import numpy as np


class Communicator:

    def __init__(self, ns_node, _id='-1', global_ip_dict=None, protocol='tcp', offline_params={}, verbose=False):
        self.ns_node = ns_node
        self._id = _id
        self.global_ip_dict = global_ip_dict
        self.protocol = protocol
        self.offline_params = offline_params
        self.verbose = verbose

        self.app_sender_dict = {}
        self.app_receiver_dict = {}
        self.is_offline = False
        self.next_phase = 1
        self.finish_callback = None
        self.finish_callback_params = None
        self.phase_callback = None
        self.phase_callback_params = None

    def __del__(self):
        self.reset()

    def get_ns_node(self):
        return self.ns_node

    def get_ns_node_system_id(self):
        return self.ns_node.GetSystemId()

    def get_id(self):
        return self._id

    def reset(self):
        self.app_sender_dict = {}
        self.app_receiver_dict = {}
        self.is_offline = False
        self.next_phase = 1
        self.finish_callback = None
        self.finish_callback_params = None
        self.phase_callback = None
        self.phase_callback_params = None

    def get_app_receiver_dict(self):
        return self.app_receiver_dict

    def get_app_sender_dict(self):
        return self.app_sender_dict

    def get_current_time(self):
        # equals to the max time of senders and receivers
        current_time = 0
        for sender in self.app_sender_dict.values():
            current_time = max(current_time, sender.get_current_time())
        for receiver in self.app_receiver_dict.values():
            current_time = max(current_time, receiver.get_current_time())
        return current_time

    def get_rcv_time(self, src=None):
        # equals to the max time of all receivers or some receiver for src node
        if src is None:
            # receive all updates from neighbours
            current_time = 0
            for receiver in self.app_receiver_dict.values():
                current_time = max(current_time, receiver.get_current_time())
        else:
            if src in self.app_receiver_dict:
                current_time = self.app_receiver_dict[src].get_current_time()
            else:
                current_time = 0
        return current_time

    def update_rcv_time(self, rcv_time, src=None):
        # update current time of receiver if fastforward is enabled
        if src is None:
            for receiver in self.app_receiver_dict.values():
               receiver.current_time = rcv_time
        else:
            if src in self.app_receiver_dict:
                self.app_receiver_dict[src] = rcv_time

    def finished_or_not(self):
        # finished if and only if senders and receivers all finished
        sender_states = [sender.is_finished for sender in self.app_sender_dict.values()]
        receiver_states = [receiver.is_finished for receiver in self.app_receiver_dict.values()]
        return np.all(sender_states) and np.all(receiver_states)

    def add_app_receiver(self, src_communicator, phase_rx_size, phases, port, start_time=0, stop_time=None):
        # create receiver and communicator with communicator_id will send data to this receiver
        communicator_id = src_communicator.get_id()
        local_address = ns.network.InetSocketAddress(ns.network.Ipv4Address.GetAny(), port)
        app_receiver = Receiver(communicator=self)
        app_receiver.Setup(local_address, phase_rx_size, phases=phases, _id=communicator_id, offline_or_online=1,
                           protocol=self.protocol, verbose=self.verbose)
        self.get_ns_node().AddApplication(app_receiver)
        app_receiver.SetStartTime(ns.core.Seconds(start_time))
        if stop_time is not None:
            app_receiver.SetStopTime(ns.core.Seconds(stop_time))
        if communicator_id in self.app_receiver_dict:
            Warning("Duplicate communicator id!")
        else:
            self.app_receiver_dict[communicator_id] = app_receiver

        return app_receiver

    def add_app_sender(self, dst_communicator, phase_rx_size, phases, port, packet_size,
                       initial_message='0', start_time=0, stop_time=None):
        # create sender and send data to communicator with communicator_id
        communicator_id = dst_communicator.get_id()
        sink_address = ns.network.InetSocketAddress(self.global_ip_dict[communicator_id], port)
        app_sender = Sender(self)
        app_sender.Setup(sink_address, phase_rx_size, phases=phases, packetSize=packet_size, _id=communicator_id,
                         offline_or_online=1, protocol=self.protocol, initial_message=initial_message,
                         verbose=self.verbose)
        self.get_ns_node().AddApplication(app_sender)
        app_sender.SetStartTime(ns.core.Seconds(start_time))
        if stop_time is not None:
            app_sender.SetStopTime(ns.core.Seconds(stop_time))
        if communicator_id in self.app_sender_dict:
            Warning("Duplicate communicator id!")
        else:
            self.app_sender_dict[communicator_id] = app_sender

        return app_sender

    def update_phase(self):
        is_update = np.all(
            [receiver.get_current_phase() >= self.next_phase for receiver in self.app_receiver_dict.values()]
        )
        if is_update:
            if self.verbose:
                print("Node %s entered %d-th phase" % (self._id, self.next_phase))
            self.generate_message()
            self.next_phase += 1

    def after_finish(self):
        if self.finished_or_not() and self.finish_callback is not None:
            self.finish_callback(self.finish_callback_params)

    def register_finish_callback(self, callback, params):
        self.finish_callback = callback
        self.finish_callback_params = params

    def after_phase(self):
        is_update = np.all(
            [receiver.get_current_phase() >= self.next_phase for receiver in self.app_receiver_dict.values()]
        )
        if is_update and self.phase_callback is not None:
            self.phase_callback(self.phase_callback_params)

    def register_phase_callback(self, callback, params):
        self.phase_callback = callback
        self.phase_callback_params = params

    def generate_message(self, message='', delay=0):
        # TODO: how generate next message and calculate the latency
        if self.verbose:
            print("Node %s generate new message" % self._id)
        for sender in self.app_sender_dict.values():
            if delay == 0:
                sender.add_message(message)
            else:
                sender.schedule_add_message(message, delay)


if __name__ == "__main__":
    import time

    np.random.seed(123456)
    phase_size = int(1e3)
    phases = 2
    protocol = 'tcp'
    verbose = False

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

    global_ip_dict = {
        'a': interfaces.GetAddress(0),
        'b': interfaces.GetAddress(1)
    }

    comm_a = Communicator(nodes.Get(0), _id='a', global_ip_dict=global_ip_dict, protocol=protocol, verbose=verbose)
    comm_b = Communicator(nodes.Get(1), _id='b', global_ip_dict=global_ip_dict, protocol=protocol, verbose=verbose)

    comm_a.add_app_receiver(comm_b, phase_size, phases, 5000, start_time=0, stop_time=None)
    comm_b.add_app_sender(comm_a, phase_size, phases, 5000, 1448, start_time=0, stop_time=None)

    comm_a.add_app_sender(comm_b, phase_size, phases, 5000, 1448, start_time=0, stop_time=None)
    comm_b.add_app_receiver(comm_a, phase_size, phases, 5000, start_time=0, stop_time=None)

    t_a = time.time()
    ns.core.Simulator.Run()
    t_b = time.time()
    ns.core.Simulator.Destroy()
    print("Communicator a finished time: %.5f" % comm_a.get_current_time())
    print("Communicator b finished time: %.5f" %  comm_b.get_current_time())
    print("Simulation time used: %.3f" % (t_b - t_a))

    if comm_a.finished_or_not() and comm_b.finished_or_not():
        print("done!")

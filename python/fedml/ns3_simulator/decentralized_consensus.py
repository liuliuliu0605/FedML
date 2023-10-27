from ns import ns
from communicator import Communicator

import numpy as np

BASE_PORT = 5000


class DecentralizedConsensus:

    def __init__(self, model_size, communicator_list, mix_matrix, packet_size=1448, protocol='tcp',
                 offline_params={}, verbose=False, mpi_comm=None):
        assert protocol in ['tcp', 'udp']
        assert len(communicator_list) == len(mix_matrix)

        self.model_size = model_size
        self.communicator_list = communicator_list
        self.mix_matrix = np.array(mix_matrix)
        self.packet_size = packet_size
        self.protocol = protocol
        self.verbose = verbose
        self.mpi_comm = mpi_comm
        self.system_id = mpi_comm.Get_rank() if mpi_comm is not None else -1
        self.system_count = mpi_comm.Get_size() if mpi_comm is not None else 0

        self.node_num = len(communicator_list)
        self.communicator_id_list = [comm.get_id() for comm in self.communicator_list]
        self.time_consuming_matrix = None

    def __del__(self):
        self.reset()

    def reset(self):
        for communicator in self.communicator_list:
            communicator.reset()

    def init_app(self, start_time=0, stop_time=None, phases=1):
        # source node src (ps_node_container.Get(src)) will send data to
        # sink node dst (ps_node_container.Get(dst)) with "src+BASE_PORT" port

        for src in range(self.node_num):
            for dst in range(self.node_num):
                if src != dst and self.mix_matrix[src, dst] > 0:

                    src_comm = self.communicator_list[src]
                    src_comm_id = self.communicator_id_list[src]
                    dst_comm = self.communicator_list[dst]
                    dst_comm_id = self.communicator_id_list[dst]

                    if self.verbose:
                        print("Mixing model: PS %d -> PS %d" % (src_comm_id, dst_comm_id))

                    if self.system_id < 0 or dst_comm.get_ns_node_system_id() == self.system_id:
                        dst_comm.add_app_receiver(src_comm, self.model_size, phases,
                                                  src+BASE_PORT, start_time, stop_time)

                    if self.system_id < 0 or src_comm.get_ns_node_system_id() == self.system_id:
                        src_comm.add_app_sender(dst_comm, self.model_size, phases,
                                                src+BASE_PORT, self.packet_size, start_time, stop_time)

    def run(self, start_time, stop_time, phases=1):
        self.init_app(start_time, stop_time, phases)
        start_of_simulation = ns.core.Simulator.Now().GetSeconds()
        ns.core.Simulator.Stop(ns.core.Seconds(stop_time))

        import time
        t_a = time.time()
        ns.core.Simulator.Run()
        t_b = time.time()
        print("ns.core.Simulator.Run() time used: %.3f seconds" % (t_b - t_a))

        ns.core.Simulator.Destroy()
        # ns.mpi.MpiInterface.Disable()

        time_rcv_matrix = np.zeros((self.node_num, self.node_num))
        for src in range(self.node_num):
            for dst in range(self.node_num):
                src_comm_id = self.communicator_id_list[src]
                time_rcv_matrix[src, dst] = max(self.communicator_list[dst].get_rcv_time(src_comm_id),
                                                      start_of_simulation + start_time)

        if self.finished_or_not():
            self.time_consuming_matrix = time_rcv_matrix - start_of_simulation
        else:
            self.time_consuming_matrix = None

        return self.time_consuming_matrix

    def gather_time_consuming_matrix(self):
        if self.system_id < 0:
            return
        if self.system_id == 0:
            for i in range(1, self.system_count):
                rev_time_arr = self.mpi_comm.recv(source=i, tag=11)
                # self.time_consuming_matrix[:, i] = rev_time_arr
                self.time_consuming_matrix = np.maximum(self.time_consuming_matrix, rev_time_arr)
        else:
            # self.mpi_comm.send(self.time_consuming_matrix[:, self.system_id], dest=0, tag=11)
            self.mpi_comm.send(self.time_consuming_matrix, dest=0, tag=11)

    def finished_or_not(self):
        communicator_states = [communicator.finished_or_not() for communicator in self.communicator_list]
        # if self.system_id < 0:
        #     return np.all(communicator_states)
        # if self.system_id == 0:
        #     for i in range(1, self.system_count):
        #         finished_or_not = self.mpi_comm.recv(source=i, tag=22)
        #         communicator_states[i] = finished_or_not
        # else:
        #     self.mpi_comm.send(self.communicator_list[self.system_id].finished_or_not(), dest=0, tag=22)
        return np.all(communicator_states)


if __name__ == '__main__':
    model_size = int(1e3)
    phases = 2
    protocol = 'tcp'
    verbose = True
    start_time = 0
    stop_time = 10

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
        123: interfaces.GetAddress(0),
        456: interfaces.GetAddress(1)
    }

    comm_a = Communicator(nodes.Get(0), _id=123, global_ip_dict=global_ip_dict, protocol=protocol, verbose=verbose)
    comm_b = Communicator(nodes.Get(1), _id=456, global_ip_dict=global_ip_dict, protocol=protocol, verbose=verbose)
    communicator_list = [comm_a, comm_b]

    mix_matrix = [
        [1, 1],
        [1, 1]
    ]

    dc = DecentralizedConsensus(model_size, communicator_list, mix_matrix, packet_size=1448, protocol=protocol,
                                verbose=verbose, mpi_comm=None)
    time_consuming_matrix = dc.run(start_time, stop_time, phases=phases)

    print(time_consuming_matrix)




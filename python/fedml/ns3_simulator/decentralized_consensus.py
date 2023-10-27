from ns import ns
from .communicator import Communicator

import numpy as np


class DecentralizedConsensus:

    def __init__(self, model_size, communicator_list, mix_matrix, packet_size=1448, protocol='tcp',
                 offline_params={}, verbose=False, mpi_comm=None, base_port=5000):
        assert protocol in ['tcp', 'udp']
        assert len(communicator_list) == len(mix_matrix)

        self.model_size = model_size    # Bytes
        self.communicator_list = communicator_list
        self.mix_matrix = np.array(mix_matrix)
        self.packet_size = packet_size
        self.protocol = protocol
        self.verbose = verbose
        self.mpi_comm = mpi_comm
        self.system_id = mpi_comm.Get_rank() if mpi_comm is not None else 0
        self.system_count = mpi_comm.Get_size() if mpi_comm is not None else 1
        self.base_port = base_port  # TODO: how to use port not in used?

        self.node_num = len(communicator_list)
        self.communicator_id_list = [comm.get_id() for comm in self.communicator_list]
        self.time_consuming_matrix = None

        self.start_time = None
        self.stop_time = None

    def __del__(self):
        pass
        # self.reset_communicator()

    def reset_communicator(self):
        for communicator in self.communicator_list:
            communicator.reset()

    def init_app(self, start_time=0, stop_time=None, phases=1, initial_message='0'):
        # source node src (ps_node_container.Get(src)) will send data to
        # sink node dst (ps_node_container.Get(dst)) with "src+self.base_port" port
        self.start_time, self.stop_time = start_time, stop_time
        ns.core.Simulator.Stop(ns.core.Seconds(self.stop_time))
        for src in range(self.node_num):
            for dst in range(self.node_num):
                if src != dst and self.mix_matrix[src, dst] > 0:

                    src_comm = self.communicator_list[src]
                    src_comm_id = self.communicator_id_list[src]
                    dst_comm = self.communicator_list[dst]
                    dst_comm_id = self.communicator_id_list[dst]

                    if self.verbose:
                        print("Mixing model: PS %s -> PS %s" % (src_comm_id, dst_comm_id))

                    if self.system_count <= 1 or dst_comm.get_ns_node_system_id() == self.system_id:
                        # print(src, dst, self.system_id)
                        dst_comm.add_app_receiver(src_comm, self.model_size, phases,
                                                  src+self.base_port, start_time, stop_time)

                    if self.system_count <= 1 or src_comm.get_ns_node_system_id() == self.system_id:
                        # print(src, dst, self.system_id)
                        src_comm.add_app_sender(dst_comm, self.model_size, phases,
                                                src+self.base_port, self.packet_size, initial_message,
                                                start_time, stop_time)

    def gather_time_consuming_matrix(self, start_of_simulation=0):
        time_rcv_matrix = np.zeros((self.node_num, self.node_num))
        for src in range(self.node_num):
            for dst in range(self.node_num):
                src_comm_id = self.communicator_id_list[src]
                time_rcv_matrix[src, dst] = max(self.communicator_list[dst].get_rcv_time(src_comm_id),
                                                start_of_simulation + self.start_time)

        if self.finished_or_not():
            self.time_consuming_matrix = time_rcv_matrix - start_of_simulation
        else:
            self.time_consuming_matrix = None

        if self.system_count > 1:
            time_consuming_matrix_list = self.mpi_comm.allgather(self.time_consuming_matrix)
            for time_consuming_matrix in time_consuming_matrix_list:
                self.time_consuming_matrix = np.maximum(self.time_consuming_matrix, time_consuming_matrix)

        # # rank 0 collect all the results
        # if self.system_count <= 1:
        #     return
        # if self.system_id == 0:
        #     for i in range(1, self.system_count):
        #         rev_time_arr = self.mpi_comm.recv(source=i, tag=11)
        #         # self.time_consuming_matrix[:, i] = rev_time_arr
        #         self.time_consuming_matrix = np.maximum(self.time_consuming_matrix, rev_time_arr)
        # else:
        #     # self.mpi_comm.send(self.time_consuming_matrix[:, self.system_id], dest=0, tag=11)
        #     self.mpi_comm.send(self.time_consuming_matrix, dest=0, tag=11)

    def finished_or_not(self):
        communicator_states = [communicator.finished_or_not() for communicator in self.communicator_list]
        # if self.system_count > 1:
        #     communicator_states_arr = np.array(self.mpi_comm.allgather(communicator_states))
        #     communicator_states = [np.any(communicator_states_arr[:, i]) for i in range(len(self.communicator_list))]
        return np.all(communicator_states)


if __name__ == '__main__':
    import numpy as np
    np.random.seed(123456)

    model_size = int(1e3)
    phases = 1
    protocol = 'tcp'
    verbose = False
    start_time = 0
    stop_time = 10

    nodes = ns.network.NodeContainer()
    nodes.Create(6)

    nodes01 = ns.network.NodeContainer()
    nodes01.Add(nodes.Get(0))
    nodes01.Add(nodes.Get(1))
    nodes02 = ns.network.NodeContainer()
    nodes02.Add(nodes.Get(0))
    nodes02.Add(nodes.Get(2))
    nodes03 = ns.network.NodeContainer()
    nodes03.Add(nodes.Get(0))
    nodes03.Add(nodes.Get(3))
    nodes04 = ns.network.NodeContainer()
    nodes04.Add(nodes.Get(0))
    nodes04.Add(nodes.Get(4))
    nodes05 = ns.network.NodeContainer()
    nodes05.Add(nodes.Get(0))
    nodes05.Add(nodes.Get(5))

    pointToPoint = ns.point_to_point.PointToPointHelper()
    pointToPoint.SetDeviceAttribute("DataRate", ns.core.StringValue("10Mbps"))
    pointToPoint.SetChannelAttribute("Delay", ns.core.StringValue("2ms"))
    devices01 = pointToPoint.Install(nodes01)
    devices02 = pointToPoint.Install(nodes02)
    devices03 = pointToPoint.Install(nodes03)
    devices04 = pointToPoint.Install(nodes04)
    devices05 = pointToPoint.Install(nodes05)

    stack = ns.internet.InternetStackHelper()
    stack.Install(nodes)

    address = ns.internet.Ipv4AddressHelper()
    address.SetBase(ns.network.Ipv4Address("10.1.1.0"),
                    ns.network.Ipv4Mask("255.255.255.0"))
    interfaces01 = address.Assign(devices01)
    address.SetBase(ns.network.Ipv4Address("10.1.2.0"),
                    ns.network.Ipv4Mask("255.255.255.0"))
    interfaces02 = address.Assign(devices02)
    address.SetBase(ns.network.Ipv4Address("10.1.3.0"),
                    ns.network.Ipv4Mask("255.255.255.0"))
    interfaces03 = address.Assign(devices03)
    address.SetBase(ns.network.Ipv4Address("10.1.4.0"),
                    ns.network.Ipv4Mask("255.255.255.0"))
    interfaces04 = address.Assign(devices04)
    address.SetBase(ns.network.Ipv4Address("10.1.5.0"),
                    ns.network.Ipv4Mask("255.255.255.0"))
    interfaces05 = address.Assign(devices05)

    ns.internet.Ipv4GlobalRoutingHelper.PopulateRoutingTables()

    global_ip_dict = {
        'a': interfaces01.GetAddress(1),
        'b': interfaces02.GetAddress(1),
        'c': interfaces03.GetAddress(1),
        'd': interfaces04.GetAddress(1),
        'e': interfaces05.GetAddress(1),
    }

    comm_a = Communicator(nodes.Get(1), _id='a', global_ip_dict=global_ip_dict, protocol=protocol, verbose=verbose)
    comm_b = Communicator(nodes.Get(2), _id='b', global_ip_dict=global_ip_dict, protocol=protocol, verbose=verbose)
    comm_c = Communicator(nodes.Get(3), _id='c', global_ip_dict=global_ip_dict, protocol=protocol, verbose=verbose)
    comm_d = Communicator(nodes.Get(4), _id='d', global_ip_dict=global_ip_dict, protocol=protocol, verbose=verbose)
    comm_e = Communicator(nodes.Get(5), _id='e', global_ip_dict=global_ip_dict, protocol=protocol, verbose=verbose)
    comm_list = [comm_a, comm_b, comm_c, comm_d, comm_e]

    mix_matrix = [
        [1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1]
    ]

    dc = DecentralizedConsensus(model_size, comm_list, mix_matrix, packet_size=1448, protocol=protocol,
                                verbose=verbose, mpi_comm=None)
    dc.init_app(start_time, stop_time, phases=phases)
    ns.core.Simulator.Run()
    ns.core.Simulator.Destroy()
    dc.gather_time_consuming_matrix()
    # TODO: refine the output
    print(dc.time_consuming_matrix)




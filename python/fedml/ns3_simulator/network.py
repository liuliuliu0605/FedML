from ns import ns
from communicator import Communicator
from decentralized_consensus import DecentralizedConsensus
# from utils import *
from matplotlib import pyplot as plt

import networkx as nx
import sys
import json
import numpy as np
import os
import time

LAN_LATENCY = 5e-7
PACKET_SIZE = 1448
BUFFER_SIZE = 1 << 20  # default sender and receiver buffer size as 1MB

# ns.core.Config.SetDefault ("ns3::TcpL4Protocol::RecoveryType", ns.core.TypeIdValue(ns.internet.TcpPrrRecovery.GetTypeId()))
# ns.core.Config.SetDefault("ns3::TcpL4Protocol::SocketType", ns.core.TypeIdValue(ns.internet.TcpNewReno.GetTypeId()))
ns.core.Config.SetDefault("ns3::TcpSocket::SndBufSize", ns.core.UintegerValue(BUFFER_SIZE))
ns.core.Config.SetDefault("ns3::TcpSocket::RcvBufSize", ns.core.UintegerValue(BUFFER_SIZE))
# ns.core.Config.SetDefault("ns3::TcpSocket::InitialCwnd", ns.core.UintegerValue(10))
# ns.core.Config.SetDefault("ns3::TcpSocket::DelAckCount", ns.core.UintegerValue(1))
# ns.core.Config.SetDefault("ns3::TcpSocket::SegmentSize", ns.core.UintegerValue(PACKET_SIZE))
# ns.core.Config.SetDefault("ns3::TcpSocketBase::Sack", ns.core.BooleanValue(True))


class Network:

    def _init__(self, client_group_list, underlay_name='gaia',
                 node_capacity=1e9, link_capacity=1e9, lan_capacity=1e9,
                 wan_latency='auto', lan_latency=5e-7,
                 model_size=1e3, coord_array=None,
                 history_span=3,
                 parent_dir='ns3_simulator', verbose=False, rs_dir='.', ps_num=9):
        """
        Construct underlay network
        """
        # PSes and clients
        self.client_group_list = client_group_list
        self.ps_num = len(client_group_list)
        self.rs_dir = rs_dir
        self.time_consumed_one_step = 0



        # delay matrix
        self._agg_delay_list = [[] for _ in range(self.ps_num)]
        self._mix_delay_matrix = [[[] for _ in range(self.ps_num)] for _ in range(self.ps_num)]
        self._history_span = history_span

        # get underlay graph and connectivity graph of all nodes
        underlay_dir = os.path.join(parent_dir, 'underlay')

        # add edge weights to connectivity graph
        for u, v, data in self._connectivity_graph.edges(data=True):
            # weight = data["latency"] + model_size / data["availableBandwidth"]
            weight = data["latency"]
            self._connectivity_graph.add_edge(u, v, weight=weight)

        # get connectivity graph of PSes randomly selected from underlay graph

        self._ps_loc_list = list(self._target_connectivity_graph.nodes())

        # initialize nodes in NS3, including PSes and intermediate nodes
        self._nodes, self._ps_clients_nodes_list, self._p2p = self._construct_network(verbose=verbose)
        # allocate node position in NS3
        self._allocate_node_position(coord_array=coord_array, verbose=verbose)

        self.history = {}

        self.model_size = int(model_size)  # Bytes
        #######



    # def __del__(self):
    #     ns.core.Simulator.Destroy()

    def __init__(self, ps_num, node_capacity, link_capacity, protocol='tcp', mpi_comm=None, packet_size=1448, verbose=False):
        self.ps_num = ps_num
        self.node_capacity = int(node_capacity)  # bps
        self.link_capacity = int(link_capacity)  # bps
        self.protocol = protocol
        self.mpi_comm = mpi_comm
        self.system_id = mpi_comm.Get_rank() if mpi_comm is not None else -1
        self.system_count = mpi_comm.Get_size() if mpi_comm is not None else 0
        self.packet_size = packet_size
        self.verbose = verbose

        self.underlay_graph = None
        self.connectivity_graph = None
        self.ps_connectivity_graph = None
        self.ps_overlay_graph = None
        self.ps_id_list = None
        self.global_ip_dict = {}
        self.backbone_router = None
        self.pses = None
        self.communicator_list = []
        self.topology_manager = None

    def read_underlay_graph(self, underlay_name='geantdistance'):
        folder = os.path.join(os.path.dirname(__file__))
        data_path = os.path.join(folder, 'underlay', '%s.gml' % underlay_name)
        self.underlay_graph = nx.read_gml(data_path, label='id')
        for x, y, data in self.underlay_graph.edges(data=True):
            # calculate latency between nodes according to distance
            distance = data['distance']
            latency = (0.0085 * distance + 4) * 1e-3
            self.underlay_graph.add_edge(x, y, latency=latency)

    def generate_connectivity_graph(self):
        self.connectivity_graph = nx.Graph()
        self.connectivity_graph.add_nodes_from(self.underlay_graph.nodes(data=True))
        dijkstra_result = nx.all_pairs_dijkstra(self.underlay_graph.copy(), weight="latency")
        for node, (weights_dict, paths_dict) in dijkstra_result:
            for neighbour in paths_dict.keys():
                if node != neighbour:
                    path = paths_dict[neighbour]
                    latency = 0.
                    for idx in range(len(path) - 1):
                        u = path[idx]
                        v = path[idx + 1]
                        data = self.underlay_graph.get_edge_data(u, v)
                        latency += data['latency']
                    available_bandwidth = self.link_capacity / (len(path) - 1)
                    if node in self.connectivity_graph.nodes() and neighbour in self.connectivity_graph.nodes():
                        self.connectivity_graph.add_edge(node, neighbour, availableBandwidth=available_bandwidth,
                                                    latency=latency, path=path)

    def generate_ps_connectivity_graph(self):
        if self.ps_num >= self.underlay_graph.number_of_nodes():
            self.ps_connectivity_graph = self.connectivity_graph

        start_node = np.random.choice(self.underlay_graph.nodes(), 1).item()
        path = [start_node]
        subset_nodes = set(path)

        # sampling by MHRW
        while len(subset_nodes) < self.ps_num:
            current_node = path[-1]
            neighbours = list(self.underlay_graph[current_node])
            next_node = np.random.choice(neighbours, 1).item()
            q = np.random.random()

            if q <= self.underlay_graph.degree[current_node] / self.underlay_graph.degree[next_node]:
                path.append(next_node)
                subset_nodes.add(next_node)
            else:
                pass

        self.ps_connectivity_graph = self.connectivity_graph.copy()
        for node in self.connectivity_graph.nodes():
            if node not in subset_nodes:
                self.ps_connectivity_graph.remove_node(node)

        self.ps_id_list = list(self.ps_connectivity_graph.nodes())
        if self.verbose:
            print("PS nodes :", self.ps_id_list)

    def plot_underlay_graph(self, figsize=(10, 10), save_path="underlay.pdf"):
        fig, ax = plt.subplots(figsize=figsize)
        graph = self.underlay_graph.copy()
        pos = nx.spring_layout(graph)
        nx.draw_networkx(graph, width=2, alpha=0.8, with_labels=True,
                         style='--', edge_color='g', pos=pos, ax=ax)
        edge_labels = {(u, v): "%d ms" % (d['latency'] * 1000) for u, v, d in graph.edges(data=True)}
        nx.draw_networkx_edge_labels(graph, edge_labels=edge_labels, pos=pos, ax=ax)
        _, save_formate = os.path.splitext(save_path)
        save_formate = 'pdf' if save_formate == '' else save_formate[1:]
        plt.savefig(save_path, format=save_formate, dpi=100, bbox_inches='tight', pad_inches=-0.03)
        print("Underlay info: ", graph)

    def plot_ps_overlay_topology(self, figsize=(10, 10), save_path="topology.pdf"):
        fig, ax = plt.subplots(figsize=figsize)
        graph = self.ps_overlay_graph.copy()

        pos = nx.spring_layout(graph)
        nx.draw_networkx(graph, width=2, alpha=0.8, with_labels=True, style='--', edge_color='g', pos=pos, ax=ax)

        edge_labels = {(u, v): "%d ms, %d Mbps" % (d['latency'] * 1000, d['availableBandwidth'] / 1e6)
                       for u, v, d in graph.edges(data=True)}  # ms, Mbps
        link_delay_list = [d['latency'] * 1000 for _, _, d in graph.edges(data=True)]
        nx.draw_networkx_edge_labels(graph, edge_labels=edge_labels, pos=pos, ax=ax)
        _, save_formate = os.path.splitext(save_path)
        save_formate = 'pdf' if save_formate == '' else save_formate[1:]
        plt.savefig(save_path, format=save_formate, dpi=100, bbox_inches='tight', pad_inches=-0.03)
        print("PS overlay info: ", graph)
        print("Link delay: max=%dms, min=%dms, avg=%dms" % 
              (max(link_delay_list), min(link_delay_list), np.mean(link_delay_list)))

    def plot_ps_connectivity_graph(self, figsize=(10, 10), save_path="ps_connectivity.pdf"):
        fig, ax = plt.subplots(figsize=figsize)
        graph = self.ps_connectivity_graph.copy()

        pos = nx.spring_layout(graph)
        nx.draw_networkx(graph, width=2, alpha=0.8, with_labels=True, style='--', edge_color='g', pos=pos, ax=ax)

        edge_labels = {(u, v): "%d ms, %d Mbps" % (d['latency'] * 1000, d['availableBandwidth'] / 1e6)
                       for u, v, d in graph.edges(data=True)}  # ms, Mbps
        nx.draw_networkx_edge_labels(graph, edge_labels=edge_labels, pos=pos, ax=ax)
        _, save_formate = os.path.splitext(save_path)
        save_formate = 'pdf' if save_formate == '' else save_formate[1:]
        plt.savefig(save_path, format=save_formate, dpi=100, bbox_inches='tight', pad_inches=-0.03)
        print("PS connectivity info: ", graph)

    def connect_pses(self, topology_manager, enable_optimization=True):
        self.topology_manager = topology_manager
        # np.random.shuffle(self.ps_id_list)
        rows, cols = [], []
        for i in range(self.ps_num):
            for j in range(i + 1, self.ps_num):
                if topology_manager.topology[i, j] > 0:
                    rows.append(i)
                    cols.append(j)

        if enable_optimization:

            def get_choice(current_maximum, v, candidates, edges, ps_label_list, topology_manager):
                current_choice = None
                for k in candidates:

                    maximum_1 = max([edges[ps_label_list[v], ps_label_list[i]]['latency']
                                     for i in topology_manager.get_out_neighbor_idx_list(k) if i != v])
                    if maximum_1 >= current_maximum:
                        continue
                    maximum_2 = max([edges[ps_label_list[k], ps_label_list[i]]['latency']
                                     for i in topology_manager.get_out_neighbor_idx_list(v) if i != k])
                    if maximum_2 >= current_maximum:
                        continue
                    current_maximum = max(maximum_1, maximum_2)
                    current_choice = (v, k)
                return current_maximum, current_choice

            # count = 0
            while True:
                loc = np.argmax([self.ps_connectivity_graph.edges[self.ps_id_list[i], self.ps_id_list[j]]['latency']
                                 for i, j in zip(rows, cols)])
                r, c = rows[loc], cols[loc]
                maximum = self.ps_connectivity_graph.edges[self.ps_id_list[r], self.ps_id_list[c]]['latency']
                choice = None
                candidates = [i for i in range(self.ps_num) if i != r and i != c]

                # print(count, maximum)
                # count += 1

                maximum_1, choice_1 = get_choice(maximum, r, candidates, self.ps_connectivity_graph.edges, self.ps_id_list, topology_manager)
                if maximum_1 < maximum:
                    choice = choice_1
                    maximum = maximum_1

                maximum_2, choice_2 = get_choice(maximum, c, candidates, self.ps_connectivity_graph.edges, self.ps_id_list, topology_manager)
                if maximum_2 < maximum:
                    choice = choice_2
                    maximum = maximum_2

                if choice is None:
                    break
                else:
                    tmp = self.ps_id_list[choice[0]]
                    self.ps_id_list[choice[0]] = self.ps_id_list[choice[1]]
                    self.ps_id_list[choice[1]] = tmp

        self.ps_overlay_graph = nx.Graph()
        self.ps_overlay_graph.add_nodes_from(self.ps_connectivity_graph.nodes(data=True))

        for r, w in zip(rows, cols):
            source_node = self.ps_id_list[r]
            sink_node = self.ps_id_list[w]
            edge_data = self.ps_connectivity_graph.get_edge_data(source_node, sink_node)
            self.ps_overlay_graph.add_edge(source_node, sink_node,
                                           latency=edge_data['latency'],
                                           availableBandwidth=edge_data['availableBandwidth'])

    def construct_network(self):
        if self.verbose:
            print("Create Nodes.")

        # create routers
        self.backbone_routers = ns.network.NodeContainer()
        router_num = self.underlay_graph.number_of_nodes()
        if self.mpi_comm is not None:
            # rank 0 correspond to router
            # self.backbone_routers.Create(router_num, 0)
            #
            for i in range(router_num):
                router = ns.network.CreateObject("Node")
                router.SetAttribute("SystemId", ns.core.UintegerValue(i % self.system_count))
                self.backbone_routers.Add(router)
        else:
            self.backbone_routers.Create(router_num)

        # create pses
        self.pses = ns.network.NodeContainer()
        for i in range(self.ps_num):
            ps = ns.network.CreateObject("Node")
            if self.mpi_comm is not None:
                ps.SetAttribute("SystemId", ns.core.UintegerValue(i % self.system_count))
            self.pses.Add(ps)

        # Install the L3 internet stack on routers and self.pses.
        if self.verbose:
            print("Install Internet Stack to Nodes.")
        internet = ns.internet.InternetStackHelper()
        internet.Install(self.backbone_routers)
        internet.Install(self.pses)

        # create p2p links between routers according to underlay topology
        if self.verbose:
            print("Create Links Between Routers.")
        p2p = ns.point_to_point.PointToPointHelper()
        p2p.SetDeviceAttribute("DataRate", ns.core.StringValue("{:f}bps".format(self.link_capacity)))
        # p2p.SetQueue("ns3::DropTailQueue", "MaxSize", ns.network.QueueSizeValue(ns.network.QueueSize("10p")))
        p2p.SetQueue("ns3::DropTailQueue")
        ipv4_n = ns.internet.Ipv4AddressHelper()
        ipv4_n.SetBase(ns.network.Ipv4Address("76.1.1.0"), ns.network.Ipv4Mask("255.255.255.0"))
        link_count = 0
        for i, j, data in self.underlay_graph.edges(data=True):
            n_links = ns.network.NodeContainer()
            n_links.Add(self.backbone_routers.Get(i))
            n_links.Add(self.backbone_routers.Get(j))
            link_delay = ns.core.StringValue("{:f}s".format(data['latency']))
            p2p.SetChannelAttribute("Delay", link_delay)
            n_devs = p2p.Install(n_links)
            ipv4_n.Assign(n_devs)
            ipv4_n.NewNetwork()
            link_count += 1
            if self.verbose:
                print("router [", i, "][", j, "] is physically connected")
        if self.verbose:
            print("Number of physical links is: ", link_count)
            print("Number of all routers is: ", self.backbone_routers.GetN())

        # link PS to router
        self.global_ip_dict = {}
        p2p.SetDeviceAttribute("DataRate", ns.core.StringValue("{:f}bps".format(self.node_capacity)))
        # p2p.SetChannelAttribute("Delay", ns.core.TimeValue(ns.core.Seconds(self.lan_latency)))
        for i, _id in enumerate(self.ps_id_list):
            ps_router = ns.network.NodeContainer()
            ps_router.Add(self.pses.Get(i))
            ps_router.Add(self.backbone_routers.Get(_id))
            ps_router_dev = p2p.Install(ps_router)
            ipv4_n.SetBase(ns.network.Ipv4Address("172.18.%d.0" % (i + 1)), ns.network.Ipv4Mask("255.255.255.0"))
            interfaces = ipv4_n.Assign(ps_router_dev)
            self.global_ip_dict[_id] = interfaces.GetAddress(0)

        if self.verbose:
            print("Initialize Global Routing.")
        ns.internet.Ipv4GlobalRoutingHelper.PopulateRoutingTables()

    def add_communicator(self):
        self.communicator_list = []
        for i in range(self.ps_num):
            comm = Communicator(self.pses.Get(i), _id=self.ps_id_list[i],
                                global_ip_dict=self.global_ip_dict,
                                protocol=self.protocol,
                                verbose=self.verbose)
            self.communicator_list.append(comm)

    # def test(self, times=1, start_time=0, stop_time=None, protocol='tcp', port=9, mix_start_time_list=None):
    def test(self, model_size, start_time=0, stop_time=10000000, phases=1):

        dc = DecentralizedConsensus(model_size, self.communicator_list, self.topology_manager.topology,
                                    packet_size=self.packet_size, protocol=self.protocol,
                                    verbose=self.verbose, mpi_comm=self.mpi_comm)

        dc.run(start_time, stop_time, phases=phases)
        # print(self.system_id, dc.time_consuming_matrix)
        dc.gather_time_consuming_matrix()
        return dc.time_consuming_matrix
        # print(self.system_id, dc.time_consuming_matrix)
        # print(self.system_id, np.max(dc.time_consuming_matrix))


    ####################################################################################################
    def prepare(self, topo_name='ring', params={}, verbose=False):
        self._target_overlay_graph = self._topo_formation(topo_name=topo_name, params=params)

        if verbose:
            print("Preparing overly network of %s..." % topo_name)
            self.print_overlay_info()

    def get_PS_label(self, index):
        return self._ps_loc_list[index]

    def update_agg_delay(self, agg_delay_list):
        for ps_index in range(self.ps_num):
            delay = agg_delay_list[ps_index]
            history_list = self._agg_delay_list[ps_index]
            assert len(history_list) <= self._history_span
            if len(history_list) == self._history_span:
                del history_list[0]
            history_list.append(delay)

    def print_agg_delay_list_from_history(self, precision=3):
        print("-Avg agg delay for Clusters-")
        formate_str = "{:>5.%df}" % precision
        for i, item in enumerate(self._agg_delay_list):
            print("Cluster %d:" % i, formate_str.format(np.mean(item)))

    def update_mix_delay(self, mix_delay_matrix):
        for src_ps_index in range(self.ps_num):
            for dst_ps_index in range(self.ps_num):
                if mix_delay_matrix[src_ps_index, dst_ps_index] > 0 and src_ps_index != dst_ps_index:
                    # note that history_list is a list, which will be altered in place
                    history_list = self._mix_delay_matrix[src_ps_index][dst_ps_index]
                    assert len(history_list) <= self._history_span
                    if len(history_list) == self._history_span:
                        del history_list[0]
                    delay = mix_delay_matrix[src_ps_index, dst_ps_index]
                    history_list.append(delay)

    def print_mix_delay_matrix_from_history(self, precision=3):
        print("-Avg mix delay matrix-")
        formate_str = "{:>10.%df}" % precision
        for src_ps_index in range(self.ps_num):
            for dst_ps_index in range(self.ps_num):
                item = self._mix_delay_matrix[src_ps_index][dst_ps_index]
                if len(item) > 0:
                    print(formate_str.format(np.mean(item)), end=' ')
                else:
                    print(formate_str.format(0.), end=' ')
            print("")

    def get_mix_delay(self, topo=None):
        topo = topo if topo is not None else self.get_ps_matrix()
        # print(topo)
        mix_delay = 0
        for i in range(len(topo)):
            for j in range(len(topo[i])):
                if topo[i,j] > 0 and i != j:
                    mix_delay = max(mix_delay, np.mean(self._mix_delay_matrix[i][j]))
        return mix_delay

    def get_agg_delay(self):
        agg_delay = 0
        for i in range(self.ps_num):
            agg_delay = max(np.mean(self._agg_delay_list[i]), agg_delay)
        return agg_delay

    def get_delay_matrix(self, tau=1):
        rs = np.zeros((self.ps_num, self.ps_num))
        for src_ps_index in range(self.ps_num):
            for dst_ps_index in range(self.ps_num):
                # time cost for PS i complete aggregation and transfer models to its neighbors
                avg_agg_delay = np.mean(self._agg_delay_list[src_ps_index]) * tau
                mix_delay_history = self._mix_delay_matrix[src_ps_index][dst_ps_index]
                avg_mix_delay = np.mean(mix_delay_history) if len(mix_delay_history) > 0 else 0
                rs[src_ps_index, dst_ps_index] = avg_agg_delay + avg_mix_delay
        return rs

    def print_delay_matrix_from_history(self, tau=1, precision=3):
        print("-Avg delay matrix-")
        formate_str = "{:>10.%df}" % precision
        rs = self.get_delay_matrix(tau)
        for src_ps_index in range(self.ps_num):
            for dst_ps_index in range(self.ps_num):
                item = rs[src_ps_index][dst_ps_index]
                print(formate_str.format(np.mean(item)), end=' ')
            print("")

    def get_ps_matrix(self, target_overlay_graph=None, weight=None):
        if target_overlay_graph is not None:
            return nx.adjacency_matrix(target_overlay_graph, weight=weight).toarray()
        else:
            # print(self._target_overlay_graph.nodes())
            return nx.adjacency_matrix(self._target_overlay_graph, weight=weight).toarray()

    def get_ps_wmatrix(self, target_overlay_graph=None, weight=None):
        if target_overlay_graph is not None:
            topo = nx.adjacency_matrix(target_overlay_graph, weight=weight).toarray()
        else:
            # print(self._target_overlay_graph.nodes())
            topo = nx.adjacency_matrix(self._target_overlay_graph, weight=weight).toarray()
        wmatrix = optimal_mixing_weight(topo)
        return wmatrix


    def add_connection(self, node_a, node_b, mix_weight=0):
        source_node = self._ps_loc_list[node_a]
        sink_node = self._ps_loc_list[node_b]
        self._target_overlay_graph.add_edge(source_node, sink_node,
                                            mixWeight=mix_weight,
                                            latency=self._connectivity_graph.get_edge_data(source_node, sink_node)['latency'],
                                            availableBandwidth=self._connectivity_graph.get_edge_data(source_node, sink_node)['availableBandwidth'])

    def delete_connection(self, node_a, node_b):
        self._target_overlay_graph.remove_edge(self._ps_loc_list[node_a], self._ps_loc_list[node_b])

    def print_overlay_info(self):
        latency_list = []
        for source_node, sink_node in self._target_overlay_graph.edges():
            if source_node != sink_node:
                latency = self._target_overlay_graph.get_edge_data(source_node, sink_node)['latency'] * 1000
                latency_list.append(latency)
        print("Latency: max=%dms, min=%dms, total=%dms, avg=%dms" % (max(latency_list), min(latency_list),
                                                                     sum(latency_list),
                                                                     sum(latency_list) / len(latency_list)))
        W = self.get_ps_matrix(weight='mixWeight')
        W = W - np.ones((self.ps_num, self.ps_num)) / self.ps_num
        eigen, _ = np.linalg.eig(np.matmul(W, W.T))
        p = 1 - np.max(eigen)
        # p = 1 - np.sqrt(np.max(eigen))
        print("Mixing matrix: p =", p)

    def _allocate_node_position(self, coord_array=None, verbose=False):
        """Randomly allocate node positions or according to corrd_array"""

        if verbose:
            print("Allocate Positions to Nodes.")

        if coord_array is None:
            if verbose:
                print("Allocate Positions to Nodes.")
            mobility_n = ns.mobility.MobilityHelper()
            mobility_n.SetPositionAllocator("ns3::RandomDiscPositionAllocator",
                                            "X", ns.core.StringValue("100.0"),
                                            "Y", ns.core.StringValue("100.0"),
                                            "Rho", ns.core.StringValue("ns3::UniformRandomVariable[Min=0|Max=30]"))
            mobility_n.SetMobilityModel("ns3::ConstantPositionMobilityModel")
            mobility_n.Install(self._nodes)
        else:
            mobility_n = ns.mobility.MobilityHelper()
            positionAlloc_n = ns.mobility.ListPositionAllocator()

            for m in range(len(coord_array)):  # TODO, location related with delay ?
                positionAlloc_n.Add(ns.core.Vector(coord_array[m][0], coord_array[m][1], 0))
                n0 = self._nodes.Get(m)
                nLoc = n0.GetObject(ns.mobility.MobilityModel.GetTypeId())
                if nLoc is None:
                    nLoc = ns.mobility.ConstantPositionMobilityModel()
                    n0.AggregateObject(nLoc)
                # y-coordinates are negated for correct display in NetAnim
                # NetAnim's (0,0) reference coordinates are located on upper left corner
                # by negating the y coordinates, we declare the reference (0,0) coordinate
                # to the bottom left corner
                nVec = ns.core.Vector(coord_array[m][0], -coord_array[m][1], 0)
                nLoc.SetPosition(nVec)

            mobility_n.SetPositionAllocator(positionAlloc_n)
            mobility_n.Install(self._nodes)

        if verbose:
            for i in range(self._nodes.GetN()):
                position = self._nodes.Get(i).GetObject(ns.mobility.MobilityModel.GetTypeId())
                pos = position.GetPosition()
                print("Node %d: x=%d, y=%d" % (i, pos.x, pos.y))

    def _get_ps_nodes(self):
        return [self._ps_clients_nodes_list[i].Get(0) for i in range(self.ps_num)]
        # return [self._nodes.Get(sn) for sn in self._ps_loc_list]

    def _get_ps_node_by_sn(self, sn):
        return self._ps_clients_nodes_list[self._ps_loc_list.index(sn)].Get(0)

    def fl_step_bak1(self, start_time=0, stop_time=None, protocol='tcp', verbose=False):
        if verbose:
            print("Start simulating one round of FL.")

        assert protocol in ['tcp', 'udp']
        self.time_consumed_one_step = start_time

        def rcv_packet(socket):
            src = ns.network.Address()
            while True:
                packet = socket.RecvFrom(1024, 0, src)
                if packet is None or packet.GetSize() <= 0:
                    break
                rcv_time = ns.core.Simulator.Now().GetSeconds()
                if rcv_time > self.time_consumed_one_step:
                    self.time_consumed_one_step = rcv_time
                if verbose and ns.network.InetSocketAddress.IsMatchingType(src):
                    address = ns.network.InetSocketAddress.ConvertFrom(src)
                    print("At time %.6f packet sink received %d bytes from %s port %d" %
                          (rcv_time, packet.GetSize(), address.GetIpv4(), address.GetPort()))

        def accept_callback(a, b):
            return True

        def new_connection(socket, address):
            socket.SetRecvCallback(rcv_packet)

        def normal_close(socket):
            print("normal close")

        def error_close(socket):
            print("error close")

        port = 99
        for node in self._get_ps_nodes():
            sink_socket = ns.network.Socket.CreateSocket(node, ns.core.TypeId.LookupByName("ns3::{:s}SocketFactory".
                format(
                protocol.capitalize())))
            if protocol == 'tcp':
                sink_socket.SetAcceptCallback(accept_callback, new_connection)
                # sink_socket.SetCloseCallbacks(normal_close, error_close)
            else:
                sink_socket.SetRecvCallback(rcv_packet)
            socket_address = ns.network.InetSocketAddress(ns.network.Ipv4Address.GetAny(), port)
            sink_socket.Bind(socket_address)
            sink_socket.Listen()

        # PSes aggregate models
        for i in range(self.ps_num):

            ps_node = self._ps_clients_nodes_list[i].Get(0)
            ipv4 = ps_node.GetObject(ns.internet.Ipv4.GetTypeId())
            ipv4_int_addr = ipv4.GetAddress(1, 0)
            ip_addr = ipv4_int_addr.GetLocal()

            for j in range(1, self._ps_clients_nodes_list[i].GetN()):

                client_node = self._ps_clients_nodes_list[i].Get(j)

                sender = ns.applications.BulkSendHelper("ns3::{:s}SocketFactory".format(protocol.capitalize()),
                                                        ns.network.Address(
                                                            ns.network.InetSocketAddress(ip_addr, port)))
                sender.SetAttribute("MaxBytes", ns.core.UintegerValue(self.model_size))
                apps_sender = sender.Install(client_node)
                apps_sender.Start(ns.core.Seconds(start_time))
                if stop_time is not None:
                    apps_sender.Stop(ns.core.Seconds(stop_time))

        ns.core.Simulator.Run()
        # ns.core.Simulator.Destroy()

        return self.time_consumed_one_step - start_time

    def impose_router_traffic(self, app_packet_rate=5e5, packet_size=50, port=9999, start_time=0, stop_time=None,
                              protocol='tcp', verbose=False):
        if verbose:
            ns.core.LogComponentEnable("PacketSink", ns.core.LOG_LEVEL_INFO)
            ns.core.LogComponentEnable("OnOffApplication", ns.core.LOG_LEVEL_INFO)

        AppPacketRate = "{:f}bps".format(app_packet_rate)

        for i in range(self._nodes.GetN()):
            sink = ns.applications.PacketSinkHelper("ns3::{:s}SocketFactory".format(protocol.capitalize()),
                                                    ns.network.InetSocketAddress(ns.network.Ipv4Address.GetAny(), port))
            app_sink = sink.Install(self._nodes.Get(i))
            app_sink.Start(ns.core.Seconds(start_time))
            if stop_time is not None:
                app_sink.Stop(ns.core.Seconds(stop_time))

        for i in range(self._nodes.GetN()):
            for j in range(self._nodes.GetN()):
                n = self._nodes.Get(j)
                ipv4 = n.GetObject(ns.internet.Ipv4.GetTypeId())
                ipv4_int_addr = ipv4.GetAddress(1, 0)
                ip_addr = ipv4_int_addr.GetLocal()
                # traffic flows from node[i] to node[j]
                onoff = ns.applications.OnOffHelper("ns3::{:s}SocketFactory".format(protocol.capitalize()),
                                                    ns.network.Address(ns.network.InetSocketAddress(ip_addr, port)))
                onoff.SetConstantRate(ns.network.DataRate(AppPacketRate))
                onoff.SetAttribute("DataRate", ns.core.StringValue("{:f}bps".format(app_packet_rate)))
                onoff.SetAttribute("PacketSize", ns.core.UintegerValue(packet_size))
                onoff.SetAttribute("OnTime", ns.core.StringValue("ns3::ConstantRandomVariable[Constant=1]"))
                onoff.SetAttribute("OffTime", ns.core.StringValue("ns3::ConstantRandomVariable[Constant=0]"))

                x = ns.core.UniformRandomVariable()
                x.SetAttribute("Min", ns.core.DoubleValue(0))
                x.SetAttribute("Max", ns.core.DoubleValue(1))
                rn = x.GetValue()

                app_source = onoff.Install(self._nodes.Get(i))
                app_source.Start(ns.core.Seconds(start_time + rn))
                if stop_time is not None:
                    app_source.Stop(ns.core.Seconds(stop_time))

    def fl_step(self, start_time=0, stop_time=None, protocol='tcp', port=29, agg_start_time_list=None, verbose=False,
                offline_params={}, max_block_duration=None, computation_params=None):
        if agg_start_time_list is None:
            agg_start_time_list = [0] * self.ps_num
        else:
            assert len(agg_start_time_list) == self.ps_num

        time_list = []
        for c in range(self.ps_num):
            cluster_node_container = self._ps_clients_nodes_list[c]  # the first is PS, others are clients

            # PSes distribute models to clients
            star_matrix = np.zeros((cluster_node_container.GetN(), cluster_node_container.GetN()))
            star_matrix[0, :] = 1
            star_matrix[0, 0] = 0
            cluster_app = DecentralizedConsensus(self.model_size, cluster_node_container, star_matrix,
                                                 data_rate=self.lan_capacity, protocol=protocol, port=port,
                                                 packet_size=PACKET_SIZE, verbose=verbose,
                                                 offline_params=offline_params,
                                                 max_block_duration=max_block_duration)
            start_time = agg_start_time_list[c]
            # print(start_time)
            cluster_time_list = cluster_app.run(start_time, stop_time, reduce=None)
            # print(cluster_time_list)

            # # clients conduct local updates
            if computation_params is not None:
                cpu_freq_arr = computation_params['min_cpu_freq'] + (
                        computation_params['max_cpu_freq'] - computation_params['min_cpu_freq']) * np.random.random(
                    cluster_node_container.GetN() - 1)
                computation_time_arr = computation_params['cycles_per_iteration'] * computation_params[
                    'num_of_iterations'] / cpu_freq_arr
                cluster_time_list[1:] += computation_time_arr
            # print(time_consuming)

            # clients return models to PSes
            star_matrix = np.zeros((cluster_node_container.GetN(), cluster_node_container.GetN()))
            star_matrix[:, 0] = 1
            star_matrix[0, 0] = 0
            cluster_app = DecentralizedConsensus(self.model_size, cluster_node_container, star_matrix,
                                                 data_rate=self.lan_capacity, protocol=protocol, port=port,
                                                 packet_size=PACKET_SIZE, verbose=verbose,
                                                 offline_params=offline_params,
                                                 max_block_duration=max_block_duration)
            # clients have different time for local updates and server with i=0 do not send anything
            sender_waiting_time_dict = {i: cluster_time_list[i] for i in range(len(cluster_time_list))}
            # print(sender_waiting_time_dict)
            cluster_time_list = cluster_app.run(0, stop_time, reduce=None,
                                                sender_waiting_time_dict=sender_waiting_time_dict)
            time_list.append(cluster_time_list.max())

        self.update_agg_delay([current-last for last, current in zip(agg_start_time_list, time_list)])
        self.history['fl_step'] = np.mean([current-last for last, current in zip(agg_start_time_list, time_list)])
        return time_list

    def pfl_step(self, times=1, start_time=0, stop_time=None, protocol='tcp', port=9, mix_start_time_list=None, verbose=False,
                 offline_params={}, max_block_duration=None, save_path=None):
        if mix_start_time_list is None:
            mix_start_time_list = [0] * self.ps_num
        else:
            assert len(mix_start_time_list) == self.ps_num

        ps_node_container = ns.network.NodeContainer()
        for i in range(self.ps_num):
            ps_node_container.Add(self._ps_clients_nodes_list[i].Get(0))
        # the upload capacity will vary 10% at most
        upload_capacity = self.upload_capacity + (-1)**(np.random.randint(0, 2)) * self.upload_capacity * 0.1 * np.random.random()
        app = DecentralizedConsensus(self.model_size, ps_node_container, self.get_ps_matrix(),
                                     data_rate=upload_capacity,
                                     protocol=protocol, port=port,
                                     packet_size=PACKET_SIZE, verbose=verbose, offline_params=offline_params,
                                     max_block_duration=max_block_duration)
        mix_start_time_dict = {i: mix_start_time_list[i] for i in range(len(mix_start_time_list))}
        time_list = app.run(start_time, stop_time, times, reduce=None,
                            sender_waiting_time_dict=mix_start_time_dict,
                            log_time_consuming_matrix=True)
        time_matrix = app.get_time_consuming_matrix()

        # record the dynamic topology since any node may be offline
        if save_path is not None:
            with open(save_path, 'w') as f:
                f.write(json.dumps(app.get_comm_matrix_list(), indent=4))

        for ps_index in range(self.ps_num):
            time_matrix[ps_index, :] -= mix_start_time_list[ps_index]

        self.update_mix_delay(time_matrix)
        # print(self._mix_delay_matrix)
        self.history['pfl_step'] = np.mean([current-last-start_time for last, current in zip(mix_start_time_list, time_list)])
        return time_list

    def dfl_step(self, times=1, start_time=0, stop_time=None, protocol='tcp', port=9, verbose=False,
                 offline_params={}, max_block_duration=None, save_path=None, topo_matrix=None, computation_params=None):
        client_node_container = ns.network.NodeContainer()
        for i in range(self.ps_num):
            for j in range(1, self._ps_clients_nodes_list[i].GetN()):
                client_node_container.Add(self._ps_clients_nodes_list[i].Get(j))

        # print(client_node_container.GetN(), self.get_ps_matrix().shape)
        # the upload capacity will variance 10% at most
        # upload_capacity = self.upload_capacity + (-1)**(np.random.randint(0, 2)) * self.upload_capacity * 0.1 * np.random.random()
        # # topo_matrix = np.zeros_like(topo_matrix)
        # for i in range(100):
        #     for j in range(100):
        #         if 0 <= i <= 19 and 0 <= j <= 19 or \
        #                 20 <= i <= 39 and 20 <= j <= 39 or \
        #                 40 <= i <= 59 and 40 <= j <= 59 or \
        #                 60 <= i <= 79 and 60 <= j <= 79 or \
        #                 80 <= i <= 99 and 80 <= j <= 99:
        #             pass
        #         elif i == 0 and j == 99 or i == 99 and j == 0 or \
        #                 i == 19 and j == 20 or i == 20 and j ==19:
        #             pass
        #         else:
        #             if topo_matrix[i, j] > 0 and i != j:
        #                 print(i,j)
        #                 topo_matrix[i, j] = 0

        # clients conduct local updates
        user_num = sum([len(clients) for clients in self.client_group_list])
        time_consuming = np.zeros(user_num)
        if computation_params is not None:
            cpu_freq_arr = computation_params['min_cpu_freq'] + (
                    computation_params['max_cpu_freq'] - computation_params['min_cpu_freq']) * np.random.random(user_num)
            computation_time_arr = computation_params['cycles_per_iteration'] * computation_params[
                'num_of_iterations'] / cpu_freq_arr
            time_consuming += computation_time_arr

        app = DecentralizedConsensus(self.model_size, client_node_container, topo_matrix,
                                     data_rate=self.lan_capacity,
                                     protocol=protocol, port=port,
                                     packet_size=PACKET_SIZE, verbose=verbose, offline_params=offline_params,
                                     max_block_duration=max_block_duration)
        sender_waiting_time_dict = {i: time_consuming[i] for i in range(len(time_consuming)) if i >= 1}
        time_consuming = app.run(start_time, stop_time, times, reduce=None, sender_waiting_time_dict=sender_waiting_time_dict)
        if save_path is not None:
            with open(save_path, 'w') as f:
                f.write(json.dumps(app.get_comm_matrix_list(), indent=4))

        self.history['dfl_step'] = time_consuming.max()
        return time_consuming.max()

    def hier_favg(self, start_time=0, stop_time=None, protocol='tcp', port=9, mix_start_time_list=None, verbose=False,
                  offline_params={}, max_block_duration=None, save_path=None):
        if mix_start_time_list is None:
            mix_start_time_list = [0] * self.ps_num
        else:
            assert len(mix_start_time_list) == self.ps_num

        ps_node_container = ns.network.NodeContainer()
        for i in range(self.ps_num):
            ps_node_container.Add(self._ps_clients_nodes_list[i].Get(0))

        target_overlay_graph = self._topo_formation(topo_name='star')

        # central node receive all models and return them
        time_consuming_list = []
        for k in range(2):
            matrix = self.get_ps_matrix(target_overlay_graph)
            for i in range(len(matrix)):
                if k == 0:
                    if np.alltrue(matrix[i]):
                        for j in range(len(matrix[i])):
                            matrix[i, j] = 0
                        matrix[i, i] = 1
                        break
                else:
                    if np.alltrue(matrix[:, i]):
                        for j in range(len(matrix[:, i])):
                            matrix[j, i] = 0
                        matrix[i, i] = 1
                        break
            app = DecentralizedConsensus(self.model_size, ps_node_container, matrix,
                                         data_rate=self.upload_capacity, protocol=protocol, port=port,
                                         packet_size=PACKET_SIZE, verbose=verbose, offline_params=offline_params,
                                         max_block_duration=max_block_duration)
            if k == 0:
                mix_start_time_dict = {i: mix_start_time_list[i] for i in range(len(mix_start_time_list))}
            else:
                mix_start_time_dict = {i: time_list[i] for i in range(len(time_list))}
            time_list = app.run(start_time, stop_time, reduce=None, sender_waiting_time_dict=mix_start_time_dict)
            for index in mix_start_time_dict:
                time_list[index] = max(time_list[index], mix_start_time_dict[index])

            # time_consuming_list.append(app.run(start_time, stop_time, 1))

        if save_path is not None:
            with open(save_path, 'w') as f:
                f.write(json.dumps(app.get_comm_matrix_list(), indent=4))

        self.history['hier_favg'] = max([current-last-start_time for last, current in zip(mix_start_time_list, time_list)])
        return time_list

    def ring_based_all_reduced(self, start_time=0, stop_time=None, protocol='tcp', port=9, verbose=False,
                               synchronous=False, offline_params={}, save_path=None):
        ps_node_container = ns.network.NodeContainer()
        for i in range(self.ps_num):
            ps_node_container.Add(self._ps_clients_nodes_list[i].Get(0))
        target_overlay_graph = self._topo_formation(topo_name='ring')
        matrix = self.get_ps_matrix(target_overlay_graph)

        # eliminate one circle
        ring_list = [0]
        while len(ring_list) < self.ps_num:
            neighs = list(np.where(matrix[ring_list[-1]] > 0)[0])
            for neigh in neighs:
                if neigh not in ring_list:
                    matrix[ring_list[-1], neigh] = 0
                    ring_list.append(neigh)
                    break
        matrix[ring_list[-1], ring_list[0]] = 0

        model_size_per_phase = int(self.model_size / self.ps_num)
        if synchronous:
            phases = 1
        else:
            phases = 2 * (self.ps_num - 1)
        app = DecentralizedConsensus(model_size_per_phase, ps_node_container, matrix,
                                     data_rate=self.upload_capacity, protocol=protocol, port=port,
                                     packet_size=PACKET_SIZE, verbose=verbose, offline_params=offline_params)
        time_consuming = app.run(start_time, stop_time, phases)
        time_consuming = time_consuming * 2 * (self.ps_num - 1) if synchronous else time_consuming

        if save_path is not None:
            with open(save_path, 'w') as f:
                f.write(json.dumps(app.get_comm_matrix_list(), indent=4))

        self.history['ring_based_all_reduced'] = time_consuming
        return time_consuming

    def plot_dis_bandwidth_parwise(self, figsize=(10, 8)):
        bandwidth_list = []
        for x, y, data in self._connectivity_graph.edges(data=True):
            bandwidth_list.append(data['availableBandwidth'])
        bandwidth_list.sort(reverse=True)
        fig, ax = plt.subplots(figsize=figsize)
        ax.bar(range(1, len(bandwidth_list) + 1), bandwidth_list)
        plt.show()

    def plot_flow_stat(self, figsize=(10, 8)):
        self._monitor.CheckForLostPackets()
        classifier = self._flowmon_helper.GetClassifier()

        for flow_id, flow_stats in self._monitor.GetFlowStats():
            t = classifier.FindFlow(flow_id)
            proto = {6: 'TCP', 17: 'UDP'}[t.protocol]
            print("FlowID: %i (%s %s/%s --> %s/%i)" % \
                  (flow_id, proto, t.sourceAddress, t.sourcePort, t.destinationAddress, t.destinationPort))
            self._print_stats(sys.stdout, flow_stats)

        # delays = []
        # for flow_id, flow_stats in self._monitor.GetFlowStats():
        #     tupl = classifier.FindFlow(flow_id)
        #     if tupl.protocol == 17 and tupl.sourcePort == 698:
        #         continue
        #     delays.append(flow_stats.delaySum.GetSeconds() / flow_stats.rxPackets)
        # plt.hist(delays, 20)
        # plt.xlabel("Delay (s)")
        # plt.ylabel("Number of Flows")
        # plt.show()

    def _print_stats(self, os, st):
        print("  Tx Bytes: ", st.txBytes, file=os)
        print("  Rx Bytes: ", st.rxBytes, file=os)
        print("  Tx Packets: ", st.txPackets, file=os)
        print("  Rx Packets: ", st.rxPackets, file=os)
        print("  Lost Packets: ", st.lostPackets, file=os)
        if st.rxPackets > 0:
            print("  Mean{Delay}: ", (st.delaySum.GetSeconds() / st.rxPackets), file=os)
            # print("  Mean{Jitter}: ", (st.jitterSum.GetSeconds() / (st.rxPackets - 1)), file=os)
            print("  Mean{Hop Count}: ", float(st.timesForwarded) / st.rxPackets + 1, file=os)

        for a, b in enumerate(st.bytesDropped):
            print("--------------")
            print(a, b)

        for reason, drops in enumerate(st.packetsDropped):
            print("  Packets dropped by reason %i: %i" % (reason, drops), file=os)

        # print("Delay Histogram", file=os)
        # for i in range(st.delayHistogram.GetNBins()):
        #     print(" ", i, "(", st.delayHistogram.GetBinStart(i), "-", \
        #           st.delayHistogram.GetBinEnd(i), "): ", st.delayHistogram.GetBinCount(i), file=os)
        # print("Jitter Histogram", file=os)
        # for i in range(st.jitterHistogram.GetNBins()):
        #     print(" ", i, "(", st.jitterHistogram.GetBinStart(i), "-", \
        #           st.jitterHistogram.GetBinEnd(i), "): ", st.jitterHistogram.GetBinCount(i), file=os)
        # print("PacketSize Histogram", file=os)
        # for i in range(st.packetSizeHistogram.GetNBins()):
        #     print(" ", i, "(", st.packetSizeHistogram.GetBinStart(i), "-", \
        #           st.packetSizeHistogram.GetBinEnd(i), "): ", st.packetSizeHistogram)


if __name__ == "__main__":
    np.random.seed(123456)
    ps_num = 2
    model_size = int(1e9)
    network = Network(ps_num=ps_num, node_capacity=1e9, link_capacity=1e10, verbose=False)
    network.read_underlay_graph(underlay_name='geantdistance')
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
    time_consuming_matrix = network.test(model_size=model_size,start_time=0, stop_time=10000000, phases=1)
    t_b = time.time()
    print("Simulation time used: %.3f seconds" % (t_b - t_a))
    print("%.2f MB, time consuming: %.5f seconds" % (model_size/1e6, np.max(time_consuming_matrix)))
    print("-"*50)



from ns import ns
from .communicator import Communicator
from .decentralized_consensus import DecentralizedConsensus
from matplotlib import pyplot as plt
from . import cppdef

import networkx as nx
import sys
import json
import numpy as np
import os
import time
import itertools

# LAN_LATENCY = 5e-7
# PACKET_SIZE = 1448
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

    def __init__(self, access_link_capacity, core_link_capacity, lan_capacity,
                 protocol='tcp', mpi_comm=None, packet_size=1448, verbose=False, seed=None):
        self.access_link_capacity = int(access_link_capacity)  # bps
        self.core_link_capacity = int(core_link_capacity)  # bps
        self.lan_capacity = int(lan_capacity)    # bps
        self.protocol = protocol
        self.mpi_comm = mpi_comm
        self.system_id = mpi_comm.Get_rank() if mpi_comm is not None else 0
        self.system_count = mpi_comm.Get_size() if mpi_comm is not None else 1
        self.packet_size = packet_size
        self.verbose = verbose

        self.edge_ps_num = None
        self.client_num_list = []
        self.underlay_graph = None
        self.connectivity_graph = None
        self.ps_connectivity_graph = None
        self.ps_overlay_graph = None
        self.underlay_node_pos = {}
        self.cloud_id = None
        self.edge_ps_id_list = []
        self.client_id_list = []

        self.ps_ps_ip_dict = {}
        self.ps_client_ip_dict = {}
        self.backbone_routers = None
        # self.access_routers = None
        self.pses = None
        self.clients_list = []
        self.topology_manager = None
        self.topology_map = []

        self.system_id_map = {}
        self.time_history = {}
        np.random.seed(seed)

    def __del__(self):
        pass
        # ns.core.Simulator.Destroy()

    def read_underlay_graph(self, underlay_name='geantdistance'):
        folder = os.path.join(os.path.dirname(__file__))
        data_path = os.path.join(folder, 'underlay', '%s.gml' % underlay_name)
        self.underlay_graph = nx.read_gml(data_path, label='id')
        for x, y, data in self.underlay_graph.edges(data=True):
            # calculate latency between nodes according to distance
            distance = data['distance']
            latency = (0.0085 * distance + 4) * 1e-3
            self.underlay_graph.add_edge(x, y, latency=latency)
        # TODO: decide the display position
        self.underlay_node_pos = nx.spring_layout(self.underlay_graph)
        # analyze the connectivity between any two nodes
        self._analyze_underlay()

    def _analyze_underlay(self):
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
                    available_bandwidth = self.core_link_capacity / (len(path) - 1)
                    if node in self.connectivity_graph.nodes() and neighbour in self.connectivity_graph.nodes():
                        self.connectivity_graph.add_edge(node, neighbour, availableBandwidth=available_bandwidth,
                                                    latency=latency, path=path)

    def _adjust_edge_communicators(self, edge_communicator_list, index_list=None):
        index_list = index_list if index_list is not None else list(range(self.edge_ps_num))

        for i in range(self.edge_ps_num):
            loc = index_list[i]
            loc2 = self._map_ps_overlay_loc(i)
            if loc != loc2:
                j = index_list.index(loc2)
                tmp = edge_communicator_list[j], index_list[j]
                edge_communicator_list[j], index_list[j] = edge_communicator_list[i], index_list[i]
                edge_communicator_list[i], index_list[i] = tmp

    def _map_ps_overlay_loc(self, seq):
        return self.topology_map[seq]

    def select_edge_pses(self, ps_num, method='mhrw'):
        self.edge_ps_num = ps_num
        if self.edge_ps_num >= self.underlay_graph.number_of_nodes():
            self.ps_connectivity_graph = self.connectivity_graph
            self.edge_ps_num = self.underlay_graph.number_of_nodes()
        self.topology_map = [i for i in range(self.edge_ps_num)]

        if method == 'mhrw':

            start_node = np.random.choice(self.underlay_graph.nodes(), 1).item()
            path = [start_node]
            subset_nodes = set(path)

            # sampling by MHRW
            while len(subset_nodes) < self.edge_ps_num:
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

            self.edge_ps_id_list = list(self.ps_connectivity_graph.nodes())

        else:
            raise NotImplementedError

        if self.verbose:
            print("PS nodes :", self.edge_ps_id_list)

    def select_cloud_ps(self, method='centroid'):
        id_list = [_id for _id in list(self.connectivity_graph.nodes())]

        if method == 'centroid':

            current_latency = sys.maxsize
            for c_id in id_list:
                latency = max([self.connectivity_graph.edges[c_id, _id]['latency']
                               for _id in self.edge_ps_id_list if _id != c_id])
                if latency < current_latency:
                    current_latency = latency
                    self.cloud_id = c_id
        elif method == 'random':

            self.cloud_id = int(np.random.choice(id_list))
        if self.verbose:
            print("Cloud ps is located at %d" % self.cloud_id)

    def select_clients(self, overlay_client_num_list, method='near_edge_ps'):
        if method == 'random':
            # TODO
            id_list = np.random.choice([_id for _id in list(self.connectivity_graph.nodes())], len(overlay_client_num_list))

        elif method == 'near_edge_ps':
            if self.edge_ps_id_list is None:
                raise Exception("Please select edge pses first!")
            assert len(overlay_client_num_list) == len(self.edge_ps_id_list)
            self.client_id_list = self.edge_ps_id_list.copy()
            self.client_num_list = []
            for i in range(self.edge_ps_num):
                # map client num to the right position ???
                self.client_num_list.append(overlay_client_num_list[self._map_ps_overlay_loc(i)])

    def partition_underlay_graph(self, graph_partition_method, community_id_list=[]):
        comm_num = len(community_id_list)
        comm_id_dict = {}
        if graph_partition_method == 'girvan_newman':
            comp = nx.community.girvan_newman(self.underlay_graph)
            comm_set = ()
            for communities in itertools.islice(comp, comm_num-1):
                comm_set = tuple(sorted(c) for c in communities)
            for i, comm in enumerate(comm_set):
                for node_id in comm:
                    comm_id_dict[node_id] = community_id_list[i]
        elif graph_partition_method == 'random':
            for node_id in range(self.underlay_graph.number_of_nodes()):
                comm_id_dict[node_id] = community_id_list[node_id % comm_num]
        elif graph_partition_method == 'none':
            for node_id in range(self.underlay_graph.number_of_nodes()):
                comm_id_dict[node_id] = 0
        return comm_id_dict

    def construct_network(self, graph_partition_method='girvan_newman', system_id_list=None):
        if self.verbose:
            print("Create Nodes.")

        if self.system_count > 1:
            # decide how to partition nodes when mpi is enabled
            community_id_list = system_id_list if system_id_list is not None else list(range(self.system_count))
            self.system_id_map = self.partition_underlay_graph(graph_partition_method, community_id_list)

        # create backbone routers
        self.backbone_routers = ns.network.NodeContainer()
        router_num = self.underlay_graph.number_of_nodes()
        if self.system_count > 1:
            # self.backbone_routers.Create(router_num, 0)
            for i in range(router_num):
                router = ns.network.CreateObject("Node")
                # router.SetAttribute("SystemId", ns.core.UintegerValue(0))
                # router.SetAttribute("SystemId", ns.core.UintegerValue(i % self.system_count))
                router.SetAttribute("SystemId", ns.core.UintegerValue(self.system_id_map[i]))
                self.backbone_routers.Add(router)
        else:
            self.backbone_routers.Create(router_num)

        # # create access routers
        # self.access_routers = ns.network.NodeContainer()
        # access_router_ids = self.edge_ps_id_list + self.client_id_list
        # if self.cloud_id is not None:
        #     access_router_ids += [self.cloud_id]
        # access_router_id_list = list(set(access_router_ids))
        # router_num = len(access_router_id_list)
        # if self.system_count > 1:
        #     for i in range(router_num):
        #         router = ns.network.CreateObject("Node")
        #         # router.SetAttribute("SystemId", ns.core.UintegerValue(i % self.system_count))
        #         router.SetAttribute("SystemId", ns.core.UintegerValue(self.system_id_map[access_router_id_list[i]]))
        #         self.access_routers.Add(router)
        # else:
        #     self.access_routers.Create(router_num)

        # create pses
        self.pses = ns.network.NodeContainer()
        for i in range(self.edge_ps_num):
            ps = ns.network.CreateObject("Node")
            if self.system_count > 1:
                # index = access_router_id_list.index(self.edge_ps_id_list[i])
                # system_id = self.access_routers.Get(index).GetSystemId()
                # ps.SetAttribute("SystemId", ns.core.UintegerValue(system_id))
                ps.SetAttribute("SystemId", ns.core.UintegerValue(self.system_id_map[self.edge_ps_id_list[i]]))
            self.pses.Add(ps)
        if self.cloud_id is not None:
            cloud_ps = ns.network.CreateObject("Node")
            if self.system_count > 1:
                cloud_ps.SetAttribute("SystemId", ns.core.UintegerValue(self.system_id_map[self.cloud_id]))
            self.pses.Add(cloud_ps)

        # create clients
        self.clients_list = []
        for i in range(len(self.client_num_list)):
            clients = ns.network.NodeContainer()
            if self.system_count > 1:
                # index = access_router_id_list.index(self.client_id_list[i])
                # system_id = self.access_routers.Get(index).GetSystemId()
                for _ in range(self.client_num_list[i]):
                    client = ns.network.CreateObject("Node")
                    # client.SetAttribute("SystemId", ns.core.UintegerValue(system_id))
                    client.SetAttribute("SystemId", ns.core.UintegerValue(self.system_id_map[self.client_id_list[i]]))
                    clients.Add(client)
            else:
                clients.Create(self.client_num_list[i])
            self.clients_list.append(clients)

        # Install the L3 internet stack on routers and self.pses.
        if self.verbose:
            print("Install Internet Stack to Nodes.")
        internet = ns.internet.InternetStackHelper()
        internet.Install(self.backbone_routers)
        # internet.Install(self.access_routers)
        internet.Install(self.pses)
        for clients in self.clients_list:
            internet.Install(clients)

        # create backbone routers
        if self.verbose:
            print("Create Links Between Routers.")
        p2p = ns.point_to_point.PointToPointHelper()
        p2p.SetDeviceAttribute("DataRate", ns.core.StringValue("{:f}bps".format(self.core_link_capacity)))
        # p2p.SetQueue("ns3::DropTailQueue", "MaxSize", ns.network.QueueSizeValue(ns.network.QueueSize("10p")))
        # p2p.SetQueue("ns3::DropTailQueue")
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
                print("router [%d][%d] is physically connected" % (i,j))
        if self.verbose:
            print("Number of physical links is: %d" % link_count)
            print("Number of all routers is: %d" % self.backbone_routers.GetN())

        # link ps to backbone router
        self.ps_ps_ip_dict = {}
        p2p.SetDeviceAttribute("DataRate", ns.core.StringValue("{:f}bps".format(self.access_link_capacity)))
        p2p.SetChannelAttribute("Delay", ns.core.TimeValue(ns.core.Seconds(1e-6)))
        ipv4_n.SetBase(ns.network.Ipv4Address("172.18.0.0"), ns.network.Ipv4Mask("255.255.255.0"))
        for i, _id in enumerate(self.edge_ps_id_list):
            ps_backbone = ns.network.NodeContainer()
            ps_backbone.Add(self.pses.Get(i))
            ps_backbone.Add(self.backbone_routers.Get(_id))
            devices = p2p.Install(ps_backbone)
            interfaces = ipv4_n.Assign(devices)
            ipv4_n.NewNetwork()
            self.ps_ps_ip_dict["edge_ps %s" % _id] = interfaces.GetAddress(0)
        if self.cloud_id is not None:
            ps_backbone = ns.network.NodeContainer()
            ps_backbone.Add(self.pses.Get(self.edge_ps_num))
            ps_backbone.Add(self.backbone_routers.Get(self.cloud_id))
            devices = p2p.Install(ps_backbone)
            interfaces = ipv4_n.Assign(devices)
            self.ps_ps_ip_dict["cloud_ps %s" % self.cloud_id] = interfaces.GetAddress(0)

        # link ps, clients to backbone via csma
        self.ps_client_ip_dict = {}
        csma = ns.csma.CsmaHelper()
        csma.SetChannelAttribute("DataRate", ns.core.StringValue("{:f}bps".format(self.lan_capacity)))
        csma.SetChannelAttribute("Delay", ns.core.TimeValue(ns.core.Seconds(5e-6)))  # ~5us
        ipv4_n.SetBase(ns.network.Ipv4Address("10.0.0.0"), ns.network.Ipv4Mask("255.255.255.0"))
        for i, _id in enumerate(self.client_id_list):
            lan_nodes = ns.network.NodeContainer()

            lan_nodes.Add(self.clients_list[i])
            communicator_id_list = ["client %s-%s" % (_id, j) for j in range(self.clients_list[i].GetN())]

            lan_nodes.Add(self.pses.Get(i))
            communicator_id_list.append("edge_ps %s" % _id)

            lan_nodes.Add(self.backbone_routers.Get(_id))
            devices = csma.Install(lan_nodes)
            interfaces = ipv4_n.Assign(devices)
            ipv4_n.NewNetwork()

            for j in range(len(communicator_id_list)):
                self.ps_client_ip_dict[communicator_id_list[j]] = interfaces.GetAddress(j)

        # # link access router to backbone router
        # p2p.SetDeviceAttribute("DataRate", ns.core.StringValue("{:f}bps".format(self.access_link_capacity)))
        # p2p.SetChannelAttribute("Delay", ns.core.TimeValue(ns.core.Seconds(1e-6))) # ~1us
        # for i, _id in enumerate(access_router_id_list):
        #     access_backbone_routers = ns.network.NodeContainer()
        #     access_backbone_routers.Add(self.backbone_routers.Get(_id))
        #     access_backbone_routers.Add(self.access_routers.Get(i))
        #     access_backbone_dev = p2p.Install(access_backbone_routers)
        #     ipv4_n.Assign(access_backbone_dev)
        #     ipv4_n.NewNetwork()
        #
        # # link access router, PS and clients via csma
        # self.global_ip_dict = {}
        # csma = ns.csma.CsmaHelper()
        # csma.SetChannelAttribute("DataRate", ns.core.StringValue("{:f}bps".format(self.lan_capacity)))
        # csma.SetChannelAttribute("Delay", ns.core.TimeValue(ns.core.Seconds(1e-6)))  # ~1ms
        # ipv4_n.SetBase(ns.network.Ipv4Address("10.0.0.0"), ns.network.Ipv4Mask("255.255.255.0"))
        # for i, _id in enumerate(access_router_id_list):
        #     communicator_id_list = []
        #     local_area_nodes = ns.network.NodeContainer()
        #
        #     # access router
        #     local_area_nodes.Add(self.access_routers.Get(i))
        #     communicator_id_list.append("access_router %s" % _id)
        #
        #     # edge ps
        #     if _id in self.edge_ps_id_list:
        #         index = self.edge_ps_id_list.index(_id)
        #         edge_ps = self.pses.Get(index)
        #         local_area_nodes.Add(edge_ps)
        #         communicator_id_list.append("edge_ps %s" % _id)
        #     # cloud ps
        #     if _id == self.cloud_id:
        #         local_area_nodes.Add(self.pses.Get(self.edge_ps_num))
        #         communicator_id_list.append("cloud_ps %s" % _id)
        #     # clients
        #     if _id in self.client_id_list:
        #         index = self.client_id_list.index(_id)
        #         clients = self.clients_list[index]
        #         local_area_nodes.Add(clients)
        #         for j in range(clients.GetN()):
        #             communicator_id_list.append("client %s-%s" % (_id, j))
        #
        #     lan_devices = csma.Install(local_area_nodes)
        #     interfaces = ipv4_n.Assign(lan_devices)
        #     ipv4_n.NewNetwork()
        #     # ipv4_n.SetBase(ns.network.Ipv4Address("76.250.%d.0" % (i + 1)), ns.network.Ipv4Mask("255.255.255.0"))
        #     # interfaces = ipv4_n.Assign(local_dev)
        #     for j in range(lan_devices.GetN()):
        #         if communicator_id_list[j] in self.global_ip_dict:
        #             Warning("Duplicate communicator id!")
        #         self.global_ip_dict[communicator_id_list[j]] = interfaces.GetAddress(j)

        if self.verbose:
            print("Initialize Global Routing.")
        ns.internet.Ipv4GlobalRoutingHelper.PopulateRoutingTables()

    def plot_underlay_graph(self, figsize=(10, 10), save_path="underlay.pdf", virtual_pos=False):
        fig, ax = plt.subplots(figsize=figsize)
        graph = self.underlay_graph.copy()
        pos = nx.spring_layout(graph) if virtual_pos else self.underlay_node_pos
        nx.draw_networkx(graph, width=2, alpha=0.8, with_labels=True,
                         style='--', edge_color='g', pos=pos, ax=ax)
        edge_labels = {(u, v): "%d ms" % (d['latency'] * 1000) for u, v, d in graph.edges(data=True)}
        nx.draw_networkx_edge_labels(graph, edge_labels=edge_labels, pos=pos, ax=ax)
        _, save_formate = os.path.splitext(save_path)
        save_formate = 'pdf' if save_formate == '' else save_formate[1:]
        plt.savefig(save_path, format=save_formate, dpi=100, bbox_inches='tight', pad_inches=-0.03)
        if self.verbose:
            print("Underlay info: ", graph)

    def plot_ps_overlay_topology(self, figsize=(10, 10), save_path="topology.pdf", virtual_pos=True):
        fig, ax = plt.subplots(figsize=figsize)
        graph = self.ps_overlay_graph.copy()
        pos = nx.spring_layout(graph) if virtual_pos else self.underlay_node_pos
        nx.draw_networkx(graph, width=2, alpha=0.8, with_labels=True, style='--', edge_color='g', pos=pos, ax=ax)

        edge_labels = {(u, v): "%d ms, %d Mbps" % (d['latency'] * 1000, d['availableBandwidth'] / 1e6)
                       for u, v, d in graph.edges(data=True)}  # ms, Mbps
        link_delay_list = [d['latency'] * 1000 for _, _, d in graph.edges(data=True)]
        nx.draw_networkx_edge_labels(graph, edge_labels=edge_labels, pos=pos, ax=ax)
        _, save_formate = os.path.splitext(save_path)
        save_formate = 'pdf' if save_formate == '' else save_formate[1:]
        plt.savefig(save_path, format=save_formate, dpi=100, bbox_inches='tight', pad_inches=-0.03)
        if self.verbose:
            print("PS overlay info: ", graph)
            print("Link delay: max=%dms, min=%dms, avg=%dms" %
                  (max(link_delay_list), min(link_delay_list), np.mean(link_delay_list)))

    def plot_ps_connectivity_graph(self, figsize=(10, 10), save_path="ps_connectivity.pdf", virtual_pos=True):
        fig, ax = plt.subplots(figsize=figsize)
        graph = self.ps_connectivity_graph.copy()
        pos = nx.spring_layout(graph) if virtual_pos else self.underlay_node_pos
        nx.draw_networkx(graph, width=2, alpha=0.8, with_labels=True, style='--', edge_color='g', pos=pos, ax=ax)

        edge_labels = {(u, v): "%d ms, %d Mbps" % (d['latency'] * 1000, d['availableBandwidth'] / 1e6)
                       for u, v, d in graph.edges(data=True)}  # ms, Mbps
        nx.draw_networkx_edge_labels(graph, edge_labels=edge_labels, pos=pos, ax=ax)
        _, save_formate = os.path.splitext(save_path)
        save_formate = 'pdf' if save_formate == '' else save_formate[1:]
        plt.savefig(save_path, format=save_formate, dpi=100, bbox_inches='tight', pad_inches=-0.03)
        if self.verbose:
            print("PS connectivity info: ", graph)

    def connect_pses(self, topology_manager, enable_optimization=True):
        # TODO: handle hfl case
        if topology_manager is not None:
            self.topology_manager = topology_manager
            rows, cols = [], []
            for i in range(self.edge_ps_num):
                for j in range(i + 1, self.edge_ps_num):
                    if topology_manager.topology[i, j] > 0:
                        rows.append(i)
                        cols.append(j)

            topology_map = [i for i in range(self.edge_ps_num)]
            if enable_optimization:

                def get_choice(current_maximum, v, candidates, edges, ps_label_list, topology_manager, topology_map):
                    current_choice = None
                    for k in candidates:
                        latency_list = [edges[ps_label_list[topology_map[v]], ps_label_list[topology_map[i]]]['latency']
                                        if i != v else edges[ps_label_list[topology_map[v]], ps_label_list[topology_map[k]]]['latency']
                                        for i in topology_manager.get_out_neighbor_idx_list(k)]
                        new_maximum_1 = max(latency_list)
                        if new_maximum_1 >= current_maximum:
                            continue
                        latency_list = [edges[ps_label_list[topology_map[k]], ps_label_list[topology_map[i]]]['latency']
                                        if i != k else edges[ps_label_list[topology_map[v]], ps_label_list[topology_map[k]]]['latency']
                                        for i in topology_manager.get_out_neighbor_idx_list(v)]
                        new_maximum_2 = max(latency_list)
                        if new_maximum_2 >= current_maximum:
                            continue
                        current_maximum = max(new_maximum_1, new_maximum_2)
                        current_choice = (v, k)
                    return current_maximum, current_choice

                # count = 0
                while True:
                    loc = np.argmax([self.ps_connectivity_graph.edges[self.edge_ps_id_list[topology_map[i]], self.edge_ps_id_list[topology_map[j]]]['latency']
                                     for i, j in zip(rows, cols)])
                    r, c = rows[loc], cols[loc]
                    maximum = self.ps_connectivity_graph.edges[self.edge_ps_id_list[topology_map[r]], self.edge_ps_id_list[topology_map[c]]]['latency']
                    choice = None
                    candidates = [i for i in range(self.edge_ps_num) if i != r and i != c]

                    # print(count, maximum)
                    # count += 1

                    maximum_1, choice_1 = get_choice(maximum, r, candidates, self.ps_connectivity_graph.edges, self.edge_ps_id_list, topology_manager, topology_map)
                    if maximum_1 < maximum:
                        choice = choice_1
                        maximum = maximum_1

                    maximum_2, choice_2 = get_choice(maximum, c, candidates, self.ps_connectivity_graph.edges, self.edge_ps_id_list, topology_manager, topology_map)
                    if maximum_2 < maximum:
                        choice = choice_2
                        maximum = maximum_2

                    if choice is None:
                        break
                    else:
                        # tmp = self.edge_ps_id_list[choice[0]]
                        # self.edge_ps_id_list[choice[0]] = self.edge_ps_id_list[choice[1]]
                        # self.edge_ps_id_list[choice[1]] = tmp
                        tmp = topology_map[choice[0]]
                        topology_map[choice[0]] = topology_map[choice[1]]
                        topology_map[choice[1]] = tmp
            self.topology_map = topology_map

            self.ps_overlay_graph = nx.Graph()
            self.ps_overlay_graph.add_nodes_from(self.ps_connectivity_graph.nodes(data=True))

            for r, w in zip(rows, cols):
                source_node = self.edge_ps_id_list[topology_map[r]]
                sink_node = self.edge_ps_id_list[topology_map[w]]
                edge_data = self.ps_connectivity_graph.get_edge_data(source_node, sink_node)
                self.ps_overlay_graph.add_edge(source_node, sink_node,
                                               latency=edge_data['latency'],
                                               availableBandwidth=edge_data['availableBandwidth'])
        else:
            self.ps_overlay_graph = nx.Graph()
            self.ps_overlay_graph.add_nodes_from(self.ps_connectivity_graph.nodes(data=True))

    def generate_ps_clients_communicators(self, i, _id):
        communicator_list = []

        # ps communicator
        ps_comm = Communicator(self.pses.Get(i), _id="edge_ps %s" % _id,
                               global_ip_dict=self.ps_client_ip_dict,
                               protocol=self.protocol,
                               verbose=self.verbose)
        communicator_list.append(ps_comm)

        # client communicators
        for j in range(self.client_num_list[i]):
            client_comm = Communicator(self.clients_list[i].Get(j), _id="client %s-%s" % (_id, j),
                                       global_ip_dict=self.ps_client_ip_dict,
                                       protocol=self.protocol,
                                       verbose=self.verbose)
            communicator_list.append(client_comm)

        return communicator_list

    def generate_ps_ps_communicators(self, include_cloud_ps=False):
        edge_communicator_list = []
        for i in range(self.edge_ps_num):
            edge_comm = Communicator(self.pses.Get(i), _id="edge_ps %s" % self.edge_ps_id_list[i],
                                     global_ip_dict=self.ps_ps_ip_dict,
                                     protocol=self.protocol, verbose=self.verbose)
            edge_communicator_list.append(edge_comm)
            # communicator_list[self._map_ps_overlay_loc(i)] = edge_comm

        # adjust the order of communicators to match topology if mixing
        # topology_map is initialized when selecting edge pses and refined when connecting them via some topology
        self._adjust_edge_communicators(edge_communicator_list)

        if include_cloud_ps:
            cloud_ps = self.pses.Get(self.edge_ps_num)
            cloud_comm = Communicator(cloud_ps, _id="cloud_ps %s" % self.cloud_id,
                                      global_ip_dict=self.ps_ps_ip_dict,
                                      protocol=self.protocol, verbose=self.verbose)
        else:
            cloud_comm = None

        return edge_communicator_list, cloud_comm

    def set_fl_step(self, model_size, start_time=0, stop_time=10000000, phases=1, initial_message='0',
                    local_update_config={}):

        def client_receive_model(agg_comm):
            delay = np.random.uniform(
                local_update_config.get('low', 0),
                local_update_config.get('high', 0),
                1
            ).sum()
            agg_comm.generate_message('hello', delay=delay)

        def ps_receive_model(agg_comm):
            agg_comm.generate_message('hello')

        ps_client_distribution_list = []
        client_ps_aggregation_list = []
        for i, _id in enumerate(self.edge_ps_id_list):
            distribute_communicators = self.generate_ps_clients_communicators(i, _id)
            agg_communicators = self.generate_ps_clients_communicators(i, _id)

            # client receives the model from ps and then return the model to ps
            for j in range(1, self.client_num_list[i] + 1):
                distribute_communicators[j].register_phase_callback(client_receive_model, agg_communicators[j])

            # ps receives the model from clients and then aggregate them for distribution
            agg_communicators[0].register_phase_callback(ps_receive_model, distribute_communicators[0])

            # PS distributes models to clients
            distribute_matrix = np.zeros((self.client_num_list[i] + 1, self.client_num_list[i] + 1))
            distribute_matrix[0, :] = 1
            distribute_matrix[0, 0] = 0
            ps_client_distribution = DecentralizedConsensus(model_size, distribute_communicators, distribute_matrix,
                                               packet_size=self.packet_size, protocol=self.protocol,
                                               verbose=self.verbose, mpi_comm=self.mpi_comm, base_port=5000)
            ps_client_distribution.init_app(start_time, stop_time, phases, initial_message=initial_message)
            ps_client_distribution_list.append(ps_client_distribution)

            # Clients return models to PS
            agg_matrix = np.zeros((self.client_num_list[i] + 1, self.client_num_list[i] + 1))
            agg_matrix[:, 0] = 1
            agg_matrix[0, 0] = 0
            client_ps_aggregation = DecentralizedConsensus(model_size, agg_communicators, agg_matrix,
                                                            packet_size=self.packet_size, protocol=self.protocol,
                                                            verbose=self.verbose, mpi_comm=self.mpi_comm, base_port=5000)
            client_ps_aggregation.init_app(start_time, stop_time, phases, initial_message=None)
            client_ps_aggregation_list.append(client_ps_aggregation)

        self._adjust_edge_communicators(ps_client_distribution_list)
        self._adjust_edge_communicators(client_ps_aggregation_list)

        # start_of_simulation = ns.core.Simulator.Now().GetSeconds()
        # ns.core.Simulator.Run()
        # ns.core.Simulator.Destroy()
        # ps_client_distribution.gather_time_consuming_matrix(start_of_simulation=start_of_simulation)
        # client_ps_aggregation.gather_time_consuming_matrix(start_of_simulation=start_of_simulation)
        # print(ps_client_distribution.time_consuming_matrix)
        # print(client_ps_aggregation.time_consuming_matrix)

        return ps_client_distribution_list, client_ps_aggregation_list

    def set_pfl_step(self, model_size, start_time=0, stop_time=10000000, phases=1, initial_message='0'):
        communicator_list, _ = self.generate_ps_ps_communicators()
        ps_ps_mix = DecentralizedConsensus(model_size, communicator_list, self.topology_manager.topology,
                                           packet_size=self.packet_size, protocol=self.protocol,
                                           verbose=self.verbose, mpi_comm=self.mpi_comm, base_port=6000)
        ps_ps_mix.init_app(start_time, stop_time, phases, initial_message=initial_message)

        return ps_ps_mix

    def set_hfl_step(self, model_size, start_time=0, stop_time=10000000, initial_message='0'):
        agg_edge_communicators, agg_cloud_communicator = self.generate_ps_ps_communicators(include_cloud_ps=True)
        dis_edge_communicators, dis_cloud_communicator = self.generate_ps_ps_communicators(include_cloud_ps=True)

        # edge servers send models to cloud server
        agg_matrix = np.zeros((self.edge_ps_num + 1, self.edge_ps_num + 1))
        agg_matrix[:, 0] = 1
        agg_matrix[0, 0] = 0
        edge_cloud_aggregation = DecentralizedConsensus(
            model_size, [agg_cloud_communicator] + agg_edge_communicators, agg_matrix, packet_size=self.packet_size,
            protocol=self.protocol, verbose=self.verbose, mpi_comm=self.mpi_comm, base_port=5000
        )
        edge_cloud_aggregation.init_app(start_time, stop_time, phases=1, initial_message=initial_message)

        def cloud_receive_model(agg_comm):
            agg_comm.generate_message('hello')

        agg_cloud_communicator.register_phase_callback(cloud_receive_model, dis_cloud_communicator)

        # cloud server distributes models to edge servers
        distribute_matrix = np.zeros((self.edge_ps_num + 1, self.edge_ps_num + 1))
        distribute_matrix[0, :] = 1
        distribute_matrix[0, 0] = 0
        cloud_edge_distribution = DecentralizedConsensus(
            model_size, [dis_cloud_communicator] + dis_edge_communicators, distribute_matrix,
            packet_size=self.packet_size, protocol=self.protocol, verbose=self.verbose,
            mpi_comm=self.mpi_comm, base_port=5000)
        cloud_edge_distribution.init_app(start_time, stop_time, phases=1, initial_message=None)

        return edge_cloud_aggregation, cloud_edge_distribution

    def add_history(self, config_param, data):
        if config_param in self.time_history:
            self.time_history[config_param].append(data)
        else:
            self.time_history[config_param] = [data]

    def get_history(self, config_param):
        return self.time_history[config_param][-1]

    def run_fl_pfl(self, model_size, group_comm_round=1, mix_comm_round=1, local_update_config={},
                   start_time=0, stop_time=10000000):
        ps_client_distribution_list, client_ps_aggregation_list = self.set_fl_step(model_size,
                                                                                   local_update_config=local_update_config,
                                                                                   start_time=start_time,
                                                                                   stop_time=stop_time,
                                                                                   phases=group_comm_round)
        ps_ps_mix = self.set_pfl_step(model_size,
                                      start_time=start_time,
                                      stop_time=stop_time,
                                      phases=mix_comm_round,
                                      initial_message=None)

        def ps_receive_model_from_clients(mix_comm):
            mix_comm.generate_message('hello')

        for i in range(self.edge_ps_num):
            ps_agg_communicator = client_ps_aggregation_list[i].communicator_list[0]  # index 0 refers to ps communicator
            ps_mix_communicator = ps_ps_mix.communicator_list[i]
            ps_agg_communicator.register_finish_callback(ps_receive_model_from_clients, ps_mix_communicator)

        start_of_simulation = ns.core.Simulator.Now().GetSeconds()
        ns.core.Simulator.Run()
        ns.core.Simulator.Destroy()
        for ps_client_distribution, client_ps_aggregation in zip(ps_client_distribution_list, client_ps_aggregation_list):
            ps_client_distribution.gather_time_consuming_matrix(start_of_simulation=start_of_simulation)
            client_ps_aggregation.gather_time_consuming_matrix(start_of_simulation=start_of_simulation)

        ps_ps_mix.gather_time_consuming_matrix(start_of_simulation=start_of_simulation)
        ps_ps_delay_matrix = ps_ps_mix.time_consuming_matrix
        # print(self.system_id, ps_ps_delay_matrix)
        ps_agg_delay = np.array([client_ps_aggregation.time_consuming_matrix[:, 0].max() for client_ps_aggregation in client_ps_aggregation_list])
        ps_mix_delay = np.array([ps_ps_delay_matrix[i, :].max()-ps_agg_delay[i] for i in range(self.edge_ps_num)])
        return ps_ps_delay_matrix, ps_agg_delay, ps_mix_delay

    def run_fl_hfl(self, model_size, group_comm_round=1, local_update_config={}, start_time=0, stop_time=10000000):
        ps_client_distribution_list, client_ps_aggregation_list = self.set_fl_step(model_size,
                                                                                   local_update_config=local_update_config,
                                                                                   start_time=start_time,
                                                                                   stop_time=stop_time,
                                                                                   phases=group_comm_round)
        edge_cloud_aggregation, cloud_edge_distribution = self.set_hfl_step(model_size,
                                                                            start_time=start_time,
                                                                            stop_time=stop_time,
                                                                            initial_message=None)

        # edge server receives models from clients ps, aggregate then and send to cloud ps
        def ps_receive_model_from_clients(mix_comm):
            mix_comm.generate_message('hello')

        for i in range(self.edge_ps_num):
            ps_partial_agg_communicator = client_ps_aggregation_list[i].communicator_list[0] # index 0 refers to ps communicator
            ps_global_agg_communicator = edge_cloud_aggregation.communicator_list[i+1]
            ps_partial_agg_communicator.register_finish_callback(ps_receive_model_from_clients,
                                                                 ps_global_agg_communicator)

        start_of_simulation = ns.core.Simulator.Now().GetSeconds()
        ns.core.Simulator.Run()
        ns.core.Simulator.Destroy()
        for ps_client_distribution, client_ps_aggregation in zip(ps_client_distribution_list,
                                                                 client_ps_aggregation_list):
            ps_client_distribution.gather_time_consuming_matrix(start_of_simulation=start_of_simulation)
            client_ps_aggregation.gather_time_consuming_matrix(start_of_simulation=start_of_simulation)
        cloud_edge_distribution.gather_time_consuming_matrix(start_of_simulation=start_of_simulation)
        ps_ps_delay_matrix = cloud_edge_distribution.time_consuming_matrix
        ps_partial_agg_delay = np.array([client_ps_aggregation.time_consuming_matrix[:, 0].max()
                                         for client_ps_aggregation in client_ps_aggregation_list])
        ps_global_agg_delay = np.array(
            [ps_ps_delay_matrix[:, i+1].max() - ps_partial_agg_delay[i] for i in range(self.edge_ps_num)])
        return ps_ps_delay_matrix, ps_partial_agg_delay, ps_global_agg_delay

    def run_fl_rar(self, model_size, group_comm_round=1, local_update_config={}, start_time=0, stop_time=10000000):
        ps_client_distribution_list, client_ps_aggregation_list = self.set_fl_step(model_size,
                                                                                   local_update_config=local_update_config,
                                                                                   start_time=start_time,
                                                                                   stop_time=stop_time,
                                                                                   phases=group_comm_round)
        ps_ps_mix = self.set_pfl_step(model_size//self.edge_ps_num,
                                      start_time=start_time,
                                      stop_time=stop_time,
                                      phases=2*(self.edge_ps_num -1),
                                      initial_message=None)

        def ps_receive_model_from_clients(mix_comm):
            mix_comm.generate_message('hello')

        for i in range(self.edge_ps_num):
            ps_agg_communicator = client_ps_aggregation_list[i].communicator_list[0]  # index 0 refers to ps communicator
            ps_mix_communicator = ps_ps_mix.communicator_list[i]
            ps_agg_communicator.register_finish_callback(ps_receive_model_from_clients, ps_mix_communicator)

        start_of_simulation = ns.core.Simulator.Now().GetSeconds()
        ns.core.Simulator.Run()
        ns.core.Simulator.Destroy()
        for ps_client_distribution, client_ps_aggregation in zip(ps_client_distribution_list, client_ps_aggregation_list):
            ps_client_distribution.gather_time_consuming_matrix(start_of_simulation=start_of_simulation)
            client_ps_aggregation.gather_time_consuming_matrix(start_of_simulation=start_of_simulation)

        ps_ps_mix.gather_time_consuming_matrix(start_of_simulation=start_of_simulation)
        ps_ps_delay_matrix = ps_ps_mix.time_consuming_matrix
        # print(self.system_id, ps_ps_delay_matrix)
        ps_agg_delay = np.array([client_ps_aggregation.time_consuming_matrix[:, 0].max() for client_ps_aggregation in client_ps_aggregation_list])
        ps_mix_delay = np.array([ps_ps_delay_matrix[:, i].max()-ps_agg_delay[i] for i in range(self.edge_ps_num)])
        return ps_ps_delay_matrix, ps_agg_delay, ps_mix_delay

    ############################################################################################
    def update_agg_delay(self, agg_delay_list):
        for ps_index in range(self.edge_ps_num):
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
        for src_ps_index in range(self.edge_ps_num):
            for dst_ps_index in range(self.edge_ps_num):
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
        for src_ps_index in range(self.edge_ps_num):
            for dst_ps_index in range(self.edge_ps_num):
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
        for i in range(self.edge_ps_num):
            agg_delay = max(np.mean(self._agg_delay_list[i]), agg_delay)
        return agg_delay

    def get_delay_matrix(self, tau=1):
        rs = np.zeros((self.edge_ps_num, self.edge_ps_num))
        for src_ps_index in range(self.edge_ps_num):
            for dst_ps_index in range(self.edge_ps_num):
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
        for src_ps_index in range(self.edge_ps_num):
            for dst_ps_index in range(self.edge_ps_num):
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
        W = W - np.ones((self.edge_ps_num, self.edge_ps_num)) / self.edge_ps_num
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
    enable_mpi = False

    if enable_mpi:
        # initialization if mpi mode
        from mpi4py import MPI
        ns.core.GlobalValue.Bind("SimulatorImplementationType", ns.core.StringValue("ns3::DistributedSimulatorImpl"))
        py_comm = MPI.COMM_WORLD
        comm = ns.cppyy.gbl.Convert2MPIComm(py_comm)
        ns.mpi.MpiInterface.Enable(comm)
    else:
        py_comm = None

    ps_num = 3
    group_comm_round = 1
    mix_comm_round = 1
    clients_num_list = [1 for i in range(ps_num)]
    model_size = int(1e5)

    network = Network(access_link_capacity=1e6,
                      core_link_capacity=1e10,
                      lan_capacity=1e11, verbose=False, mpi_comm=py_comm)

    network.read_underlay_graph(underlay_name='geantdistance')

    network.select_edge_pses(ps_num=ps_num, method='none')
    network.select_cloud_ps(method='centroid')
    client_num_list = [2 for i in range(ps_num)]
    network.select_clients(client_num_list, method='near_edge_ps')

    from fedml.core.distributed.topology.symmetric_topology_manager import SymmetricTopologyManager
    class ARGS:
        topo_name = 'ring'
    topology_manager = SymmetricTopologyManager(ps_num)
    topology_manager.generate_custom_topology(ARGS())
    network.connect_pses(topology_manager, enable_optimization=True)
    network.construct_network(graph_partition_method='girvan_newman')

    # network.plot_underlay_graph(save_path="underlay.png")
    # network.plot_ps_connectivity_graph(save_path="connectivity.png")
    # network.plot_ps_overlay_topology(save_path="overlay.png")

    t_a = time.time()
    # ps_ps_delay_matrix, ps_agg_delay, ps_mix_delay = network.run_fl_pfl(model_size=model_size,
    #                                                                     group_comm_round=group_comm_round,
    #                                                                     mix_comm_round=group_comm_round,
    #                                                                     start_time=0, stop_time=10000000)

    # ps_ps_delay_matrix, ps_agg_delay, ps_mix_delay = network.run_fl_hfl(model_size=model_size,
    #                                                                     group_comm_round=group_comm_round,
    #                                                                     start_time=0, stop_time=10000000)

    ps_ps_delay_matrix, ps_agg_delay, ps_mix_delay = network.run_fl_rar(model_size=model_size,
                                                                        group_comm_round=group_comm_round,
                                                                        start_time=0, stop_time=10000000)

    # network.set_fl_step(model_size, start_time=0, stop_time=10000000, phases=1, initial_message='0')
    # network.set_hfl_step(model_size, start_time=0, stop_time=10000000, initial_message='0')
    # network.set_pfl_step(model_size, start_time=0, stop_time=10000000, phases=1, initial_message='0')
    t_b = time.time()

    time_consuming_matrix = ps_ps_delay_matrix
    if time_consuming_matrix is not None:
        print(ps_agg_delay)
        print(ps_mix_delay)
        print("%.2f MB: total=%.5fs, agg=%.5f, mix=%.5f (%ds)" %
              (model_size/1e6,
               np.max(time_consuming_matrix),
               ps_agg_delay.mean()/group_comm_round,
               ps_mix_delay.mean(),
               t_b-t_a))
    print("-"*50)

    if enable_mpi:
        # destroy mpi if mpi mode
        ns.mpi.MpiInterface.Disable()
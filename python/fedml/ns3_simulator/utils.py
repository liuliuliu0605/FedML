import networkx as nx
from networkx.utils import pairwise
import math
from .tsp_christofides import christofides_tsp
import numpy as np
import itertools
import os
import json
from itertools import combinations, groupby
import random as rd
import numpy as np



def get_neighbour_overlay(connectivity_graph, hops=1):
    topo = nx.Graph()
    topo.add_nodes_from(connectivity_graph.nodes(data=True))

    for source_node, sink_node in connectivity_graph.edges():
        path = connectivity_graph.get_edge_data(source_node, sink_node)['path']
        if len(path) <= hops + 1:
            topo.add_edge(source_node, sink_node,
                          latency=connectivity_graph.get_edge_data(source_node, sink_node)['latency'],
                          availableBandwidth=connectivity_graph.get_edge_data(source_node, sink_node)[
                              'availableBandwidth'],
                          mixWeight=0)

    for source_node, sink_node in topo.edges():
        mix_weight = 1 / max(topo.degree[source_node] + 1, topo.degree[sink_node] + 1)
        topo.edges[source_node, sink_node].update({'mixWeight': mix_weight})

    for node in topo.nodes():
        mix_weight = 1 - sum([data['mixWeight'] for _, _, data in topo.edges(node, data=True)])
        topo.add_edge(node, node, mixWeight=mix_weight)

    # print(nx.adjacency_matrix(topo, weight="mixWeight").toarray())

    return topo


def get_random_overlay(connectivity_graph, probability=0.5, seed=0):
    rd.seed(seed)
    random = nx.Graph()
    random.add_nodes_from(connectivity_graph.nodes(data=True))
    node_num = random.number_of_nodes()
    matrix = np.zeros((node_num, node_num))
    edges = combinations(range(node_num), 2)
    for _, node_edges in groupby(edges, key=lambda x: x[0]):
        node_edges = list(node_edges)
        random_edge = rd.choice(node_edges)
        matrix[random_edge[0], random_edge[1]] = 1
        matrix[random_edge[1], random_edge[0]] = 1
        for e in node_edges:
            if rd.random() <= probability:
                matrix[e[0], e[1]] = 1
                matrix[e[1], e[0]] = 1

    graph_nodes = list(connectivity_graph.nodes())
    for i in range(node_num):
        for j in range(i, node_num):
            if matrix[i, j] > 0:
                random.add_edge(graph_nodes[i], graph_nodes[j],
                                latency=connectivity_graph.get_edge_data(graph_nodes[i], graph_nodes[j])['latency'],
                                availableBandwidth=connectivity_graph.get_edge_data(graph_nodes[i], graph_nodes[j])[
                                    'availableBandwidth'],
                                mixWeight=0)

    for source_node, sink_node in random.edges():
        mix_weight = 1 / max(random.degree[source_node] + 1, random.degree[sink_node] + 1)
        random.edges[source_node, sink_node].update({'mixWeight': mix_weight})

    for node in random.nodes():
        mix_weight = 1 - sum([data['mixWeight'] for _, _, data in random.edges(node, data=True)])
        random.add_edge(node, node, mixWeight=mix_weight)

    return random


def get_ring_overlay(connectivity_graph):
    adjacency_matrix = nx.adjacency_matrix(connectivity_graph, weight="weight").toarray()
    tsp_nodes = christofides_tsp(adjacency_matrix)

    ring = nx.Graph()
    ring.add_nodes_from(connectivity_graph.nodes(data=True))

    # total_latency = 0
    # max_latency = 0

    for idx in range(len(tsp_nodes) - 1):
        # get the label of source and sink nodes from the original graph
        source_node = list(connectivity_graph.nodes())[tsp_nodes[idx]]
        sink_node = list(connectivity_graph.nodes())[tsp_nodes[idx + 1]]

        ring.add_edge(source_node, sink_node,
                      latency=connectivity_graph.get_edge_data(source_node, sink_node)['latency'],
                      availableBandwidth=connectivity_graph.get_edge_data(source_node, sink_node)['availableBandwidth'],
                      mixWeight=1 / 3)
        ring.add_edge(source_node, source_node, mixWeight=1 / 3)

    # add final link to close the circuit
    source_node = list(connectivity_graph.nodes())[tsp_nodes[-1]]
    sink_node = list(connectivity_graph.nodes())[tsp_nodes[0]]
    ring.add_edge(source_node, sink_node,
                  latency=connectivity_graph.get_edge_data(source_node, sink_node)['latency'],
                  availableBandwidth=connectivity_graph.get_edge_data(source_node, sink_node)['availableBandwidth'],
                  mixWeight=1 / 3)
    ring.add_edge(source_node, source_node, mixWeight=1 / 3)

    return ring


def get_regular_graph(connectivity_graph, k=2):
    regular = nx.Graph()
    regular.add_nodes_from(connectivity_graph.nodes(data=True))
    node_num = connectivity_graph.number_of_nodes()
    assert (2 <= k < node_num) and (k * node_num) % 2 == 0
    adjacency_matrix = nx.adjacency_matrix(connectivity_graph, weight="weight").toarray()
    tsp_nodes = christofides_tsp(adjacency_matrix)

    for idx in range(len(tsp_nodes) - 1):
        # get the label of source and sink nodes from the original graph
        source_node = list(connectivity_graph.nodes())[tsp_nodes[idx]]
        sink_node = list(connectivity_graph.nodes())[tsp_nodes[idx + 1]]

        regular.add_edge(source_node, sink_node,
                         latency=connectivity_graph.get_edge_data(source_node, sink_node)['latency'],
                         availableBandwidth=connectivity_graph.get_edge_data(source_node, sink_node)[
                             'availableBandwidth'],
                         mixWeight=1 / (k + 1))

    source_node = list(connectivity_graph.nodes())[tsp_nodes[-1]]
    sink_node = list(connectivity_graph.nodes())[tsp_nodes[0]]
    regular.add_edge(source_node, sink_node,
                     latency=connectivity_graph.get_edge_data(source_node, sink_node)['latency'],
                     availableBandwidth=connectivity_graph.get_edge_data(source_node, sink_node)['availableBandwidth'],
                     mixWeight=1 / (k + 1))

    ring_path = [x for x, y in nx.find_cycle(regular)]
    extra_num = k - 2
    for i in range(len(ring_path)):
        source_node = ring_path[i]
        for j in range(extra_num):
            sink_node = ring_path[(i + j + 1) % len(ring_path)]
            regular.add_edge(source_node, sink_node,
                             latency=connectivity_graph.get_edge_data(source_node, sink_node)['latency'],
                             availableBandwidth=connectivity_graph.get_edge_data(source_node, sink_node)[
                                 'availableBandwidth'],
                             mixWeight=1 / (k + 1))

    for node in connectivity_graph.nodes():
        regular.add_edge(node, node, mixWeight=1 / (k + 1))

    return regular


def get_complete_overlay(connectivity_graph):
    complete = nx.Graph()
    complete.add_nodes_from(connectivity_graph.nodes(data=True))
    dijkstra_result = nx.all_pairs_dijkstra(connectivity_graph, weight="weight")
    node_num = connectivity_graph.number_of_nodes()

    for node, (weights_dict, paths_dict) in dijkstra_result:
        for neighbour in paths_dict.keys():
            if node != neighbour:
                path = paths_dict[neighbour]

                latency = 0.
                available_bandwidth = connectivity_graph.get_edge_data(node, neighbour)['availableBandwidth']
                for idx in range(len(path) - 1):
                    u = path[idx]
                    v = path[idx + 1]

                    data = connectivity_graph.get_edge_data(u, v)
                    latency += data["latency"]
                    available_bandwidth = min(available_bandwidth, data["availableBandwidth"])

                complete.add_edge(node, neighbour, availableBandwidth=available_bandwidth,
                                  latency=latency, mixWeight=1 / node_num)
            else:
                complete.add_edge(node, node, mixWeight=1 / node_num)

    return complete


def get_2d_torus_overlay(connectivity_graph):
    # how to calculate optimal ???
    node_num = connectivity_graph.number_of_nodes()
    side_len = node_num ** 0.5
    assert math.ceil(side_len) == math.floor(side_len)
    side_len = int(side_len)

    torus = nx.Graph()
    torus.add_nodes_from(connectivity_graph.nodes(data=True))
    node_label_list = list(connectivity_graph.nodes())

    def add_edge(i, j):
        if i != j:
            torus.add_edge(node_label_list[i], node_label_list[j],
                           latency=connectivity_graph.get_edge_data(node_label_list[i], node_label_list[j])['latency'],
                           availableBandwidth=connectivity_graph.get_edge_data(node_label_list[i], node_label_list[j])[
                               'availableBandwidth'],
                           mixWeight=1 / 5)
        else:
            torus.add_edge(node_label_list[i], node_label_list[j], mixWeight=1 / 5)

    for i in range(side_len):
        for j in range(side_len):
            idx = i * side_len + j
            add_edge(idx, idx)
            add_edge(idx, (((i + 1) % side_len) * side_len + j))
            add_edge(idx, (((i - 1) % side_len) * side_len + j))
            add_edge(idx, (i * side_len + (j + 1) % side_len))
            add_edge(idx, (i * side_len + (j - 1) % side_len))

    return torus


def get_star_overlay(connectivity_graph, centrality='distance'):
    if centrality == "distance":
        centrality_dict = nx.algorithms.centrality.closeness_centrality(connectivity_graph, distance="latency")
        server_node = max(centrality_dict, key=centrality_dict.get)

    elif centrality == "information":
        centrality_dict = nx.algorithms.centrality.information_centrality(connectivity_graph, weight="latency")
        server_node = max(centrality_dict, key=centrality_dict.get)

    else:
        # centrality = load_centrality
        centrality_dict = nx.algorithms.centrality.load_centrality(connectivity_graph, weight="latency")
        server_node = max(centrality_dict, key=centrality_dict.get)

    weights, paths = nx.single_source_dijkstra(connectivity_graph, source=server_node, weight="weight")

    star = nx.Graph()
    star.add_nodes_from(connectivity_graph.nodes(data=True))
    node_num = connectivity_graph.number_of_nodes()

    for node in paths.keys():
        if node != server_node:
            star.add_edge(server_node, node,
                          latency=connectivity_graph.get_edge_data(server_node, node)['latency'],
                          availableBandwidth=connectivity_graph.get_edge_data(server_node, node)['availableBandwidth'],
                          mixWeight=1 / node_num)
            star.add_edge(node, node, mixWeight=1 - 1 / node_num)
        else:
            star.add_edge(server_node, server_node, mixWeight=1 / connectivity_graph.number_of_nodes())

    return star


def get_ring_overlay2(node_label_list, connectivity_graph):
    """

    :param connectivity_graph:
    :param computation_time:
    :param model_size:
    :return:
    """

    # adjacency_matrix = nx.adjacency_matrix(connectivity_graph, weight="weight").toarray()
    # tsp_nodes = christofides_tsp(adjacency_matrix)
    # node_label_list = list(connectivity_graph.nodes())
    ring = nx.Graph()
    ring.add_nodes_from(connectivity_graph.nodes(data=True))

    for idx in range(len(node_label_list) - 1):
        # get the label of source and sink nodes from the original graph
        source_node = node_label_list[idx]
        sink_node = node_label_list[idx + 1]

        ring.add_edge(source_node, sink_node,
                      latency=connectivity_graph.get_edge_data(source_node, sink_node)['latency'],
                      availableBandwidth=connectivity_graph.get_edge_data(source_node, sink_node)['availableBandwidth'],
                      # weight=connectivity_graph.get_edge_data(source_node, sink_node)['weight'],
                      mixWeight=1 / 3)
        ring.add_edge(source_node, source_node, mixWeight=1 / 3)

    # add final link to close the circuit
    source_node = node_label_list[-1]
    sink_node = node_label_list[0]
    ring.add_edge(source_node, sink_node,
                  latency=connectivity_graph.get_edge_data(source_node, sink_node)['latency'],
                  availableBandwidth=connectivity_graph.get_edge_data(source_node, sink_node)['availableBandwidth'],
                  # weight=connectivity_graph.get_edge_data(source_node, sink_node)['weight'],
                  mixWeight=1 / 3)
    ring.add_edge(source_node, source_node, mixWeight=1 / 3)

    return ring


def get_isolated_overlay(connectivity_graph):
    isolated = nx.Graph()
    isolated.add_nodes_from(connectivity_graph.nodes(data=True))
    node_label_list = connectivity_graph.nodes()
    for node in node_label_list:
        isolated.add_edge(node, node, mixWeight=1)
    return isolated


def get_mst_overlay(connectivity_graph, delta=-1):
    if delta > 0:
        tree = delta_prim(connectivity_graph, delta)
    else:
        tree = nx.minimum_spanning_tree(connectivity_graph, weight="weight")
    for x, y in tree.edges():
        tree.add_edge(x, y, mixWeight=1 / (max(tree.degree[x], tree.degree[y]) + 1))
    for node in tree.nodes():
        mixWeight = 1 - sum([item['mixWeight'] for item in tree[node].values()])
        tree.add_edge(node, node, mixWeight=mixWeight)
    return tree


def get_balanced_tree_overlay(node_label_list, connectivity_graph, degree=2):
    tree = nx.Graph()
    tree.add_nodes_from(connectivity_graph.nodes(data=True))

    # node_label_list = list(connectivity_graph.nodes())

    def add_edge(i, j):
        if i != j:
            tree.add_edge(node_label_list[i], node_label_list[j],
                          latency=connectivity_graph.get_edge_data(node_label_list[i], node_label_list[j])['latency'],
                          availableBandwidth=connectivity_graph.get_edge_data(node_label_list[i], node_label_list[j])[
                              'availableBandwidth'],
                          mixWeight=1 / (degree + 1))
        else:
            tree.add_edge(node_label_list[i], node_label_list[j], mixWeight=1 / (degree + 1))

    for i in range(len(node_label_list)):
        for j in range(1, degree + 1):
            if i * 2 + j >= len(node_label_list):
                break
            add_edge(i, i * 2 + j)

    for node in tree.nodes():
        mixWeight = 1 - sum([item['mixWeight'] for item in tree[node].values()])
        tree.add_edge(node, node, mixWeight=mixWeight)

    return tree


def get_balanced_tree_overlay2(node_label_list, connectivity_graph, degree=2):
    # node_label_list = list(connectivity_graph.nodes())
    optimal_tree = None
    minimal_weights = np.inf

    for root in node_label_list:
        tree = nx.Graph()
        tree.add_nodes_from(connectivity_graph.nodes(data=True))
        q = [root]
        used_nodes = set([root])
        total_weights = 0
        while len(used_nodes) < len(node_label_list):
            parent = q.pop(0)
            candidates = [node for node in connectivity_graph.adj[parent] if node not in used_nodes]
            candidates.sort(key=lambda node: connectivity_graph.edges[(parent, node)]['weight'])
            for i in range(min(degree, len(candidates))):
                total_weights += connectivity_graph.get_edge_data(parent, candidates[i])['weight']
                tree.add_edge(parent, candidates[i],
                              latency=connectivity_graph.get_edge_data(parent, candidates[i])['latency'],
                              availableBandwidth=connectivity_graph.get_edge_data(parent, candidates[i])[
                                  'availableBandwidth'],
                              mixWeight=1 / (degree + 1))
                q.append(candidates[i])
                used_nodes.add(candidates[i])
        if total_weights < minimal_weights:
            minimal_weights = total_weights
            optimal_tree = tree

    for node in optimal_tree.nodes():
        mixWeight = 1 - sum([item['mixWeight'] for item in optimal_tree[node].values()])
        optimal_tree.add_edge(node, node, mixWeight=mixWeight)
    return optimal_tree


def get_barbell_overlay(node_label_list, connectivity_graph, m1=1, m2=0):
    # node_label_list = list(connectivity_graph.nodes())
    assert 2 * m1 + m2 == len(node_label_list)

    barbell = nx.Graph()
    barbell.add_nodes_from(connectivity_graph.nodes(data=True))

    # left barbell
    nx.complete_graph(node_label_list[:m1])
    edges = itertools.combinations(node_label_list[:m1], 2)
    barbell.add_edges_from(edges)
    # connecting path
    if m2 > 1:
        barbell.add_edges_from(pairwise(node_label_list[m1:m1 + m2]))
    # right barbell
    barbell.add_edges_from(
        (node_label_list[u], node_label_list[v]) for u in range(m1 + m2, 2 * m1 + m2) for v in range(u + 1, 2 * m1 + m2)
    )
    # connect it up
    barbell.add_edge(node_label_list[m1 - 1], node_label_list[m1])
    if m2 > 0:
        barbell.add_edge(node_label_list[m1 + m2 - 1], node_label_list[m1 + m2])

    dijkstra_result = nx.all_pairs_dijkstra(connectivity_graph, weight="weight")

    for node, (weights_dict, paths_dict) in dijkstra_result:

        for neighbour in paths_dict.keys():
            if node != neighbour and barbell.has_edge(node, neighbour):
                path = paths_dict[neighbour]

                latency = 0.
                available_bandwidth = 1e32
                for idx in range(len(path) - 1):
                    u = path[idx]
                    v = path[idx + 1]

                    data = connectivity_graph.get_edge_data(u, v)
                    latency += data["latency"]
                    available_bandwidth = min(available_bandwidth, data["availableBandwidth"])

                barbell.add_edge(node, neighbour, availableBandwidth=available_bandwidth, latency=latency,
                                 mixWeight=1 / (max(barbell.degree[node], barbell.degree[neighbour]) + 1))

    for node in node_label_list:
        mixWeight = 1 - sum([item['mixWeight'] for item in barbell[node].values()])
        barbell.add_edge(node, node, mixWeight=mixWeight)

    return barbell


def get_random_overlay2(node_label_list, connectivity_graph, by="optimal"):
    node_num = connectivity_graph.number_of_nodes()

    random = nx.Graph()
    random.add_nodes_from(connectivity_graph.nodes(data=True))

    def add_edge(i, j, w):
        if i != j:
            random.add_edge(node_label_list[i], node_label_list[j],
                            latency=connectivity_graph.get_edge_data(node_label_list[i], node_label_list[j])['latency'],
                            availableBandwidth=connectivity_graph.get_edge_data(node_label_list[i], node_label_list[j])[
                                'availableBandwidth'],
                            mixWeight=w)
        else:
            random.add_edge(node_label_list[i], node_label_list[j], mixWeight=w)

    probability = 0.5
    nodes_path = "./utils/data/overlay/%d_nodes_%.2f_matrix.json" % (node_num, probability)
    if os.path.exists(nodes_path):
        with open(nodes_path) as f:
            matrix = np.array(json.load(f))
    else:
        matrix = (np.random.random((node_num, node_num)) >= probability).astype("d")
        for i in range(node_num):
            matrix[i, i] = 1
        for i in range(node_num):
            for j in range(node_num):
                matrix[i, j] = matrix[j, i]
        with open(nodes_path, 'w') as f:
            json.dump(matrix.tolist(), f)
    print(matrix)

    if by == "optimal":
        w_matrix = sdp_solve(matrix)
    else:
        w_matrix = np.zeros_like(matrix)
        matrix_sum = matrix.sum(1)

    for i in range(node_num):
        for j in range(node_num):
            if i != j and matrix[i, j] > 0:
                if by == "uniform":
                    w_matrix[i, j] = 1 / max(matrix_sum[i], matrix_sum[j])
                    # w_matrix[i, j] = 1 / matrix_sum[i]
                    add_edge(i, j, w_matrix[i, j])
                elif by == "optimal":
                    add_edge(i, j, w_matrix[i, j])
        if by == "uniform":
            w_matrix[i, i] = 1 - w_matrix[i].sum()
            add_edge(i, i, w_matrix[i, i])
        elif by == "optimal":
            add_edge(i, i, w_matrix[i, i])

    if by == 'other':
        random = random.to_directed()
        for i in range(node_num):
            for j in range(node_num):
                if matrix[i, j] > 0:
                    w_matrix[i, j] = 1 / matrix_sum[i]
                    add_edge(i, j, w_matrix[i, j])

    if by == 'random':
        while True:
            for i in range(node_num):
                for j in range(i, node_num):
                    if matrix[i, j] > 0. and i != j:
                        d = 1 - sum(w_matrix[i, :j])
                        w_matrix[i][j] = w_matrix[j][i] = d * np.random.random()
            success = True
            for i in range(node_num):
                w_matrix[i, i] = 1 - sum(w_matrix[i])
                if w_matrix[i, i] < 0:
                    success = False
                    break
            if not success:
                w_matrix = np.zeros_like(matrix)
            else:
                for i in range(node_num):
                    for j in range(i, node_num):
                        if w_matrix[i, j] > 0:
                            add_edge(i, j, w_matrix[i, j])
                break

    # for line in w_matrix:
    #     print('  '.join(map(lambda x: "%.2f"%x, line)))
    # print(np.sum(w_matrix, axis=0))
    # print(np.sum(w_matrix, axis=1))

    return random


def sdp_solve(matrix):
    # with open("/root/models/utils/data/overlay/9_nodes_optimal.json") as f:
    #     w_matrix = json.load(f)
    #     return np.array(w_matrix)

    node_num = matrix.shape[0]

    # Construct the problem.
    lambd = cp.Variable(pos=True, name="lambd")
    W = cp.Variable((node_num, node_num), symmetric=True)
    AVG = np.ones((node_num, node_num)) / node_num
    ONE = np.ones((node_num, 1))
    INDENTITY = np.identity(node_num)

    objective = cp.Minimize(lambd)
    constraints = [W @ ONE == ONE, W - AVG >> INDENTITY * (-lambd), W - AVG << INDENTITY * lambd]
    for i in range(node_num):
        for j in range(node_num):
            if matrix[i, j] == 1.0:
                constraints.append(W[i, j] >= 0)
            else:
                constraints.append(W[i, j] == 0)

    prob = cp.Problem(objective, constraints)

    # The optimal objective value is returned by `prob.solve()`.
    result = prob.solve()
    # The optimal value for x is stored in `x.value`.
    # print(prob.value)
    # The optimal Lagrange multiplier for a constraint is stored in
    # `constraint.dual_value`.
    # print(W.value)
    w_matrix = matrix.copy()
    for i in range(node_num):
        for j in range(node_num):
            # w_matrix[i, j] = max(W.value[i, j], 0)
            w_matrix[i, j] = W.value[i, j]

    for i in range(node_num):
        # x = w_matrix[i,:]/sum(w_matrix[i,:])
        # print(x)
        # w_matrix[i,:] /= sum(w_matrix[i,:])
        # print(w_matrix[i,:])
        pass

    return w_matrix


def delta_prim(G_complete, delta):
    """
    implementation of delta prim algorithm from https://ieeexplore.ieee.org/document/850653
    :param G: (nx.Graph())
    :param delta: (int)
    :return: a tree T with degree at most delta
    """
    N = G_complete.number_of_nodes()
    T = nx.Graph()

    T.add_node(list(G_complete.nodes)[0])

    while len(T.edges) < N - 1:
        smallest_weight = np.inf
        edge_to_add = None
        for u in T.nodes:
            for v in G_complete.nodes:
                if (v not in T.nodes) and (T.degree[u] < delta):
                    weight = G_complete.get_edge_data(u, v)["weight"]
                    if weight < smallest_weight:
                        smallest_weight = weight
                        edge_to_add = (u, v)

        T.add_edge(*edge_to_add, weight=smallest_weight)

    T.add_nodes_from(G_complete.nodes(data=True))

    return T


def optimal_mixing_weight(topo_matrix):
    w_matrix = np.zeros_like(topo_matrix, dtype=float)
    for i in range(0, len(topo_matrix)):
        for j in range(i+1, len(topo_matrix)):
            if topo_matrix[i,j] > 0:
                w_matrix[i, j] = w_matrix[j, i] = 1./(max(topo_matrix[i, :].sum()-1, topo_matrix[j, :].sum()-1)+1)
    for i in range(0, len(w_matrix)):
        w_matrix[i, i] = 1 - w_matrix[i, :].sum()
    return w_matrix


def formate_topo(tau, init_topo_matrix, delay_matrix, lr, K, U, estimate_params):


    def H(topo_matrix):
        d = (delay_matrix * topo_matrix).max()
        weight_matrix = optimal_mixing_weight(topo_matrix)
        spectral_norm = np.linalg.norm(weight_matrix - 1. / len(weight_matrix), ord=2) ** 2
        p = 1 - spectral_norm
        A = d / (U * tau) * estimate_params['loss'] / (lr * K)
        B = 12 * lr ** 2 * K * estimate_params['L'] ** 2 * (1 + 2 / p) * (tau ** 2 / p - tau) * (
                estimate_params['sigma'] + 6 * K * estimate_params['psi'])
        B = estimate_params['control'] * B
        obj = A + B
        print("obj=%.5f, p=%.5f, A=%.5f, B=%.5f" % (obj, p, A, B))
        return obj

    topo_matrix = init_topo_matrix.copy()
    min_obj = H(topo_matrix)
    used = []
    un_used = []
    for i in range(len(init_topo_matrix)):
        for j in range(i+1, len(init_topo_matrix[i])):
            if init_topo_matrix[i,j] > 0:
                used.append((i,j, (delay_matrix[i,j]+delay_matrix[j,i])/2))
            else:
                un_used.append((i, j, (delay_matrix[i, j] + delay_matrix[j, i]) / 2))
    used = sorted(used, key=lambda item:item[-1])
    un_used = sorted(un_used, key=lambda item:item[-1])
    # print(used)
    # print(un_used)
    last_success = True
    current_success = True
    last_action = 0
    current_action = 0
    while True:
        # print(topo_matrix)
        # print(weight_matrix)
        print("-"*10)
        if len(un_used) == 0:
            # should delete an edge
            probability = 0
        elif len(used) == 0:
            # should add an edge
            probability = 1
        else:
            probability = rd.random()
        new_topo_matrix = topo_matrix.copy()
        if probability > 0.5:
            # add
            current_action = 1
            i, j, delay = un_used[0]
            # i, j, delay = un_used.pop(0)
            # used.append((i, j, delay))
            new_topo_matrix[i, j] = new_topo_matrix[j, i] = 1
            print("add:", (i, j, delay))

        else:
            # delete
            current_action = -1
            i, j, delay = used[-1]
            # i, j, delay = used.pop()
            # un_used.append((i, j, delay))
            new_topo_matrix[i, j] = new_topo_matrix[j, i] = 0
            print("delete:", (i, j, delay))

        current_obj = H(new_topo_matrix)
        if current_obj > min_obj:
            current_success = False
        else:
            current_success = True
            if current_action == 1:
                i, j, delay = un_used.pop(0)
                used.append((i, j, delay))
            else:
                i, j, delay = used.pop()
                un_used.append((i, j, delay))
            min_obj = current_obj
            topo_matrix = new_topo_matrix
            used = sorted(used, key=lambda item: item[-1])
            un_used = sorted(un_used, key=lambda item: item[-1])
        print(current_success)

        if current_action * last_action < 0 and not last_success and not current_success or \
               not current_success and probability == 1 or probability == 0:
            break
        else:
            last_action = current_action
            last_success = current_success
    H(topo_matrix)
    print(topo_matrix)


def formate_topo_bak(tau, init_topo_matrix, delay_matrix, lr, K, U, estimate_params):

    def H(topo_matrix):
        d = (delay_matrix * topo_matrix).max()
        weight_matrix = optimal_mixing_weight(topo_matrix)
        spectral_norm = np.linalg.norm(weight_matrix - 1. / len(weight_matrix), ord=2) ** 2
        p = 1 - spectral_norm
        A = d / (U * tau) * estimate_params['loss'] / (lr * K)
        B = 12 * lr ** 2 * K * estimate_params['L'] ** 2 * (1 + 2 / p) * (tau ** 2 / p - tau) * (
                estimate_params['sigma'] + 6 * K * estimate_params['psi'])
        B = estimate_params['control'] * B
        obj = A + B
        print("obj=%.5f, p=%.5f, A=%.5f, B=%.5f" % (obj, p, A, B))
        return obj

    topo_matrix = init_topo_matrix.copy()
    min_obj = H(topo_matrix)
    used = []
    un_used = []
    for i in range(len(init_topo_matrix)):
        for j in range(i+1, len(init_topo_matrix[i])):
            if init_topo_matrix[i,j] > 0:
                used.append((i,j, (delay_matrix[i,j]+delay_matrix[j,i])/2))
            else:
                un_used.append((i, j, (delay_matrix[i, j] + delay_matrix[j, i]) / 2))
    used = sorted(used, key=lambda item:item[-1])
    un_used = sorted(un_used, key=lambda item:item[-1])
    # print(used)
    # print(un_used)
    last_success = True
    current_success = True
    last_action = 0
    current_action = 0
    while True:
        # print(topo_matrix)
        # print(weight_matrix)
        print("-"*10)
        if len(un_used) == 0:
            # should delete an edge
            probability = 0
        elif len(used) == 0:
            # should add an edge
            probability = 1
        else:
            probability = rd.random()
        new_topo_matrix = topo_matrix.copy()
        if probability > 0.5:
            # add
            current_action = 1
            i, j, delay = un_used[0]
            # i, j, delay = un_used.pop(0)
            # used.append((i, j, delay))
            new_topo_matrix[i, j] = new_topo_matrix[j, i] = 1
            print("add:", (i, j, delay))

        else:
            # delete
            current_action = -1
            i, j, delay = used[-1]
            # i, j, delay = used.pop()
            # un_used.append((i, j, delay))
            new_topo_matrix[i, j] = new_topo_matrix[j, i] = 0
            print("delete:", (i, j, delay))

        current_obj = H(new_topo_matrix)
        if current_obj > min_obj:
            current_success = False
        else:
            current_success = True
            if current_action == 1:
                i, j, delay = un_used.pop(0)
                used.append((i, j, delay))
            else:
                i, j, delay = used.pop()
                un_used.append((i, j, delay))
            min_obj = current_obj
            topo_matrix = new_topo_matrix
            used = sorted(used, key=lambda item: item[-1])
            un_used = sorted(un_used, key=lambda item: item[-1])
        print(current_success)

        if current_action * last_action < 0 and not last_success and not current_success or \
               not current_success and probability == 1 or probability == 0:
            break
        else:
            last_action = current_action
            last_success = current_success
    H(topo_matrix)
    print(topo_matrix)

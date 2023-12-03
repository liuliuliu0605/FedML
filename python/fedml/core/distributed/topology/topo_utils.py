import math
import numpy as np
import networkx as nx
import cvxpy as cp


def sdp_solve(matrix):
    node_num = matrix.shape[0]

    # Construct the problem.
    _lambda = cp.Variable(pos=True, name="_lambda")
    W = cp.Variable((node_num, node_num), symmetric=True)
    AVG = np.ones((node_num, node_num)) / node_num
    ONE = np.ones((node_num, 1))
    INDENTITY = np.identity(node_num)

    objective = cp.Minimize(_lambda)
    constraints = [W @ ONE == ONE,
                   W - AVG >> INDENTITY * (-_lambda),
                   W - AVG << INDENTITY * _lambda,
                   _lambda >= 0]
    for i in range(node_num):
        for j in range(node_num):
            if matrix[i, j] > 0:
                constraints.append(W[i, j] >= 0)
                # constraints.append(W[i, j] <= 1)
                # print(i,j)
            else:
                constraints.append(W[i, j] == 0)
    prob = cp.Problem(objective, constraints)

    # The optimal objective value is returned by prob.solve().
    result = prob.solve()
    # The optimal value for x is stored in x.value.
    # print("opt lambda:", prob.value)
    # The optimal Lagrange multiplier for a constraint is stored in
    # constraint.dual_value.
    # print(W.value)
    w_matrix = W.value
    for i in range(node_num):
        for j in range(node_num):
            if matrix[i, j] == 0 or i == j:
                w_matrix[i, j] = 0
    for i in range(node_num):
        w_matrix[i, i] = 1 - w_matrix[i, :].sum()
    return w_matrix


def get_topology(adjacency_matrix, method='optimal'):
    assert np.allclose(np.array(adjacency_matrix), np.array(adjacency_matrix).T)

    if method == 'optimal':
        return sdp_solve(adjacency_matrix)
    else:
        topology = np.zeros_like(adjacency_matrix, dtype=np.float64)
        for i in range(0, len(adjacency_matrix)):
            for j in range(i + 1, len(adjacency_matrix)):
                if adjacency_matrix[i, j] == 1:
                    topology[i, j] = topology[j, i] = \
                        1. / (max(adjacency_matrix[i, :].sum() - 1, adjacency_matrix[j, :].sum() - 1) + 1)
        for i in range(0, len(topology)):
            topology[i, i] = 1 - topology[i, :].sum()
        return topology


def get_2d_torus_overlay(node_num):
    side_len = node_num ** 0.5
    assert math.ceil(side_len) == math.floor(side_len)
    side_len = int(side_len)

    torus = np.zeros((node_num, node_num), dtype=np.float64)

    for i in range(side_len):
        for j in range(side_len):
            idx = i * side_len + j
            torus[idx, idx] = 1 / 5
            torus[idx, (((i + 1) % side_len) * side_len + j)] = 1 / 5
            torus[idx, (((i - 1) % side_len) * side_len + j)] = 1 / 5
            torus[idx, (i * side_len + (j + 1) % side_len)] = 1 / 5
            torus[idx, (i * side_len + (j - 1) % side_len)] = 1 / 5

    return torus


def get_star_overlay(node_num):

    star = np.zeros((node_num, node_num), dtype=np.float64)
    for i in range(node_num):
        if i == 0:
            star[i, i] = 1 / node_num
        else:
            star[0, i] = star[i, 0] = 1 / node_num
            star[i, i] = 1 - 1 / node_num

    return star


def get_complete_overlay(node_num):

    complete = np.ones((node_num, node_num), dtype=np.float64)
    complete /= node_num

    return complete


def get_isolated_overlay(node_num):

    isolated = np.zeros((node_num, node_num), dtype=np.float64)

    for i in range(node_num):
        isolated[i, i] = 1

    return isolated


def get_balanced_tree_overlay(node_num, degree=2):

    tree = np.zeros((node_num, node_num), dtype=np.float64)

    for i in range(node_num):
        for j in range(1, degree+1):
            k = i * 2 + j
            if k >= node_num:
                break
            tree[i, k] = 1 / (degree+1)

    for i in range(node_num):
        tree[i, i] = 1 - tree[i, :].sum()

    return tree


def get_barbell_overlay(node_num, m1=1, m2=0):

    barbell = None

    return barbell


def get_random_overlay(node_num, probability=0.5):

    random = np.array(
        nx.to_numpy_matrix(nx.fast_gnp_random_graph(node_num, probability)), dtype=np.float64
    )

    matrix_sum = random.sum(1)

    for i in range(node_num):
        for j in range(node_num):
            if i != j and random[i, j] > 0:
                random[i, j] = 1 / (1 + max(matrix_sum[i], matrix_sum[j]))
        random[i, i] = 1 - random[i].sum()

    return random
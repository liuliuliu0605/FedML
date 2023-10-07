import os
import sys
import time

import numpy as np
import torch
import wandb
import logging

from sklearn.cluster import KMeans
from math import sqrt


def calculate_optimal_tau(loss_delta, K, L, sigma, gamma, psi, p, agg_cost, mix_cost, U, N_tilde, zeta=1.0):

    def h(tau):
        a1 = L * loss_delta / sqrt(K) + sigma * zeta / sqrt(K) / sqrt(N_tilde)
        a2 = sigma * zeta + K * gamma * zeta
        a3 = K * sigma * zeta
        a4 = K * psi * zeta

        d_agg = agg_cost
        d_mix = mix_cost

        phi = (tau * d_agg + d_mix) / U / tau

        A = a1 * sqrt(phi)
        B = phi * (a2 + a3 * tau / p + a4 * tau**2 / p**2)
        H = A + B

        return H

    opt_tau = 1
    opt_value = sys.maxsize
    for tau in range(1, 200):
        h_value = h(tau)
        if h_value < opt_value:
            opt_tau = tau
            opt_value = h_value

    return opt_tau


def agg_parameter_estimation(param_estimation_dict, var_name, log_wandb=False):
    agg_param_estimation_dict = {}
    size = len(param_estimation_dict)
    for k in param_estimation_dict[0].keys():
        if k == 'grad':
            agg_param_estimation = {}
            var = 0
            for name in param_estimation_dict[0][k]:
                for i in range(size):
                    layer_grad = param_estimation_dict[i][k][name]
                    if k not in agg_param_estimation:
                        agg_param_estimation[name] = layer_grad / size
                    else:
                        agg_param_estimation[name] += layer_grad / size
                    var += (layer_grad ** 2).sum() / size

            for name in agg_param_estimation:
                var -= (agg_param_estimation[name] ** 2).sum()

            agg_param_estimation_dict[var_name] = var
            if var_name == 'gamma':
                agg_param_estimation_dict['grad'] = agg_param_estimation

        else:
            agg_param_estimation_dict[k] = sum(
                [param_estimation_dict[idx][k] for idx in range(size)]
            ) / size

    if log_wandb:
        wandb.log(agg_param_estimation_dict)
    return agg_param_estimation_dict


def cal_mixing_consensus_speed(topo_weight_matrix, global_round_idx, args):
    n_rows, n_cols = np.shape(topo_weight_matrix)
    assert n_rows == n_cols
    A = np.array(topo_weight_matrix) - 1 / n_rows
    p = 1 - np.linalg.norm(A, ord=2) ** 2
    if args.enable_wandb:
        wandb.log({"Groups/p": p, "comm_round": global_round_idx})
    return p


def stats_group(group_to_client_indexes, train_data_local_dict, train_data_local_num_dict, class_num, args):

    xs = [i for i in range(class_num)]
    ys = []
    keys = []
    for group_idx in range(len(group_to_client_indexes)):
        data_size = 0
        group_y_train = []
        for client_id in group_to_client_indexes[group_idx]:
            data_size += train_data_local_num_dict[client_id]
            y_train = torch.concat([y for _, y in train_data_local_dict[client_id]]).tolist()
            group_y_train.extend(y_train)

        labels, counts = np.unique(group_y_train, return_counts=True)

        count_vector = np.zeros(class_num)
        count_vector[labels] = counts
        ys.append(count_vector/count_vector.sum())
        keys.append("Group {}".format(group_idx))

        if args.enable_wandb:
            wandb.log({"Groups/Client_num": len(group_to_client_indexes[group_idx]), "group_id": group_idx})
            wandb.log({"Groups/Data_size": data_size, "group_id": group_idx})

        logging.info("Group {}: client num={}, data size={} ".format(
            group_idx,
            len(group_to_client_indexes[group_idx]),
            data_size
        ))

    if args.enable_wandb:
        wandb.log({"Groups/Data_distribution":
                       wandb.plot.line_series(xs=xs, ys=ys, keys=keys, title="Data distribution", xname="Label")}
                  )


def hetero_partition_groups(clients_type_list, num_groups, alpha=0.5):
    min_size = 0
    num_type = np.unique(clients_type_list).size
    N = len(clients_type_list)
    group_to_client_indexes = {}
    while min_size < 10:
        idx_batch = [[] for _ in range(num_groups)]
        # for each type in clients
        for k in range(num_type):
            idx_k = np.where(np.array(clients_type_list) == k)[0]
            np.random.shuffle(idx_k)
            proportions = np.random.dirichlet(np.repeat(alpha, num_groups))
            ## Balance
            proportions = np.array([p * (len(idx_j) < N / num_groups) for p, idx_j in zip(proportions, idx_batch)])
            proportions = proportions / proportions.sum()
            proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
            idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
            min_size = min([len(idx_j) for idx_j in idx_batch])

    group_indexes = [0 for _ in range(N)]
    for j in range(num_groups):
        np.random.shuffle(idx_batch[j])
        group_to_client_indexes[j] = idx_batch[j]
        for client_id in group_to_client_indexes[j]:
            group_indexes[client_id] = j

    return group_indexes, group_to_client_indexes


def analyze_clients_type(train_data_local_dict, class_num, num_type=5):
    client_feature_list = []
    for i in range(len(train_data_local_dict)):
        y_train = torch.concat([y for _, y in train_data_local_dict[i]])
        labels, counts = torch.unique(y_train, return_counts=True)
        data_feature = np.zeros(class_num)
        total = 0
        for label, count in zip(labels, counts):
            data_feature[label.item()] = count.item()
            total += count.item()
        data_feature /= total
        client_feature_list.append(data_feature)

    kmeans = KMeans(n_clusters=num_type, random_state=0, n_init="auto").fit(client_feature_list)



    # for k in range(num_type):
    #     tmp = []
    #     for i, j in enumerate(kmeans.labels_):
    #         if j == k:
    #             indexes = np.where(np.array(client_feature_list[i]) > 0)
    #             tmp.extend(indexes[0].tolist())
    #     print(np.unique(tmp))
    #
    # exit(0)
    return kmeans.labels_


def transform_list_to_tensor(model_params_list):
    for k in model_params_list.keys():
        model_params_list[k] = torch.from_numpy(
            np.asarray(model_params_list[k])
        ).float()
    return model_params_list


def transform_tensor_to_list(model_params):
    for k in model_params.keys():
        model_params[k] = model_params[k].detach().numpy().tolist()
    return model_params


def post_complete_message_to_sweep_process(args):
    pipe_path = "./tmp/fedml"
    os.system("mkdir -p ./tmp/; touch ./tmp/fedml")
    if not os.path.exists(pipe_path):
        os.mkfifo(pipe_path)
    pipe_fd = os.open(pipe_path, os.O_WRONLY)

    with os.fdopen(pipe_fd, "w") as pipe:
        pipe.write("training is finished! \n%s\n" % (str(args)))
    time.sleep(3)


if __name__ == '__main__':

    params = {'sigma': 19.19629517252362, 'L': 7225.820201343054, 'gamma': 34.19849039359468,
              'psi': 0.29461052466206183,
              'K': 7.417171717171717, 'loss': 4.122456542323485}

    params = {'sigma': 19.140667618985383, 'L': 7095.644001106019, 'gamma': 33.10166782010226,
              'psi': 0.7018323426485723, 'K': 8.14754010695187, 'loss': 4.135336632172672}

    loss_delta = params['loss']
    K = params['K']
    L = params['L']
    sigma = params['sigma']
    gamma = params['gamma']
    psi = params['psi']
    p = 1.0
    agg_cost = 1
    mix_cost = 100
    U = 100
    N_tilde = 1000  # TODO
    total_params =None
    zeta = 1#1e-10 * p**2

    opt_tau = calculate_optimal_tau(loss_delta, K, L, sigma, gamma, psi, p, agg_cost, mix_cost, U, N_tilde,
                                    total_params, zeta)
    print(opt_tau)
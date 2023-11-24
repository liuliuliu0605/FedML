import os
import sys
import time

import numpy as np
import torch
import wandb
import logging

from sklearn.cluster import KMeans
from math import sqrt
from ns import ns


def time_consuming_one_round(
        args, process_id, mpi_comm, network, sampled_client_indexes, model_size,
        topology_manager, system_id_list
):
    config_param = "{}-{}".format(args.group_comm_pattern, args.group_comm_round)
    if args.fast_mode and config_param in network.time_history:
        logging.info("Rank {} runs in fast mode".format(process_id))
        delay_matrix, region_delay, global_delay = network.get_history(config_param)
        args.ns3_time += delay_matrix.max()
    else:
        logging.info("Rank {} is running ns3 simulator".format(process_id))
        network.connect_pses(topology_manager, enable_optimization=True)

        client_num_list = [len(sampled_client_indexes[i]) for i in range(args.group_num)]
        network.select_clients(client_num_list, method='near_edge_ps')

        comm = ns.cppyy.gbl.Convert2MPIComm(mpi_comm)
        ns.mpi.MpiInterface.Enable(comm)
        network.construct_network(graph_partition_method='girvan_newman', system_id_list=system_id_list)

        # run simulation
        if args.group_comm_pattern == 'decentralized':
            delay_matrix, region_delay, global_delay = network.run_fl_pfl(model_size=model_size,
                                                                          group_comm_round=args.group_comm_round,
                                                                          mix_comm_round=1,
                                                                          start_time=0, stop_time=10000000)
        elif args.group_comm_pattern == 'centralized':
            delay_matrix, region_delay, global_delay = network.run_fl_hfl(model_size=model_size,
                                                                          group_comm_round=args.group_comm_round,
                                                                          start_time=0, stop_time=10000000)
        elif args.group_comm_pattern == 'allreduce':
            delay_matrix, region_delay, global_delay = network.run_fl_rar(model_size=model_size,
                                                                          group_comm_round=args.group_comm_round,
                                                                          start_time=0, stop_time=10000000)
        else:
            raise NotImplementedError

        # record consumed time in history
        network.add_history(config_param, (delay_matrix, region_delay, global_delay))

        ns.mpi.MpiInterface.Disable()

        args.ns3_time += delay_matrix.max()

        if process_id == 0 and args.enable_wandb:
            wandb.log({"Estimation/ps_client_time": region_delay.mean(), "comm_round": args.round_idx})
            wandb.log({"Estimation/ps_ps_time": global_delay.mean(), "comm_round": args.round_idx})
            wandb.log({"Estimation/model_size": model_size, "comm_round": args.round_idx})

            # TODO: save in wandb
            # network.plot_ps_overlay_topology(save_path="overlay.png")


def calculate_optimal_tau1(args, convergence_param_dict, time_dict, p, N_tilde, zeta=1.0):
    # {'sigma': 1.6627908909580238, 'L': 80.46860672820361, 'gamma': 4.5684504507218975, 'psi': 0.041040657805953944,
    #  'K': 7.530555555555556, 'loss': 2.31964097155026}
    loss_delta = convergence_param_dict['loss']
    L = convergence_param_dict['L']
    sigma = convergence_param_dict['sigma']
    gamma = convergence_param_dict['gamma']
    psi = convergence_param_dict['psi']
    K= convergence_param_dict['K']

    agg_cost = time_dict['agg_cost']
    mix_cost = time_dict['mix_cost']
    U = time_dict['budget']

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
    for tau in range(1, 1001):
        h_value = h(tau)
        if h_value < opt_value:
            opt_tau = tau
            opt_value = h_value

    if args.enable_wandb:
        wandb.log({"Estimation/tau": opt_tau, "comm_round": args.round_idx})

    return opt_tau


def agg_parameter_estimation(args, param_estimation_dict, var_name, log_wandb=False):
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
            # if var_name == 'gamma':
            agg_param_estimation_dict['grad'] = agg_param_estimation
        elif k == 'cum_grad_delta':
            agg_param_estimation = {}
            for name in param_estimation_dict[0][k]:
                for i in range(size):
                    layer_grad = param_estimation_dict[i][k][name]
                    if k not in agg_param_estimation:
                        agg_param_estimation[name] = layer_grad / size
                    else:
                        agg_param_estimation[name] += layer_grad / size

            agg_param_estimation_dict['cum_grad_delta'] = agg_param_estimation
        else:
            agg_param_estimation_dict[k] = sum(
                [param_estimation_dict[idx][k] for idx in range(size)]
            ) / size

    if var_name == 'psi':
        cum_grad_delta_square = agg_param_estimation_dict['cum_grad_delta_square']
        cum_grad_delta_square2 = 0
        for name in agg_param_estimation_dict['cum_grad_delta']:
            cum_grad_delta_square2 += (agg_param_estimation_dict['cum_grad_delta'][name]**2).sum().item()
        zeta = cum_grad_delta_square2 / cum_grad_delta_square
        agg_param_estimation_dict['zeta'] = zeta

    if log_wandb:
        for key in agg_param_estimation_dict:
            if key not in ['cum_grad_delta', 'grad']:
                wandb.log({"Estimation/%s" % key: agg_param_estimation_dict[key], "comm_round": args.round_idx})
    return agg_param_estimation_dict


def cal_mixing_consensus_speed(args, topo_weight_matrix):
    n_rows, n_cols = np.shape(topo_weight_matrix)
    assert n_rows == n_cols
    A = np.array(topo_weight_matrix) - 1 / n_rows
    p = 1 - np.linalg.norm(A, ord=2) ** 2
    if args.enable_wandb:
        wandb.log({"Estimation/p": p, "comm_round": args.round_idx})
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


def analyze_clients_type(train_data_local_dict, class_num, num_type=5, random_seed=0):
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

    kmeans = KMeans(n_clusters=num_type, random_state=random_seed).fit(client_feature_list)

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
        pipe.write("training is finished!!!! \n%s\n" % (str(args)))
    time.sleep(3)


def calculate_optimal_tau(args, convergence_param_dict, time_dict, p, N_tilde):
    loss_delta = convergence_param_dict['loss']
    L = convergence_param_dict['L']
    sigma = convergence_param_dict['sigma']
    gamma = convergence_param_dict['gamma']
    psi = convergence_param_dict['psi']
    K = convergence_param_dict['K']
    zeta = convergence_param_dict['zeta']

    agg_cost = time_dict['agg_cost']
    mix_cost = time_dict['mix_cost']
    U = time_dict['budget']

    def h(tau):
        a1 = 16 * L * loss_delta / sqrt(K) + 16 * sigma * zeta / sqrt(K) / sqrt(N_tilde)
        a2 = 48 * (sigma * zeta + 18 * K * gamma * zeta)
        a3 = 768 * sigma * zeta
        a4 = 768 * 16 * K * psi * zeta

        d_agg = agg_cost
        d_mix = mix_cost

        phi = (tau * d_agg + d_mix) / U / tau

        A = a1 * sqrt(phi)
        B = phi * (a2 + a3 * tau / p + a4 * tau**2 / p**2)
        H = A + B

        return H

    opt_tau = 1
    opt_value = sys.maxsize
    for tau in range(1, 1001):
        h_value = h(tau)
        if h_value < opt_value:
            opt_tau = tau
            opt_value = h_value

    if args.enable_wandb:
        wandb.log({"Estimation/tau": opt_tau, "comm_round": args.round_idx})

    return opt_tau


if __name__ == '__main__':

    fmnist_params = {'sigma': 1.7, 'L': 72.897, 'gamma': 4.713,
              'psi': 0.06666, 'K': 6.471, 'loss': 2.32, 'zeta': 4.5e-7}

    cifar_params =  {'sigma': 1346.6006447640223, 'L': 2205.9963775601145, 'gamma': 705.6212093459262,
                     'psi': 8.30353176121967, 'K': 4.61919191919192, 'loss': 2.4087824613888023,
                     'local_update_time': 0.07494886665461212, 'num_params': 600372}
    p = 0.2
    N_tilde = 1000
    convergence_param_dict = fmnist_params
    # zeta = 1/convergence_param_dict['num_params'] #1e-10 * p**2
    zeta = convergence_param_dict['zeta']
    # zeta = 1 #1e-10 * p**2

    time_dict = {
        'agg_cost': 1.2,
        'mix_cost': 0.99,
        'budget': 2000
    }

    class ARGS:
        enable_wandb = False

    opt_tau = calculate_optimal_tau(ARGS(), convergence_param_dict, time_dict, p, N_tilde)
    print(opt_tau)
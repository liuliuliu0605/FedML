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
from PIL import Image


def calculate_optimal_tau(args, convergence_param_dict, time_dict, p):
    # del convergence_param_dict['grad']
    # del convergence_param_dict['cum_grad_delta']
    # print("------", convergence_param_dict)
    # exit(0)
    loss_delta = convergence_param_dict['loss']
    L = convergence_param_dict['L']
    sigma = convergence_param_dict['sigma']
    gamma = convergence_param_dict['gamma']
    psi = convergence_param_dict['psi']
    K = convergence_param_dict['K']
    zeta = convergence_param_dict['zeta']
    N = convergence_param_dict['N']
    N_tilde = convergence_param_dict['N_tilde']
    n = convergence_param_dict['n']
    n_tilde = convergence_param_dict['n_tilde']
    avgN_minN = convergence_param_dict['avgN_minN']

    agg_cost = time_dict['agg_cost']
    mix_cost = time_dict['mix_cost']
    U = time_dict['budget']

    def h(tau):
        a1 = 16 * L * loss_delta / sqrt(K*n_tilde) \
             + 8 * sigma * zeta / sqrt(K*n_tilde) \
             + 24 * L * N * (N-n) * sqrt(n_tilde) * (sigma + 18 * K *gamma) / (N-1) /n / sqrt(K) * zeta \
             + avgN_minN * 432 * sqrt(K) * (N - n) * sqrt(n_tilde) * psi / (N-1) / n * zeta
        a2 = 24 * n_tilde* (sigma + 18 * K * gamma) * zeta
        a3 = 768 * n_tilde * sigma * zeta
        a4 = 768 * 16 * n_tilde * K * psi * zeta

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
        wandb.log({"Estimation/objective": opt_value, "comm_round": args.round_idx})

    # logging.info(
    #     "convergence_param_dict={}, time_dict={}, p={}, opt_tau={}".format(convergence_param_dict, time_dict, p, opt_tau)
    # )

    return opt_tau, opt_value


def adjust_topo(args, topo_action_objective_list, network):
    # 1 refers to add and -1 refers to remove
    action = 1

    if len(topo_action_objective_list) >= 2:
        if topo_action_objective_list[-1][1] <= topo_action_objective_list[-2][1]:
            action = topo_action_objective_list[-1][0][0]
        else:
            action = -topo_action_objective_list[-1][0][0]

    topology = network.topology_manager.topology
    choice = None
    if action == 1:
        # add edge with the lowest latency
        minimum = sys.maxsize
        for i in range(len(topology)):
            for j in range(i+1, len(topology[i])):
                if topology[i, j] == 0:
                    latency = network.get_latency(i, j)
                    if latency < minimum:
                        minimum = latency
                        choice = (i, j)
        if choice is not None:
            network.add_edge(*choice)
    elif action == -1:
        # remove edge with the highest latency
        maximum = -1
        for i in range(len(topology)):
            for j in range(i+1, len(topology[i])):
                if topology[i, j] > 0 and \
                        len(network.topology_manager.get_in_neighbor_idx_list(i)) > 2 and \
                        len(network.topology_manager.get_in_neighbor_idx_list(j)) > 2:
                    latency = network.get_latency(i, j)
                    if latency > maximum:
                        maximum = latency
                        choice = (i,j)
        if choice is not None:
            network.remove_edge(*choice)

    action = action if choice is not None else 0
    topo_action_objective_list.append([(action, choice), None])

    return action, choice


def time_consuming_one_round(
        args, process_id, mpi_comm, network, sampled_group_to_client_indexes, model_size, system_id_list
):
    config_param = "{}-{}-{}".format(args.group_comm_pattern, args.group_comm_round,
                                     network.topology_manager.topology if network.topology_manager is not None else 'none')
    if args.fast_mode and config_param in network.time_history:
        logging.info("Rank {} runs in fast mode".format(process_id))
        delay_matrix, region_delay, global_delay = network.get_history(config_param)

    else:
        logging.info("Rank {} is running ns3 simulator".format(process_id))
        # network.connect_pses(topology_manager, enable_optimization=True)

        client_num_list = [len(sampled_group_to_client_indexes[i][0]) for i in range(args.group_num)]
        network.select_clients(client_num_list, method='near_edge_ps')

        comm = ns.cppyy.gbl.Convert2MPIComm(mpi_comm)
        ns.mpi.MpiInterface.Enable(comm)
        network.construct_network(graph_partition_method='girvan_newman', system_id_list=system_id_list)

        # run simulation
        if args.group_comm_pattern == 'decentralized':
            delay_matrix, region_delay, global_delay = network.run_fl_pfl(model_size=model_size,
                                                                          group_comm_round=args.group_comm_round,
                                                                          mix_comm_round=1,
                                                                          start_time=0, stop_time=10000000,
                                                                          fast_forward=args.fast_mode)
        elif args.group_comm_pattern == 'centralized':
            delay_matrix, region_delay, global_delay = network.run_fl_hfl(model_size=model_size,
                                                                          group_comm_round=args.group_comm_round,
                                                                          start_time=0, stop_time=10000000,
                                                                          fast_forward=args.fast_mode)
        elif args.group_comm_pattern == 'async-centralized':
            delay_matrix, region_delay, global_delay = network.run_async_fl_hfl(model_size=model_size,
                                                                                group_comm_round=args.group_comm_round,
                                                                                start_time=0, stop_time=10000000,
                                                                                fast_forward=args.fast_mode)
        elif args.group_comm_pattern == 'allreduce':
            delay_matrix, region_delay, global_delay = network.run_fl_rar(model_size=model_size,
                                                                          group_comm_round=args.group_comm_round,
                                                                          start_time=0, stop_time=10000000,
                                                                          fast_forward=args.fast_mode)
        else:
            raise NotImplementedError

        # record consumed time in history
        network.add_history(config_param, (delay_matrix, region_delay, global_delay))

        ns.mpi.MpiInterface.Disable()

        if process_id == 0 and args.enable_wandb:
            wandb.log({"Estimation/ps_client_time": region_delay.mean(), "comm_round": args.round_idx})
            wandb.log({"Estimation/ps_ps_time": global_delay.mean(), "comm_round": args.round_idx})
            wandb.log({"Estimation/model_size": model_size, "comm_round": args.round_idx})

            if args.enable_dynamic_topo or args.round_idx == 0:
                import io
                buf = io.BytesIO()
                network.plot_ps_overlay_topology(figsize=(10, 10), save_path=buf)
                buf.seek(0)
                img = Image.open(buf)
                wandb.log({"Topology": wandb.Image(img), "comm_round": args.round_idx})

    if args.group_comm_pattern in ['decentralized', 'centralized', 'allreduce']:
        args.ns3_time += delay_matrix.max()
    elif args.group_comm_pattern in ['async-centralized']:
        current_time_list = region_delay + global_delay + args.ns3_time_arr
        min_index = np.argmin(current_time_list)
        args.ns3_time = current_time_list[min_index]
        args.ns3_time_arr[min_index] = args.ns3_time


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
        # wandb.log({"Groups/Data_distribution": wandb.Table(data=ys, columns=list(range(class_num)))})
        wandb.log({"Groups/Data_distribution":
                       wandb.plot.line_series(xs=xs, ys=ys, keys=keys, title="Data distribution", xname="Label")}
                  )


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
    # waiting for the results to be uploaded
    time.sleep(20)


# def adjust_topo2(tau, network, loss_delta, K, L, sigma, gamma, psi, U, N_tilde, total_params):
#     init_topo_matrix = network.get_ps_matrix()
#     init_weight_matrix = optimal_mixing_weight(init_topo_matrix)
#     p = 1 - np.linalg.norm(init_weight_matrix - 1 / network.ps_num, ord=2) ** 2
#
#     delay_matrix = network.get_delay_matrix()
#     topo_matrix = init_topo_matrix.copy()
#     mix_delay = network.get_mix_delay()
#     # network.print_mix_delay_matrix_from_history()
#     # print(mix_delay)
#     agg_delay = network.get_agg_delay()
#
#     min_obj = calculate_optimal_tau(tau, loss_delta , K, L, sigma, gamma, psi, p, agg_delay, mix_delay, U, N_tilde, total_params)
#     used = []
#     un_used = []
#     for i in range(len(init_topo_matrix)):
#         for j in range(i+1, len(init_topo_matrix[i])):
#             if init_topo_matrix[i, j] > 0:
#                 # used.append((i, j, (delay_matrix[i, j]+delay_matrix[j, i])/2))
#                 used.append((i, j, max(delay_matrix[i, j], delay_matrix[j, i])))
#             else:
#                 # un_used.append((i, j, (delay_matrix[i, j] + delay_matrix[j, i])/2))
#                 un_used.append((i, j, max(delay_matrix[i, j], delay_matrix[j, i])))
#     used = sorted(used, key=lambda item: item[-1])
#     un_used = sorted(un_used, key=lambda item: item[-1])
#     # print(used)
#     # print(un_used)
#     last_success = True
#     current_success = True
#     last_action = 0
#     current_action = 0
#     count = 0
#     while True:
#         count += 1
#         # print(topo_matrix)
#         # print(weight_matrix)
#         print("----------%d--------"%count)
#         if len(un_used) == 0:
#             # should delete an edge
#             probability = 0
#         elif len(used) == 0:
#             # should add an edge
#             probability = 1
#         else:
#             probability = np.random.random()
#         new_topo_matrix = topo_matrix.copy()
#
#         if probability > 0.5:
#             # add
#             current_action = 1
#             i, j, delay = un_used[0]
#             new_topo_matrix[i, j] = new_topo_matrix[j, i] = 1
#             print("prepare to add:", (i, j, delay))
#
#         else:
#             # delete
#             current_action = -1
#             i, j, delay = used[-1]
#             new_topo_matrix[i, j] = new_topo_matrix[j, i] = 0
#             print("prepare to delete:", (i, j, delay))
#
#         new_weight_matrix = optimal_mixing_weight(new_topo_matrix)
#         # print_matrix(new_weight_matrix)
#         current_p = 1 - np.linalg.norm(new_weight_matrix - 1 / len(new_weight_matrix), ord=2) ** 2
#         current_mix_delay = network.get_mix_delay(new_topo_matrix)
#
#         current_obj = calculate_optimal_tau(tau, loss_delta , K, L, sigma, gamma, psi, current_p, agg_delay, current_mix_delay, U, N_tilde, total_params)
#
#         print("min obj (p=%.5f, mix=%.5f): %.5f" % (p, mix_delay, min_obj))
#         print("current obj (p=%.5f, mix=%.5f): %.5f" % (current_p, current_mix_delay, current_obj))
#
#         if current_obj >= min_obj:
#             current_success = False
#             print("Cancel!")
#         else:
#             current_success = True
#             if current_action == 1:
#                 i, j, delay = un_used.pop(0)
#                 used.append((i, j, delay))
#                 network.add_connection(i, j, mix_weight=0)  # TODO
#                 print("Add PS %d - PS %d" % (network.get_PS_label(i), network.get_PS_label(j)))
#             else:
#                 i, j, delay = used.pop()
#                 un_used.append((i, j, delay))
#                 network.delete_connection(i, j)
#                 print("Delete PS %d - PS %d" % (network.get_PS_label(i), network.get_PS_label(j)))
#             min_obj = current_obj
#             p = current_p
#             mix_delay = current_mix_delay
#             topo_matrix = new_topo_matrix
#             used = sorted(used, key=lambda item: item[-1])
#             un_used = sorted(un_used, key=lambda item: item[-1])
#             print("Done!")
#         # print(current_success)
#
#         if current_action * last_action < 0 and not last_success and not current_success or \
#                not current_success and probability == 1 or probability == 0:
#             break
#         else:
#             last_action = current_action
#             last_success = current_success
#
#     # print(network.get_ps_matrix())
#     # network.plot_network('overlay', node_label=True, figsize=(4, 4))
#     return topo_matrix, p


if __name__ == '__main__':

    cifar_params =   {'N_tilde': 985.457284505729, 'n_tilde': 98.22344710621157, 'N': 1000, 'n': 100,
                      'avgN_minN': 1.3888888888888888, 'sigma': 26.11897600861355, 'L': 130.2531982786767,
                      'gamma': 20.74032638633736, 'psi': 0.0016904592404552765, 'K': 47.955192955192956,
                      'loss': 2.4603762316026785, 'cum_grad_delta_square': 115463.85614020767,
                      'zeta': 1.0733820774717823e-05}
    p = 0.3
    convergence_param_dict = cifar_params

    time_dict = {
        'agg_cost': 1.37,
        'mix_cost': 9.025,
        'budget': 1000
    }

    class ARGS:
        enable_wandb = False

    opt_tau = calculate_optimal_tau(ARGS(), convergence_param_dict, time_dict, p)
    print(opt_tau)
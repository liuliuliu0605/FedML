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
import scipy


def time_consuming_one_round(
        args, process_id, mpi_comm, network, sampled_group_to_client_indexes, model_size, group_comm_round_list,
        system_id_list
):
    # model_size=1000
    config_param = "{}-{}-{}".format(args.group_comm_pattern, group_comm_round_list,
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
            # delay_matrix, region_delay, global_delay = network.run_fl_pfl(model_size=model_size,
            #                                                               group_comm_round=args.group_comm_round,
            #                                                               mix_comm_round=1,
            #                                                               start_time=0, stop_time=10000000,
            #                                                               fast_forward=args.fast_mode)
            # this is a temporary solution to different taus
            delay_matrix, region_delay, global_delay = network.run_fl_pfl2(model_size=model_size,
                                                                           group_comm_round_list=group_comm_round_list,
                                                                           mix_comm_round=1,
                                                                           start_time=0, stop_time=10000000,
                                                                           fast_forward=True)
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
        logging.info("region delay:{}".format(region_delay))
        logging.info("global delay:{}".format(global_delay))

        # record consumed time in history
        network.add_history(config_param, (delay_matrix, region_delay, global_delay))

        ns.mpi.MpiInterface.Disable()

        if process_id == 0 and args.enable_wandb:
            wandb.log({"Estimation/ps_client_time_mean": region_delay.mean(), "comm_round": args.round_idx})
            wandb.log({"Estimation/ps_client_time_min": region_delay.min(), "comm_round": args.round_idx})
            wandb.log({"Estimation/ps_client_time_max": region_delay.max(), "comm_round": args.round_idx})
            wandb.log({"Estimation/ps_ps_time_mean": global_delay.mean(), "comm_round": args.round_idx})
            wandb.log({"Estimation/ps_ps_time_min": global_delay.min(), "comm_round": args.round_idx})
            wandb.log({"Estimation/ps_ps_time_max": global_delay.max(), "comm_round": args.round_idx})
            wandb.log({"Estimation/total_time_max": delay_matrix.max(), "comm_round": args.round_idx})
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
        current_time_list = delay_matrix.max(axis=0)[1:] + args.ns3_time_arr
        min_index = np.argmin(current_time_list)
        args.ns3_time = current_time_list[min_index]
        args.ns3_time_arr[min_index] = args.ns3_time
        if args.rank == 0:
            logging.info("==={}-{}-{}".format(args.ns3_time, args.ns3_time_arr, delay_matrix.max(axis=0)[1:]))


def calculate_optimal_tau(args, convergence_param_dict, time_dict, p, num_of_model_params):
    # del convergence_param_dict['grad']
    # del convergence_param_dict['cum_grad_delta']
    # print("------", convergence_param_dict)
    # exit(0)
    loss_delta = convergence_param_dict['loss']
    L = convergence_param_dict['L']
    sigma = convergence_param_dict['sigma']
    gamma = convergence_param_dict['gamma']
    psi = convergence_param_dict['psi']
    K = convergence_param_dict['num_of_local_updates_one_epoch'] * args.epochs
    # zeta = convergence_param_dict['zeta'] / num_of_model_params
    # zeta = convergence_param_dict['zeta'] / 2e5
    zeta = convergence_param_dict['zeta']
    N = convergence_param_dict['N']
    N_tilde = convergence_param_dict['N_tilde']
    n = convergence_param_dict['n']
    n_tilde = convergence_param_dict['n_tilde']
    avgN_minN = convergence_param_dict['avgN_minN']

    agg_cost = time_dict['agg_cost']
    mix_cost = time_dict['mix_cost']
    U = time_dict['budget']

    # # same tau
    if not args.enable_diff_tau:
        print("!!!!debug")
        print(p, loss_delta, L, sigma, gamma, psi, K, zeta, N, N_tilde, n, n_tilde, avgN_minN, agg_cost, mix_cost, U)
        exit
        def h(tau):
            a1 = 16 * L * loss_delta / sqrt(K * n_tilde) \
                 + 8 * sigma / sqrt(K * n_tilde) \
                 + 24 * N * (N - n) * sqrt(n_tilde) * (sigma + 18 * K * gamma) / N_tilde / (N - 1) / n / sqrt(K) \
                 + 432 * avgN_minN * sqrt(K) * (N - n) * sqrt(n_tilde) * psi / (N - 1) / n

            a2 = 24 * n_tilde * (sigma + 18 * K * gamma) * zeta
            a3 = 768 * n_tilde * sigma * zeta
            a4 = 768 * 16 * n_tilde * K * psi * zeta

            d_agg = agg_cost
            d_mix = mix_cost

            # T_mix = np.min(U / (tau * d_agg + d_mix))
            # T = T_mix * tau
            T_mix = min((U - args.ns3_time) / (tau * d_agg + d_mix))
            T = args.round_idx + T_mix * tau
            phi = 1 / T

            A = a1 * sqrt(phi)
            B = phi * (a2 + a3 * tau / p + a4 * tau ** 2 / p ** 2)
            H = A + B

            return H

        opt_tau = 1
        opt_value = sys.maxsize
        for tau in range(1, 1001):
            h_value = h(tau)
            if h_value < opt_value:
                opt_tau = tau
                opt_value = h_value
        opt_tau = np.array([opt_tau] * args.group_num, dtype=int)
    else:
        # different tau
        def h2(tau_list):
            tau_max = np.max(tau_list)
            tau_mean = np.mean(tau_list)
            tau_square_mean = np.mean(tau_list ** 2)
            mu = tau_mean / tau_max
            mu = 1

            a1 = 16 * L * loss_delta / sqrt(K * n_tilde * mu) \
                 + 8 * sigma / sqrt(K * n_tilde * mu) \
                 + 24 * N * (N - n) * sqrt(n_tilde) * (sigma + 18 * K * gamma * mu) / N_tilde / (N - 1) / n / sqrt(K * mu) \
                 + 432 * avgN_minN * sqrt(K * mu) * (N - n) * sqrt(n_tilde) * psi / (N - 1) / n

            a2 = 24 * n_tilde * (sigma + 18 * K * gamma * mu) * zeta
            a3 = 768 * n_tilde * sigma * zeta
            a4 = 768 * 16 * n_tilde * K * psi * zeta * mu

            d_agg = agg_cost
            d_mix = mix_cost

            # T_mix = np.min(U / (tau_list * d_agg + d_mix))
            # T = T_mix * tau_mean
            T_mix = min((U-args.ns3_time) / (tau_list * d_agg + d_mix))
            T = args.round_idx + T_mix * tau_mean
            phi = 1 / T

            A = a1 * sqrt(phi)
            B = phi * (a2 + a3 * tau_mean / p + a4 * tau_square_mean / p ** 2)
            H = A + B

            return H

        tau_ini_2 = np.array([1.0] * args.group_num)
        bounds = [(1, None)] * args.group_num
        res = scipy.optimize.minimize(h2, tau_ini_2, bounds=bounds)
        opt_tau = np.array([max(round(tau), 1) for tau in res.x])
        opt_value = h2(opt_tau)

    if args.enable_wandb:
        wandb.log({"Estimation/tau": np.mean(opt_tau), "comm_round": args.round_idx})
        wandb.log({"Estimation/tau_min": np.min(opt_tau), "comm_round": args.round_idx})
        wandb.log({"Estimation/tau_max": np.max(opt_tau), "comm_round": args.round_idx})
        if args.enable_diff_tau:
            for i, tau in enumerate(opt_tau):
                wandb.log({"Estimation/tau(%d)" % i: tau, "comm_round": args.round_idx})
        wandb.log({"Estimation/objective": opt_value, "comm_round": args.round_idx})

    # logging.info(
    #     "convergence_param_dict={}, time_dict={}, p={}, opt_tau={}".format(convergence_param_dict, time_dict, p, opt_tau)
    # )

    return opt_tau, opt_value


def cal_control_ratio(args, convergence_param_dict, log_wandb=False):
    loss_delta = convergence_param_dict['loss']
    L = convergence_param_dict['L']
    sigma = convergence_param_dict['sigma']
    gamma = convergence_param_dict['gamma']
    psi = convergence_param_dict['psi']
    K = convergence_param_dict['K']
    N = convergence_param_dict['N']
    N_tilde = convergence_param_dict['N_tilde']
    n = convergence_param_dict['n']
    n_tilde = convergence_param_dict['n_tilde']
    avgN_minN = convergence_param_dict['avgN_minN']
    grad_square = convergence_param_dict['grad_square']

    a = 16 * L * loss_delta / sqrt(K * N_tilde)

    b = 8 * sigma / sqrt(K * N_tilde) \
        + 24 * N_tilde * (sigma + 18 * K * gamma) \
        + 768 * N_tilde * (sigma + 16 * K * psi) \
        # + 24 * N * (N - n) * sqrt(n_tilde) * (sigma + 18 * K * gamma) / N_tilde / (N - 1) / n / sqrt(K) \
    # + 432 * avgN_minN * sqrt(K) * (N - n) * sqrt(n_tilde) * psi / (N - 1) / n

    # print("~~~~~~~~~~~~~{}-{}-{}-{}".format(L, loss_delta, K, n_tilde))
    # print("~~~~~~~~~~~~{}-{}-{}".format(grad_square, a, b))
    # exit(0)
    control_ratio = (grad_square - a) / b

    if log_wandb:
        wandb.log({"Estimation/control_ratio": control_ratio, "comm_round": args.round_idx})

    return control_ratio


def agg_parameter_estimation(args, param_estimation_dict, var_name, log_wandb=False):
    agg_param_estimation_dict = {}
    size = len(param_estimation_dict)
    # total_sample_number = sum([param_estimation_dict[i]['sample_number'] for i in range(size)])

    for k in param_estimation_dict[0].keys():
        if k == 'grad':
            agg_param_estimation = {}
            var = 0
            for name in param_estimation_dict[0][k]:
                for i in range(size):
                    layer_grad = param_estimation_dict[i][k][name]
                    if name not in agg_param_estimation:
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
                    if name not in agg_param_estimation:
                        agg_param_estimation[name] = layer_grad / size
                    else:
                        agg_param_estimation[name] += layer_grad / size

            agg_param_estimation_dict['cum_grad_delta'] = agg_param_estimation
        else:
            agg_param_estimation_dict[k] = sum(
                [param_estimation_dict[i][k] / size for i in range(size)]
            )

    # agg_param_estimation_dict['sample_number'] = total_sample_number

    # if var_name == 'psi':
    #     # cum_grad_delta_square = agg_param_estimation_dict['cum_grad_delta_square']
    #     # cum_grad_delta_square2 = 0
    #     # for name in agg_param_estimation_dict['cum_grad_delta']:
    #     #     cum_grad_delta_square2 += (agg_param_estimation_dict['cum_grad_delta'][name]**2).sum()
    #     # zeta = cum_grad_delta_square2 / cum_grad_delta_square
    #     # agg_param_estimation_dict['zeta'] = zeta
    #     grad_square = 0
    #     for name in agg_param_estimation_dict['grad']:
    #         grad_square += (agg_param_estimation_dict['grad'][name] ** 2).sum()
    #     agg_param_estimation_dict['grad_square'] = grad_square

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
            for j in range(i + 1, len(topology[i])):
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
            for j in range(i + 1, len(topology[i])):
                if topology[i, j] > 0 and \
                        len(network.topology_manager.get_in_neighbor_idx_list(i)) > 2 and \
                        len(network.topology_manager.get_in_neighbor_idx_list(j)) > 2:
                    latency = network.get_latency(i, j)
                    if latency > maximum:
                        maximum = latency
                        choice = (i, j)
        if choice is not None:
            network.remove_edge(*choice)

    action = action if choice is not None else 0
    topo_action_objective_list.append([(action, choice), None])

    return action, choice


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
        ys.append(count_vector / count_vector.sum())
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

    # cifar10, group_alpha=5, tau=6
    dataset = 'cifar100'
    group_alpha = 5

    if dataset == 'cifar10':

        if group_alpha == 5:
            # opt_tau = 35
            cifar_params = {'N_tilde': 985.457284505729, 'n_tilde': 98.22344710621157, 'N': 1000, 'n': 100,
                            'avgN_minN': 1.3888888888888888, 'sigma': 13.636278193897414, 'L': 11.951036648737805,
                            'gamma': 23.952738702108928, 'psi': 1.5046829705776297, 'K': 45.74451084941445,
                            'loss': 2.4747159191000696,
                            'local_update_time': 0.5466945874508538, 'cum_grad_delta_square': 293724.68497117446,
                            'zeta': 0.030683999249572606, 'grad_square': 2.6138900524440487}
            cifar_params['psi'] = 0.9431
            cifar_params['gamma'] = 20.868
            cifar_params['L'] = 9.483
            cifar_params['sigma'] = 35.422
        elif group_alpha == 0.05:
            # opt_tau = 13
            cifar_params = {'N_tilde': 985.457284505729, 'n_tilde': 98.22344710621157, 'N': 1000, 'n': 100,
                            'avgN_minN': 1.3888888888888888, 'sigma': 13.636278193897414, 'L': 11.951036648737805,
                            'gamma': 2.927, 'psi': 40.48, 'K': 45.74451084941445, 'loss': 2.4747159191000696,
                            'local_update_time': 0.5466945874508538, 'cum_grad_delta_square': 293724.68497117446,
                            'zeta': 0.041, 'grad_square': 2.6138900524440487}
            cifar_params['psi'] = 27.016
            cifar_params['gamma'] = 4.372
            cifar_params['L'] = 8.315
            cifar_params['sigma'] = 14.813

        time_dict = {
            'agg_cost': 1.37,
            'mix_cost': 9.119,
            'budget': 1000
        }

    elif dataset == 'cifar100':
        if group_alpha == 5:
            cifar_params = {'N_tilde': 999.9017697388065, 'n_tilde': 99.73880597014924, 'N': 1000, 'n': 100,
                            'avgN_minN': 1.019367991845056,
                            'sigma': 49.293, 'L': 7.53596809112238, 'gamma': 13.959, 'psi': 0.245,
                            'K': 47.60115661831789, 'loss': 6.959489085204291, 'local_update_time': 1.0178889619863662,
                            'cum_grad_delta_square': 3004725.042394018, 'zeta': 0.00078,
                            'grad_square': 1.2530372370224214}
            cifar_params['psi'] = 0.9126
            cifar_params['gamma'] = 96.854
            cifar_params['L'] = 10.156
            cifar_params['sigma'] = 365.439
        elif group_alpha == 0.05:
            # opt_tau = 22
            # cifar100, group_alpha=0.01
            cifar_params = {'N_tilde': 990.7201393675736, 'n_tilde': 98.40424778761061, 'N': 1000, 'n': 100,
                            'avgN_minN': 1.3071895424836601,
                            'sigma': 37.284, 'L': 8.324539218501027, 'gamma': 9.638, 'psi': 20.465,
                            'K': 46.09412893597616, 'loss': 6.960351847540296, 'local_update_time': 0.9825921302211116,
                            'cum_grad_delta_square': 3194317.8920561844, 'zeta': 0.0028,
                            'grad_square': 1.1858041433614863}
            cifar_params['psi'] = 4.003
            cifar_params['gamma'] = 115.683
            cifar_params['L'] = 10.199
            cifar_params['sigma'] = 423.837

        time_dict = {
            'agg_cost': 22.84,
            'mix_cost': 149.6,
            'budget': 20000
        }

    p = 1.0
    convergence_param_dict = cifar_params


    class ARGS:
        enable_wandb = False


    opt_tau = calculate_optimal_tau(ARGS(), convergence_param_dict, time_dict, p, num_of_model_params=0)
    print(opt_tau)

import logging
import numpy as np

from .message_define import MyMessage
from ....core.distributed.fedml_comm_manager import FedMLCommManager
from ....core.distributed.communication.message import Message
from .utils import post_complete_message_to_sweep_process, time_consuming_one_round, \
    cal_mixing_consensus_speed, calculate_optimal_tau, adjust_topo, cal_control_ratio


class HierFedAVGCloudManager(FedMLCommManager):
    def __init__(
            self,
            args,
            aggregator,
            group_indexes,
            group_to_client_indexes,
            comm=None,
            rank=0,
            size=0,
            backend="MPI",
            topology_manager=None,
            network=None,
    ):
        super().__init__(args, comm, rank, size, backend)
        self.args = args
        self.aggregator = aggregator
        self.group_indexes = group_indexes
        self.group_to_client_indexes = group_to_client_indexes
        self.round_num = args.comm_round
        self.args.round_idx = 0
        self.args.global_round_idx = 0
        self.topology_manager = topology_manager
        self.network = network

        total_clients = len(self.group_indexes)
        self.group_to_client_num_per_round = [
            args.client_num_per_round * len(self.group_to_client_indexes[i]) // total_clients
            for i in range(args.group_num)
        ]

        remain_client_num_list_per_round = args.client_num_per_round - sum(self.group_to_client_num_per_round)
        while remain_client_num_list_per_round > 0:
            self.group_to_client_num_per_round[remain_client_num_list_per_round - 1] += 1
            remain_client_num_list_per_round -= 1

        # make sure each group has at least one client
        for i in range(args.group_num):
            if self.group_to_client_num_per_round[i] == 0:
                self.group_to_client_num_per_round[i] = 1

        if hasattr(self.args, 'enable_ns3') and self.args.enable_ns3:
            self.args.ns3_time = 0
            self.args.ns3_time_arr = np.array([0. for _ in range(self.size-1)])
        self.trigger_dynamic_group_comm = True if self.args.group_comm_round <= 0 else False
        self.args.group_comm_round = 1 if self.args.group_comm_round <= 0 else self.args.group_comm_round

        self.num_of_model_params = 0

        N_tilde, n_tilde = 0, 0
        for i in range(args.group_num):
            N_tilde += 1/len(self.group_to_client_indexes[i])/args.group_num**2
            n_tilde += 1/self.group_to_client_num_per_round[i]/args.group_num**2

        self.convergence_param_dict = {
            'N_tilde': 1 / sum([1/len(self.group_to_client_indexes[i])/args.group_num**2 for i in range(args.group_num)]),
            'n_tilde': 1 / sum([1/self.group_to_client_num_per_round[i]/args.group_num**2 for i in range(args.group_num)]),
            'N': total_clients,
            'n': args.client_num_per_round,
            'avgN_minN': total_clients/args.group_num/min([len(self.group_to_client_indexes[i]) for i in range(args.group_num)])
        }

        self.topo_action_objective_list = [[(0, None), None]]  # (action, objective)

    def run(self):
        super().run()

    def send_init_msg(self):
        # broadcast to edge servers
        global_model_params = self.aggregator.get_global_model_params()

        # get the number of model params
        self.num_of_model_params = 0
        for k in global_model_params:
            self.num_of_model_params += global_model_params[k].numel()

        # sample clients for the next group_comm_round, this could be done for each edge server
        # we use cloud server to do this for simple implementation
        sampled_group_to_client_indexes = {group_idx: [] for group_idx in range(self.args.group_num)}
        for group_comm_idx in range(self.args.group_comm_round):

            for group_idx, client_num_per_round in enumerate(self.group_to_client_num_per_round):
                client_num_in_total = len(self.group_to_client_indexes[group_idx])
                sampled_client_indexes = self.aggregator.client_sampling(
                    self.args.global_round_idx,
                    client_num_in_total,
                    client_num_per_round,
                )
                client_idx_list = []
                for index in sampled_client_indexes:
                    client_idx = self.group_to_client_indexes[group_idx][index]
                    client_idx_list.append(client_idx)
                sampled_group_to_client_indexes[group_idx].append(client_idx_list)
            self.args.global_round_idx += 1

        logging.info(
            "client_indexes of each group = {}".format(sampled_group_to_client_indexes)
        )

        group_to_data_size = {}
        for group_idx in range(self.args.group_num):
            data_size = 0
            for client_idx in self.group_to_client_indexes[group_idx]:
                data_size += self.aggregator.train_data_local_num_dict[client_idx]
            group_to_data_size[group_idx] = data_size

        for process_id in range(1, self.size):
            self.send_message_init_config(
                process_id,
                global_model_params,
                self.group_to_client_indexes,
                sampled_group_to_client_indexes,
                group_to_data_size,
                process_id - 1
            )

        if self.args.enable_ns3:
            # time consumed int the coming round
            time_consuming_one_round(
                self.args, self.rank, self.comm, self.network, sampled_group_to_client_indexes,
                self.num_of_model_params * 4, list(range(1, self.size))
            )

    def register_message_receive_handlers(self):
        self.register_message_receive_handler(
            MyMessage.MSG_TYPE_E2C_SEND_MODEL_TO_CLOUD,
            self.handle_message_receive_model_from_edge,
        )

    def handle_message_receive_model_from_edge(self, msg_params):
        logging.info("handle_message_receive_model_from_edge.")
        sender_id = msg_params.get(MyMessage.MSG_ARG_KEY_SENDER)
        model_params_list = msg_params.get(MyMessage.MSG_ARG_KEY_MODEL_PARAMS_LIST)
        sample_num_list = msg_params.get(MyMessage.MSG_ARG_KEY_NUM_SAMPLES)
        param_estimation_dict = msg_params.get(MyMessage.MSG_ARG_KEY_PARAMETER_ESTIMATION_DICT)

        self.aggregator.add_local_trained_result(
            sender_id - 1, model_params_list, sample_num_list, param_estimation_dict
        )

        b_all_received = self.aggregator.check_whether_all_receive()
        logging.info("b_all_received = " + str(b_all_received))

        # aggregate after receiving all models or asynchronously aggregate models
        # if b_all_received or self.args.group_comm_pattern == 'async-centralized':
        if b_all_received:

            # mix or average according to group comm pattern
            global_model_params = None
            expected_sender_id = -1     # used in async mode
            if self.args.group_comm_pattern in ['centralized', 'allreduce']:
                global_model_params = self.aggregator.aggregate()
            elif self.args.group_comm_pattern in ['decentralized']:
                global_model_params = self.aggregator.mix(self.topology_manager)
            elif self.args.group_comm_pattern in ['async-centralized']:
                expected_sender_id = np.where(self.args.ns3_time_arr == self.args.ns3_time)[0][0] + 1
                global_model_params = self.aggregator.async_aggregate(expected_sender_id-1)

            # estimate parameters
            if (
                    self.args.round_idx == 0 and self.trigger_dynamic_group_comm
                    or self.args.enable_parameter_estimation
            ):
                self.convergence_param_dict.update(self.aggregator.aggregate_estimated_params())
                # control_ratio = cal_control_ratio(self.args, self.convergence_param_dict, True)

            # start the next round
            self.args.round_idx += 1
            if 0 < self.round_num <= self.args.round_idx or \
                    self.args.enable_ns3 and 0 < self.args.time_budget <= self.args.ns3_time:
                post_complete_message_to_sweep_process(self.args)
                self.finish()
                return

            # adjust group comm round if enabled
            if self.trigger_dynamic_group_comm:
                assert self.args.enable_ns3 is True
                p = cal_mixing_consensus_speed(self.args, self.topology_manager.topology)

                config_param = "{}-{}-{}".format(self.args.group_comm_pattern, self.args.group_comm_round,
                                                 self.network.topology_manager.topology if self.network.topology_manager is not None else 'none')
                data = self.network.get_history(config_param)

                time_dict = {
                    "agg_cost": data[1].mean() / self.args.group_comm_round,
                    "mix_cost": data[2].mean(),
                    "budget": self.args.time_budget
                }

                # calculate optimal tau
                next_group_comm_round, objective_value = calculate_optimal_tau(
                    self.args, self.convergence_param_dict, time_dict, p, self.num_of_model_params
                )
                self.args.group_comm_round = next_group_comm_round
                self.topo_action_objective_list[-1][1] = objective_value

            # adjust topology if enabled
            if self.args.enable_dynamic_topo:
                adjust_topo(self.args, self.topo_action_objective_list, self.network)

            # sample clients
            sampled_group_to_client_indexes = {group_idx: [] for group_idx in range(self.args.group_num)}
            for group_comm_idx in range(self.args.group_comm_round):
                for group_idx, client_num_per_round in enumerate(self.group_to_client_num_per_round):
                    client_num_in_total = len(self.group_to_client_indexes[group_idx])
                    sampled_client_indexes = self.aggregator.client_sampling(
                        self.args.global_round_idx,
                        client_num_in_total,
                        client_num_per_round,
                    )
                    client_idx_list = []
                    for index in sampled_client_indexes:
                        client_idx = self.group_to_client_indexes[group_idx][index]
                        client_idx_list.append(client_idx)
                    sampled_group_to_client_indexes[group_idx].append(client_idx_list)
                self.args.global_round_idx += 1
            logging.info(
                "client_indexes of each group = {}".format(sampled_group_to_client_indexes)
            )

            group_to_data_size = {}
            for group_idx in range(self.args.group_num):
                data_size = 0
                for client_idx in self.group_to_client_indexes[group_idx]:
                    data_size += self.aggregator.train_data_local_num_dict[client_idx]
                group_to_data_size[group_idx] = data_size

            # distribute models
            for receiver_id in range(1, self.size):
                edge_model = None
                if self.args.group_comm_pattern in ['centralized', 'allreduce']:
                    edge_model = global_model_params
                elif self.args.group_comm_pattern in ['decentralized']:
                    edge_model = global_model_params[receiver_id - 1]
                elif self.args.group_comm_pattern in ['async-centralized']:
                    # if edge model is None, edge will not train in the coming round
                    if receiver_id == expected_sender_id:
                        edge_model = global_model_params
                    # total_sampled_data_size = 0

                self.send_message_sync_model_to_edge(
                    receiver_id,
                    edge_model,
                    sampled_group_to_client_indexes,
                    group_to_data_size,
                    receiver_id - 1
                )

            # time consumed int the coming round
            if self.args.enable_ns3:
                time_consuming_one_round(
                    self.args, self.rank, self.comm, self.network, sampled_group_to_client_indexes,
                    self.num_of_model_params * 4, list(range(1, self.size))
                )

    def send_message_init_config(self, receive_id, global_model_params, total_client_indexes,
                                 sampled_client_indexed, group_to_data_size, edge_index):
        message = Message(
            MyMessage.MSG_TYPE_C2E_INIT_CONFIG, self.get_sender_id(), receive_id
        )
        message.add_params(MyMessage.MSG_ARG_KEY_TOTAL_EDGE_CLIENTS, total_client_indexes)
        message.add_params(MyMessage.MSG_ARG_KEY_SAMPLED_EDGE_CLIENTS, sampled_client_indexed)
        message.add_params(MyMessage.MSG_ARG_KEY_GROUP_TO_DATA_SIZE, group_to_data_size)
        message.add_params(MyMessage.MSG_ARG_KEY_MODEL_PARAMS, global_model_params)
        message.add_params(MyMessage.MSG_ARG_KEY_EDGE_INDEX, edge_index)
        if self.topology_manager is not None:
            message.add_params(MyMessage.MSG_ARG_KEY_TOPOLOGY_MANAGER, self.topology_manager)
        # if enabling dynamic group communication, the default group_comm_round is set 1
        if self.trigger_dynamic_group_comm:
            message.add_params(MyMessage.MSG_ARG_KEY_GROUP_COMM_ROUND, self.args.group_comm_round)
        self.send_message(message)

    def send_message_sync_model_to_edge(
            self, receive_id, global_model_params, sampled_client_indexed, group_to_data_size, edge_index
    ):
        logging.info("send_message_sync_model_to_edge. receive_id = %d" % receive_id)
        message = Message(
            MyMessage.MSG_TYPE_C2E_SYNC_MODEL_TO_EDGE,
            self.get_sender_id(),
            receive_id,
        )
        message.add_params(MyMessage.MSG_ARG_KEY_SAMPLED_EDGE_CLIENTS, sampled_client_indexed)
        message.add_params(MyMessage.MSG_ARG_KEY_GROUP_TO_DATA_SIZE, group_to_data_size)
        message.add_params(MyMessage.MSG_ARG_KEY_MODEL_PARAMS, global_model_params)
        message.add_params(MyMessage.MSG_ARG_KEY_EDGE_INDEX, edge_index)
        # if self.topology_manager is not None:
        #     message.add_params(MyMessage.MSG_ARG_KEY_TOPOLOGY_MANAGER, self.topology_manager)
        if self.trigger_dynamic_group_comm:
            message.add_params(MyMessage.MSG_ARG_KEY_GROUP_COMM_ROUND, self.args.group_comm_round)
        if self.args.enable_dynamic_topo:
            message.add_params(MyMessage.MSG_ARG_KEY_ADJUST_TOPO, self.topo_action_objective_list[-1][0])
        self.send_message(message)

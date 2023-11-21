import logging

from .message_define import MyMessage
from ....core.distributed.fedml_comm_manager import FedMLCommManager
from ....core.distributed.communication.message import Message
from .utils import post_complete_message_to_sweep_process, time_consuming_one_round, \
    cal_mixing_consensus_speed, calculate_optimal_tau


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
            # is_preprocessed=False,
            # preprocessed_client_lists=None,
    ):
        super().__init__(args, comm, rank, size, backend)
        self.args = args
        self.aggregator = aggregator
        self.group_indexes = group_indexes
        self.group_to_client_indexes = group_to_client_indexes
        self.round_num = args.comm_round
        self.args.round_idx = 0
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

        self.convergence_param_dict = {}
        if hasattr(self.args, 'enable_ns3') and self.args.enable_ns3:
            self.args.ns3_time = 0
        self.trigger_dynamic_group_comm = True if self.args.group_comm_round <= 0 else False
        self.args.group_comm_round = 1 if self.args.group_comm_round <= 0 else self.args.group_comm_round
        self.num_of_model_params = 0

        # self.is_preprocessed = is_preprocessed
        # self.preprocessed_client_lists = preprocessed_client_lists

    def run(self):
        super().run()

    def send_init_msg(self):
        # broadcast to edge servers
        global_model_params = self.aggregator.get_global_model_params()

        # get the number of model params
        self.num_of_model_params = 0
        for k in global_model_params:
            self.num_of_model_params += global_model_params[k].numel()

        sampled_group_to_client_indexes = {}
        total_sampled_data_size = 0
        for group_idx, client_num_per_round in enumerate(self.group_to_client_num_per_round):
            client_num_in_total = len(self.group_to_client_indexes[group_idx])
            sampled_client_indexes = self.aggregator.client_sampling(
                self.args.round_idx,
                client_num_in_total,
                client_num_per_round,
            )
            sampled_group_to_client_indexes[group_idx] = []
            for index in sampled_client_indexes:
                client_idx = self.group_to_client_indexes[group_idx][index]
                sampled_group_to_client_indexes[group_idx].append(client_idx)
                total_sampled_data_size += self.aggregator.train_data_local_num_dict[client_idx]

        logging.info(
            "client_indexes of each group = {}".format(sampled_group_to_client_indexes)
        )

        for process_id in range(1, self.size):
            total_sampled_data_size = 0 if self.topology_manager is None else total_sampled_data_size
            self.send_message_init_config(
                process_id,
                global_model_params,
                self.group_to_client_indexes,
                sampled_group_to_client_indexes,
                total_sampled_data_size,
                process_id - 1
            )

        if self.args.enable_ns3:
            # time consumed int the coming round
            time_consuming_one_round(
                self.args, self.rank, self.comm, self.network, sampled_group_to_client_indexes,
                self.num_of_model_params * 4, self.topology_manager, list(range(1, self.size))
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
        if b_all_received:
            # If topology_manage is None, it is simple average. Otherwise, it is mixing between neighbours.
            global_model_params = None
            global_model_params_list = []
            if self.args.group_comm_pattern in ['centralized', 'allreduce']:
                global_model_params = self.aggregator.aggregate()
            else:
                global_model_params_list = self.aggregator.mix(self.topology_manager)

            if (
                    self.args.round_idx == 0 and self.trigger_dynamic_group_comm
                    or self.args.enable_parameter_estimation
            ):
                self.convergence_param_dict = self.aggregator.aggregate_estimated_params()

            if self.trigger_dynamic_group_comm:
                assert self.args.enable_ns3 is True
                p = cal_mixing_consensus_speed(self.args, self.topology_manager.topology)

                config_param = "{}-{}".format(self.args.group_comm_pattern, self.args.group_comm_round)
                data = self.network.get_history(config_param)

                time_dict = {
                    "agg_cost": data[1].mean() / self.args.group_comm_round,
                    "mix_cost": data[2].mean(),
                    "budget": self.args.time_budget
                }
                N_tilde = self.args.client_num_in_total  # TODO

                # calculate optimal tau
                next_group_comm_round = calculate_optimal_tau(self.args,
                                                              self.convergence_param_dict,
                                                              time_dict, p, N_tilde)
                self.args.group_comm_round = next_group_comm_round

            # start the next round
            self.args.round_idx += 1
            if self.args.round_idx == self.round_num or \
                    self.args.enable_ns3 and 0 < self.args.time_budget <= self.args.ns3_time:
                post_complete_message_to_sweep_process(self.args)
                self.finish()
                return

            sampled_group_to_client_indexes = {}
            total_sampled_data_size = 0
            for group_idx, client_num_per_round in enumerate(self.group_to_client_num_per_round):
                client_num_in_total = len(self.group_to_client_indexes[group_idx])
                sampled_client_indexes = self.aggregator.client_sampling(
                    self.args.round_idx,
                    client_num_in_total,
                    client_num_per_round,
                )
                sampled_group_to_client_indexes[group_idx] = []
                for index in sampled_client_indexes:
                    client_idx = self.group_to_client_indexes[group_idx][index]
                    sampled_group_to_client_indexes[group_idx].append(client_idx)
                    total_sampled_data_size += self.aggregator.train_data_local_num_dict[client_idx]

            logging.info(
                "client_indexes of each group = {}".format(sampled_group_to_client_indexes)
            )

            for receiver_id in range(1, self.size):
                if self.args.group_comm_pattern in ['centralized', 'allreduce']:
                    total_sampled_data_size = 0
                else:
                    global_model_params = global_model_params_list[receiver_id - 1]

                self.send_message_sync_model_to_edge(
                    receiver_id,
                    global_model_params,
                    sampled_group_to_client_indexes,
                    total_sampled_data_size,
                    receiver_id - 1
                )

            if self.args.enable_ns3:
                # time consumed int the coming round
                time_consuming_one_round(
                    self.args, self.rank, self.comm, self.network, sampled_group_to_client_indexes,
                    self.num_of_model_params * 4, self.topology_manager, list(range(1, self.size))
                )

    def send_message_init_config(self, receive_id, global_model_params, total_client_indexes,
                                 sampled_client_indexed, total_sampled_data_size, edge_index):
        message = Message(
            MyMessage.MSG_TYPE_C2E_INIT_CONFIG, self.get_sender_id(), receive_id
        )
        message.add_params(MyMessage.MSG_ARG_KEY_TOTAL_EDGE_CLIENTS, total_client_indexes)
        message.add_params(MyMessage.MSG_ARG_KEY_SAMPLED_EDGE_CLIENTS, sampled_client_indexed)
        message.add_params(MyMessage.MSG_ARG_KEY_TOTAL_SAMPLED_DATA_SIZE, total_sampled_data_size)
        message.add_params(MyMessage.MSG_ARG_KEY_MODEL_PARAMS, global_model_params)
        message.add_params(MyMessage.MSG_ARG_KEY_EDGE_INDEX, edge_index)
        if self.topology_manager is not None:
            message.add_params(MyMessage.MSG_ARG_KEY_TOPOLOGY_MANAGER, self.topology_manager)
        # if enabling dynamic group communication, the default group_comm_round is set 1
        if self.trigger_dynamic_group_comm:
            message.add_params(MyMessage.MSG_ARG_KEY_GROUP_COMM_ROUND, self.args.group_comm_round)
        self.send_message(message)

    def send_message_sync_model_to_edge(
            self, receive_id, global_model_params, sampled_client_indexed, total_sampled_data_size, edge_index
    ):
        logging.info("send_message_sync_model_to_edge. receive_id = %d" % receive_id)
        message = Message(
            MyMessage.MSG_TYPE_C2E_SYNC_MODEL_TO_EDGE,
            self.get_sender_id(),
            receive_id,
        )
        message.add_params(MyMessage.MSG_ARG_KEY_SAMPLED_EDGE_CLIENTS, sampled_client_indexed)
        message.add_params(MyMessage.MSG_ARG_KEY_TOTAL_SAMPLED_DATA_SIZE, total_sampled_data_size)
        message.add_params(MyMessage.MSG_ARG_KEY_MODEL_PARAMS, global_model_params)
        message.add_params(MyMessage.MSG_ARG_KEY_EDGE_INDEX, edge_index)
        if self.topology_manager is not None:
            message.add_params(MyMessage.MSG_ARG_KEY_TOPOLOGY_MANAGER, self.topology_manager)
        if self.trigger_dynamic_group_comm:
            message.add_params(MyMessage.MSG_ARG_KEY_GROUP_COMM_ROUND, self.args.group_comm_round)
        self.send_message(message)

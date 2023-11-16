import logging

from .message_define import MyMessage
from ....core.distributed.fedml_comm_manager import FedMLCommManager
from ....core.distributed.communication.message import Message
from .utils import post_complete_message_to_sweep_process, time_consuming_one_round


class HierFedAVGEdgeManager(FedMLCommManager):
    def __init__(
        self,
        group,
        args,
        comm=None,
        rank=0,
        size=0,
        backend="MPI",
        network=None
    ):
        super().__init__(args, comm, rank, size, backend)
        self.num_rounds = args.comm_round
        self.args.round_idx = 0
        self.group = group
        self.network = network

        if hasattr(self.args, 'enable_ns3') and self.args.enable_ns3:
            self.args.ns3_time = 0
        self.trigger_dynamic_group_comm = True if self.args.group_comm_round <= 0 else False
        self.num_of_model_params = 0

    def run(self):
        super().run()

    def register_message_receive_handlers(self):
        self.register_message_receive_handler(
            MyMessage.MSG_TYPE_C2E_INIT_CONFIG, self.handle_message_init
        )
        self.register_message_receive_handler(
            MyMessage.MSG_TYPE_C2E_SYNC_MODEL_TO_EDGE,
            self.handle_message_receive_model_from_cloud,
        )

    def handle_message_init(self, msg_params):
        global_model_params = msg_params.get(MyMessage.MSG_ARG_KEY_MODEL_PARAMS)
        total_client_indexes = msg_params.get(MyMessage.MSG_ARG_KEY_TOTAL_EDGE_CLIENTS)
        sampled_client_indexes = msg_params.get(MyMessage.MSG_ARG_KEY_SAMPLED_EDGE_CLIENTS)
        total_sampled_data_size = msg_params.get(MyMessage.MSG_ARG_KEY_TOTAL_SAMPLED_DATA_SIZE)
        edge_index = msg_params.get(MyMessage.MSG_ARG_KEY_EDGE_INDEX)
        topology_manager = msg_params.get(MyMessage.MSG_ARG_KEY_TOPOLOGY_MANAGER)
        group_comm_round = msg_params.get(MyMessage.MSG_ARG_KEY_GROUP_COMM_ROUND)

        self.group.setup_clients(total_client_indexes[edge_index])
        self.args.round_idx = 0

        # get the number of model params
        self.num_of_model_params = 0
        for k in global_model_params:
            self.num_of_model_params += global_model_params[k].numel()

        if group_comm_round is not None:
            self.args.group_comm_round = group_comm_round

        if self.args.enable_ns3:
            # time consumed in the current round
            time_consuming_one_round(
                self.args, self.rank, self.comm, self.network, sampled_client_indexes,
                self.num_of_model_params * 4, topology_manager, list(range(1, self.size))
            )

        is_estimate = False
        if (
                self.args.round_idx == 0 and self.trigger_dynamic_group_comm
                or self.args.enable_parameter_estimation
        ):
            is_estimate = True
        w_group_list, sample_num_list, param_estimation_dict = self.group.train(self.args.round_idx,
                                                                                global_model_params,
                                                                                sampled_client_indexes[edge_index],
                                                                                total_sampled_data_size,
                                                                                is_estimate)


        self.send_model_to_cloud(0, w_group_list, sample_num_list, param_estimation_dict)

    def handle_message_receive_model_from_cloud(self, msg_params):
        logging.info("handle_message_receive_model_from_cloud.")
        sampled_client_indexes = msg_params.get(MyMessage.MSG_ARG_KEY_SAMPLED_EDGE_CLIENTS)
        total_sampled_data_size = msg_params.get(MyMessage.MSG_ARG_KEY_TOTAL_SAMPLED_DATA_SIZE)
        global_model_params = msg_params.get(MyMessage.MSG_ARG_KEY_MODEL_PARAMS)
        edge_index = msg_params.get(MyMessage.MSG_ARG_KEY_EDGE_INDEX)
        topology_manager = msg_params.get(MyMessage.MSG_ARG_KEY_TOPOLOGY_MANAGER)
        group_comm_round = msg_params.get(MyMessage.MSG_ARG_KEY_GROUP_COMM_ROUND)

        self.args.round_idx += 1

        if group_comm_round is not None:
            self.args.group_comm_round = group_comm_round

        if self.args.enable_ns3:
            time_consuming_one_round(
                self.args, self.rank, self.comm, self.network, sampled_client_indexes,
                self.num_of_model_params * 4, topology_manager, list(range(1, self.size))
            )

        is_estimate = False
        if (
                self.args.round_idx == 0 and self.trigger_dynamic_group_comm
                or self.args.enable_parameter_estimation
        ):
            is_estimate = True
        w_group_list, sample_num_list, param_estimation_dict = \
            self.group.train(self.args.round_idx, global_model_params, sampled_client_indexes[edge_index],
                             total_sampled_data_size, is_estimate)
        self.send_model_to_cloud(0, w_group_list, sample_num_list, param_estimation_dict)

        if self.args.round_idx == self.num_rounds:
            post_complete_message_to_sweep_process(self.args)
            self.finish()

    def send_model_to_cloud(self, receive_id, w_group_list, edge_sample_num, param_estimation_dict):
        message = Message(
            MyMessage.MSG_TYPE_E2C_SEND_MODEL_TO_CLOUD,
            self.get_sender_id(),
            receive_id,
        )
        message.add_params(MyMessage.MSG_ARG_KEY_MODEL_PARAMS_LIST, w_group_list)
        message.add_params(MyMessage.MSG_ARG_KEY_NUM_SAMPLES, edge_sample_num)
        message.add_params(MyMessage.MSG_ARG_KEY_PARAMETER_ESTIMATION_DICT, param_estimation_dict)
        self.send_message(message)

import logging

from .HierClient import HFLClient
from ...sp.fedavg.fedavg_api import FedAvgAPI
from .utils import agg_parameter_estimation


class HierGroup(FedAvgAPI):
    def __init__(
        self,
        idx,
        train_data_local_dict,
        test_data_local_dict,
        train_data_local_num_dict,
        args,
        device,
        model,
        model_trainer,
    ):
        self.idx = idx
        self.args = args
        self.device = device
        self.client_dict = {}
        self.train_data_local_num_dict = train_data_local_num_dict
        self.train_data_local_dict = train_data_local_dict
        self.test_data_local_dict = test_data_local_dict
        self.model = model
        self.model_trainer = model_trainer
        self.args = args

    def setup_clients(self, total_client_indexes):
        self.client_dict = {}
        for client_idx in total_client_indexes:
            self.client_dict[client_idx] = HFLClient(
                client_idx,
                self.train_data_local_dict[client_idx],
                self.test_data_local_dict[client_idx],
                self.train_data_local_num_dict[client_idx],
                self.args,
                self.device,
                self.model,
                self.model_trainer,
            )

    def get_sample_number(self, sampled_client_indexes):
        self.group_sample_number = 0
        for client_idx in sampled_client_indexes:
            self.group_sample_number += self.train_data_local_num_dict[client_idx]
        return self.group_sample_number

    def train(self, round_idx=0, w=None, sampled_client_indexes=None, group_to_data_size=None, is_estimate=False):
        if is_estimate:
            # param_estimation_dict = self._estimate(w, sampled_client_indexes)
            param_estimation_dict = self._estimate(w)
        else:
            param_estimation_dict = {}

        data_size_dict = {}
        for group_round_idx in range(self.args.group_comm_round):
            data_size = 0
            for group_idx, client_idx_list in sampled_client_indexes.items():
                data_size += self.get_sample_number(client_idx_list[group_round_idx])
            data_size_dict[group_round_idx] = data_size

        w_group = w
        w_group_list = []
        sample_num_list = []
        sampled_client_indexes = sampled_client_indexes[self.idx]
        for group_round_idx in range(self.args.group_comm_round):
            logging.info("Group ID : {} / Group Communication Round : {}".format(self.idx, group_round_idx))
            w_locals = []

            global_round_idx = (
                    round_idx * self.args.group_comm_round
                    + group_round_idx
            )

            # train each client
            sampled_client_list = [self.client_dict[client_idx] for client_idx in sampled_client_indexes[group_round_idx]]
            group_client_num = len(sampled_client_list)
            for client in sampled_client_list:
                if group_to_data_size is not None:
                    # TODO: how to aggregate model between PSes in case of non-iid data
                    # scaled_loss_factor = (
                    #         self.args.group_num * group_to_data_size[self.idx] / sum(group_to_data_size.values())
                    # )
                    # scaled_loss_factor = self.args.group_num * self.args.client_num_per_round * client.local_sample_number / data_size_dict[group_round_idx]
                    scaled_loss_factor = self.args.group_num * group_client_num * client.local_sample_number / data_size_dict[group_round_idx]
                    scaled_loss_factor = min(scaled_loss_factor, 1)
                    # scaled_loss_factor = 1
                    w_local = client.train(w_group, scaled_loss_factor)
                else:
                    w_local = client.train(w_group)
                w_locals.append((client.get_sample_number(), w_local))

            # aggregate local weights
            # TODO: debug
            w_group_list.append((global_round_idx, self._aggregate_noniid_avg(w_locals)))
            # w_group_list.append((global_round_idx, self._aggregate(w_locals)))
            sample_num_list.append(self.get_sample_number(sampled_client_indexes[group_round_idx]))

            # update the group weight
            w_group = w_group_list[-1][1]
        return w_group_list, sample_num_list, param_estimation_dict

    def _estimate(self, w):
        # use total clients to estimate, this is more accurate
        sampled_client_list = [self.client_dict[client_idx] for client_idx in self.client_dict]
        group_client_num = len(sampled_client_list)
        total_data_size = sum(self.train_data_local_num_dict.values())
        # sampled_client_list = [self.client_dict[client_idx] for client_idx in sampled_client_indexes[self.idx][0]]
        w_group = w
        param_estimation_dict = {}

        for idx, client in enumerate(sampled_client_list):
            # scaled_loss_factor = self.args.group_num * self.args.client_num_per_round * client.local_sample_number / \
            #                      data_size_dict[group_round_idx]
            scaled_loss_factor = self.args.group_num * group_client_num * client.local_sample_number / total_data_size
            param_rs = client.estimate_parameters(w_group, scaled_loss_factor=scaled_loss_factor)
            param_estimation_dict[idx] = param_rs

        agg_param_estimation_dict = agg_parameter_estimation(self.args, param_estimation_dict, 'gamma')

        return agg_param_estimation_dict
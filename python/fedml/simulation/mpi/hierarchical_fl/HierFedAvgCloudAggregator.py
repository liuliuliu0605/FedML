import copy
import logging
import random
import time
import numpy as np
import torch
import wandb

from ....core.security.fedml_attacker import FedMLAttacker
from ....core.security.fedml_defender import FedMLDefender
from .utils import cal_mixing_consensus_speed, agg_parameter_estimation, calculate_optimal_tau


class HierFedAVGCloudAggregator(object):
    def __init__(
        self,
        train_global,
        test_global,
        all_train_data_num,
        train_data_local_dict,
        test_data_local_dict,
        train_data_local_num_dict,
        worker_num,
        device,
        args,
        server_aggregator,
    ):
        self.aggregator = server_aggregator
        self.args = args
        self.train_global = train_global
        self.test_global = test_global
        self.val_global = self._generate_validation_set()
        self.all_train_data_num = all_train_data_num

        self.train_data_local_dict = train_data_local_dict
        self.test_data_local_dict = test_data_local_dict
        self.train_data_local_num_dict = train_data_local_num_dict

        self.worker_num = worker_num
        self.device = device
        self.model_dict = dict()
        self.sample_num_dict = dict()
        self.param_estimation_dict = dict()
        self.flag_client_model_uploaded_dict = dict()
        for idx in range(self.worker_num):
            self.flag_client_model_uploaded_dict[idx] = False

        # used in async model
        if args.group_comm_pattern == 'async-centralized':
            self.num_of_partial_updates = [0 for i in range(self.worker_num)]
            self.init_model_params = copy.deepcopy(self.get_global_model_params())

    def get_global_model_params(self):
        return self.aggregator.get_model_params()

    def set_global_model_params(self, model_parameters):
        self.aggregator.set_model_params(model_parameters)

    def add_local_trained_result(self, index, model_params_list, sample_num, param_estimation_dict):
        logging.info("add_model. index = %d" % index)
        self.model_dict[index] = model_params_list if model_params_list is not None else self.model_dict[index]
        self.sample_num_dict[index] = sample_num if sample_num is not None else self.sample_num_dict[index]
        self.param_estimation_dict[index] = param_estimation_dict if param_estimation_dict is not None else self.param_estimation_dict[index]
        self.flag_client_model_uploaded_dict[index] = True

    def check_whether_all_receive(self):
        logging.debug("worker_num = {}".format(self.worker_num))
        for idx in range(self.worker_num):
            if not self.flag_client_model_uploaded_dict[idx]:
                return False
        for idx in range(self.worker_num):
            self.flag_client_model_uploaded_dict[idx] = False
        return True

    def aggregate_estimated_params(self):
        log_wandb = True if hasattr(self.args, 'enable_wandb') and self.args.enable_wandb else False
        agg_param_estimation_dict = agg_parameter_estimation(self.args, self.param_estimation_dict, 'psi', log_wandb)
        return agg_param_estimation_dict

    def aggregate(self):
        # Edge server may conduct partial aggregation multiple times, so cloud server will receive a model list
        group_comm_round = len(self.sample_num_dict[0])

        for group_round_idx in range(group_comm_round):
            model_list = []
            global_round_idx = self.model_dict[0][group_round_idx][0]

            for idx in range(0, self.worker_num):
                model_list.append((1, self.model_dict[idx][group_round_idx][1]))
                # model_list.append((self.sample_num_dict[idx][group_round_idx],
                #                    self.model_dict[idx][group_round_idx][1]))

            averaged_params = self._fedavg_aggregation_(model_list)
            self.set_global_model_params(averaged_params)
            self.test_on_cloud_for_all_clients(global_round_idx)

        if FedMLAttacker.get_instance().is_model_attack():
            model_list = FedMLAttacker.get_instance().attack_model(raw_client_grad_list=model_list, extra_auxiliary_info=None)

        if FedMLDefender.get_instance().is_defense_enabled():
            # todo: update extra_auxiliary_info according to defense type
            averaged_params = FedMLDefender.get_instance().defend(
                raw_client_grad_list=model_list,
                base_aggregation_func=self._fedavg_aggregation_,
                extra_auxiliary_info=self.get_global_model_params(),
            )
        else:
            averaged_params = self._fedavg_aggregation_(model_list)

        # update the global model which is cached in the cloud
        self.set_global_model_params(averaged_params)

        return averaged_params

    def mix(self, topology_manager):
        # edge server may conduct partial aggregation multiple times, so cloud server will receive a model list
        group_comm_round = len(self.sample_num_dict[0])

        for group_round_idx in range(group_comm_round):
            model_list = []
            global_round_idx = self.model_dict[0][group_round_idx][0]

            for idx in range(0, self.worker_num):
                model_list.append((1, self.model_dict[idx][group_round_idx][1]))

            mixed_params_list = []
            for idx in range(self.worker_num):
                mixed_params_list.append((1, self._pfedavg_mixing_(model_list,
                                                                   topology_manager.get_in_neighbor_weights(idx))))

            averaged_params = self._fedavg_aggregation_(mixed_params_list)
            self.set_global_model_params(averaged_params)
            self.test_on_cloud_for_all_clients(global_round_idx)

        return [mixed_params for _, mixed_params in mixed_params_list]

    def async_aggregate(self, index):
        group_comm_round = len(self.sample_num_dict[index])
        self.num_of_partial_updates[index] += 1
        sorted_edge_id_list = np.argsort(self.num_of_partial_updates)
        weight_list = [0 for _ in range(self.worker_num)]

        for i, _id in enumerate(sorted_edge_id_list):
            weight_list[_id] = self.num_of_partial_updates[self.worker_num - i - 1]

        for group_round_idx in range(group_comm_round):
            global_round_idx = self.args.round_idx * self.args.group_comm_round + group_round_idx

            model_list = []
            for idx in range(0, self.worker_num):
                if self.num_of_partial_updates[idx] == 0:
                    model = self.init_model_params
                else:
                    model = self.model_dict[idx][group_round_idx][1]
                model_list.append((weight_list[idx], model))

            averaged_params = self._fedavg_aggregation_(model_list)
            # averaged_params = self.model_dict[index][group_round_idx][1]
            self.set_global_model_params(averaged_params)
            self.test_on_cloud_for_all_clients(global_round_idx)

        return averaged_params

    def _fedavg_aggregation_(self, model_list):
        training_num = 0
        for i in range(0, len(model_list)):
            local_sample_number, local_model_params = model_list[i]
            training_num += local_sample_number

        (num0, averaged_params) = model_list[0]
        averaged_params = copy.deepcopy(averaged_params)

        for k in averaged_params.keys():
            for i in range(0, len(model_list)):
                local_sample_number, local_model_params = model_list[i]
                if i == 0:
                    averaged_params[k] = (
                        local_model_params[k] * local_sample_number / training_num
                    )
                else:
                    averaged_params[k] += (
                        local_model_params[k] * local_sample_number / training_num
                    )
        return averaged_params

    def _pfedavg_mixing_(self, model_list, neighbor_topo_weight_list):

        (num0, averaged_params) = model_list[0]
        averaged_params = copy.deepcopy(averaged_params)

        for k in averaged_params.keys():
            for i in range(0, len(model_list)):
                local_sample_number, local_model_params = model_list[i]
                topo_weight = neighbor_topo_weight_list[i]
                if i == 0:
                    averaged_params[k] = (
                            local_model_params[k] * topo_weight
                    )
                else:
                    averaged_params[k] += (
                            local_model_params[k] * topo_weight
                    )

        return averaged_params

    def client_sampling(self, round_idx, client_num_in_total, client_num_per_round):
        if client_num_in_total == client_num_per_round:
            client_indexes = [
                client_index for client_index in range(client_num_in_total)
            ]
        else:
            num_clients = min(client_num_per_round, client_num_in_total)
            np.random.seed(
                self.args.random_seed + round_idx
            )  # make sure for each comparison, we are selecting the same clients each round
            client_indexes = np.random.choice(
                range(client_num_in_total), num_clients, replace=False
            )
        return client_indexes

    def _generate_validation_set(self, num_samples=10000):
        if self.args.dataset.startswith("stackoverflow"):
            test_data_num = len(self.test_global.dataset)
            sample_indices = random.sample(
                range(test_data_num), min(num_samples, test_data_num)
            )
            subset = torch.utils.data.Subset(self.test_global.dataset, sample_indices)
            sample_testset = torch.utils.data.DataLoader(
                subset, batch_size=self.args.batch_size
            )
            return sample_testset
        else:
            return self.test_global

    def test_on_cloud_for_all_clients(self, global_round_idx):
        if self.aggregator.test_all(
            self.train_data_local_dict,
            self.test_data_local_dict,
            self.device,
            self.args,
        ):
            return

        if (
                global_round_idx % self.args.frequency_of_the_test == 0
                or global_round_idx == self.args.comm_round * self.args.group_comm_round - 1
                or self.args.enable_ns3 and 0 < self.args.time_budget <= self.args.ns3_time
        ):

            logging.info("################test_on_cloud_for_all_clients : {}".format(global_round_idx))

            # We may want to test the intermediate results of partial aggregated models, so we play a trick and let
            # args.round_idx be total number of partial aggregated times

            round_idx = self.args.round_idx
            self.args.round_idx = global_round_idx

            if global_round_idx == self.args.comm_round - 1:
                # we allow to return four metrics, such as accuracy, AUC, loss, etc.
                metric_result_in_current_round = self.aggregator.test(self.test_global, self.device, self.args)
            else:
                metric_result_in_current_round = self.aggregator.test(self.val_global, self.device, self.args)

            self.args.round_idx = round_idx

            if self.args.enable_wandb and self.args.enable_ns3:
                wandb.log({"Test/Acc": metric_result_in_current_round[0], "time": self.args.ns3_time})
                wandb.log({"Test/Loss": metric_result_in_current_round[1], "time": self.args.ns3_time})

            logging.info("metric_result_in_current_round = {}".format(metric_result_in_current_round))

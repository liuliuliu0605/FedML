import copy
import time
import numpy as np
import torch
import torch.nn as nn

from ...sp.fedavg.client import Client


class HFLClient(Client):
    def __init__(self, client_idx, local_training_data, local_test_data, local_sample_number, args, device, model,
                 model_trainer):

        super().__init__(client_idx, local_training_data, local_test_data, local_sample_number, args, device,
                         model_trainer)
        self.client_idx = client_idx
        self.local_training_data = local_training_data
        self.local_test_data = local_test_data
        self.local_sample_number = local_sample_number

        self.args = args
        self.device = device
        self.model = model
        self.model_trainer = model_trainer
        self.criterion = nn.CrossEntropyLoss().to(device)

    def train(self, w, scaled_loss_factor=1.0):
        self.model.load_state_dict(w)
        self.model.to(self.device)

        # scaled_loss_factor = min(scaled_loss_factor, 1.0)
        if self.args.client_optimizer == "sgd":
            optimizer = torch.optim.SGD(self.model.parameters(), lr=self.args.learning_rate * scaled_loss_factor)
        else:
            optimizer = torch.optim.Adam(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                lr=self.args.learning_rate * scaled_loss_factor,
                weight_decay=self.args.weight_decay,
                amsgrad=True,
            )

        for epoch in range(self.args.epochs):
            for x, labels in self.local_training_data:
                x, labels = x.to(self.device), labels.to(self.device)
                self.model.zero_grad()
                log_probs = self.model(x)
                loss = self.criterion(log_probs, labels)  # pylint: disable=E1102
                loss.backward()
                optimizer.step()

        return copy.deepcopy(self.model.cpu().state_dict())

    def estimate_parameters(self, w, scaled_loss_factor=1.0):
        self.model.load_state_dict(w)
        self.model.to(self.device)

        if self.args.client_optimizer == "sgd":
            optimizer = torch.optim.SGD(self.model.parameters(), lr=self.args.learning_rate)
            # optimizer = torch.optim.SGD(self.model.parameters(), lr=1e-10)
        else:
            optimizer = torch.optim.Adam(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                lr=self.args.learning_rate,
                # lr=1e-10,
                weight_decay=self.args.weight_decay,
                amsgrad=True,
            )

        batch_num = len(self.local_training_data)

        # calculate the initial grad and the variance of stochastic gradients
        grad = {}
        grad_square = 0
        loss_value = 0
        local_update_time = 0
        num_of_local_updates = 0
        for x, labels in self.local_training_data:
            start_time = time.time()
            x, labels = x.to(self.device), labels.to(self.device)
            self.model.zero_grad()
            log_probs = self.model(x)
            loss = self.criterion(log_probs, labels) * scaled_loss_factor
            loss_value += loss.item() / batch_num
            loss.backward()
            end_time = time.time()
            time_consumed = end_time - start_time
            local_update_time += time_consumed
            num_of_local_updates += 1

            for name, param in self.model.named_parameters():
                grad_square += (param.grad ** 2).sum().item() / batch_num
                if name not in grad:
                    grad[name] = copy.deepcopy(param.grad.cpu().numpy()) / batch_num
                else:
                    grad[name] += param.grad.cpu().numpy() / batch_num

        sigma = grad_square
        for name, param in self.model.named_parameters():
            sigma -= (grad[name] ** 2).sum().item()
        sigma = max(sigma, 0)

        # forward according to the last batch, it is used for estimating L in the following step
        optimizer.step()

        # calculate L
        L = 0
        for x, labels in self.local_training_data:
            x, labels = x.to(self.device), labels.to(self.device)
            self.model.zero_grad()
            log_probs = self.model(x)
            loss = self.criterion(log_probs, labels)
            loss *= scaled_loss_factor
            loss.backward()

            delta_grad_square, delta_model_square = 0, 0
            for name, param in self.model.named_parameters():
                delta_grad = grad[name] - param.grad.cpu().numpy()
                delta_model = w[name] - param.detach().cpu().numpy()
                delta_grad_square += (delta_grad**2).sum().item()
                delta_model_square += (delta_model**2).sum().item()
                L = np.sqrt(delta_grad_square / delta_model_square)

        return {
            'sigma': sigma,
            'L': L,
            'loss': loss_value,
            'num_of_local_updates_one_epoch': num_of_local_updates,
            'time_used_for_local_updates_one_epoch': local_update_time,
            'grad': grad,  # dict
        }

    # def estimate_parameters(self, w, scaled_loss_factor=1.0):
    #     self.model.load_state_dict(w)
    #     self.model.to(self.device)
    #
    #     if self.args.client_optimizer == "sgd":
    #         optimizer = torch.optim.SGD(self.model.parameters(), lr=self.args.learning_rate)
    #         # optimizer = torch.optim.SGD(self.model.parameters(), lr=1e-10)
    #     else:
    #         optimizer = torch.optim.Adam(
    #             filter(lambda p: p.requires_grad, self.model.parameters()),
    #             lr=self.args.learning_rate,
    #             # lr=1e-10,
    #             weight_decay=self.args.weight_decay,
    #             amsgrad=True,
    #         )
    #
    #     batch_num = len(self.local_training_data)
    #
    #     # calculate the initial grad and the variance of stochastic gradients
    #     grad = {}
    #     grad_square = 0
    #     loss_value = 0
    #     for x, labels in self.local_training_data:
    #         x, labels = x.to(self.device), labels.to(self.device)
    #         self.model.zero_grad()
    #         log_probs = self.model(x)
    #         loss = self.criterion(log_probs, labels) * scaled_loss_factor
    #         loss_value += loss.item() / batch_num
    #         loss.backward()
    #
    #         for name, param in self.model.named_parameters():
    #             grad_square += (param.grad ** 2).sum().item() / batch_num
    #             if name not in grad:
    #                 grad[name] = copy.deepcopy(param.grad.cpu().numpy()) / batch_num
    #             else:
    #                 grad[name] += param.grad.cpu().numpy() / batch_num
    #
    #     sigma = grad_square
    #     for name, param in self.model.named_parameters():
    #         sigma -= (grad[name] ** 2).sum().item()
    #
    #     # forward according to the last batch, it is used for estimating L in the following steps
    #     optimizer.step()
    #
    #     # calculate zeta and L
    #     self.model.to(self.device)
    #     cum_grad_delta = {}
    #     cum_grad_delta_square = 0
    #     num_of_local_updates = 0
    #     local_update_time = 0
    #     L = 0
    #     for epoch in range(self.args.epochs):
    #         for x, labels in self.local_training_data:
    #             start_time = time.time()
    #
    #             x, labels = x.to(self.device), labels.to(self.device)
    #             self.model.zero_grad()
    #             log_probs = self.model(x)
    #             loss = self.criterion(log_probs, labels)
    #             loss *= scaled_loss_factor
    #             loss.backward()
    #
    #             end_time = time.time()
    #             time_consumed = end_time - start_time
    #             local_update_time += time_consumed
    #
    #             delta_grad_square, delta_model_square = 0, 0
    #             for name, param in self.model.named_parameters():
    #                 delta_grad = grad[name] - param.grad.cpu().numpy()
    #                 delta_model = w[name] - param.detach().cpu().numpy()
    #                 if name not in cum_grad_delta:
    #                     cum_grad_delta[name] = copy.deepcopy(delta_grad)
    #                 else:
    #                     cum_grad_delta[name] += delta_grad
    #                 cum_grad_delta_square += (delta_grad ** 2).sum().item()
    #                 delta_grad_square += (delta_grad**2).sum().item()
    #                 delta_model_square += (delta_model**2).sum().item()
    #
    #             # L += np.sqrt(delta_grad_square / delta_model_square)
    #             L = max(L, np.sqrt(delta_grad_square / delta_model_square))
    #             num_of_local_updates += 1
    #
    #             optimizer.step()
    #
    #     cum_grad_delta_square *= num_of_local_updates
    #
    #     return {
    #         'sigma': max(sigma, 0),
    #         'L': L,
    #         'grad': grad,  # dict
    #         'K': num_of_local_updates,
    #         'loss': loss_value,
    #         'local_update_time': local_update_time,
    #         'cum_grad_delta': cum_grad_delta,  # dict
    #         'cum_grad_delta_square': cum_grad_delta_square
    #     }

    # def estimate_parameters(self, w, scaled_loss_factor=1.0):
    #     self.model.load_state_dict(w)
    #     self.model.to(self.device)
    #
    #     if self.args.client_optimizer == "sgd":
    #         # optimizer = torch.optim.SGD(self.model.parameters(), lr=self.args.learning_rate)
    #         optimizer = torch.optim.SGD(self.model.parameters(), lr=1e-5)
    #     else:
    #         optimizer = torch.optim.Adam(
    #             filter(lambda p: p.requires_grad, self.model.parameters()),
    #             # lr=self.args.learning_rate,
    #             lr=1e-5,
    #             weight_decay=self.args.weight_decay,
    #             amsgrad=True,
    #         )
    #
    #     batch_num = len(self.local_training_data)
    #
    #     # calculate full gradient of the first local update
    #     loss_value = 0
    #     grad = {}
    #     self.model.zero_grad()
    #     for x, labels in self.local_training_data:
    #         x, labels = x.to(self.device), labels.to(self.device)
    #         log_probs = self.model(x)
    #         loss = self.criterion(log_probs, labels) * scaled_loss_factor  # rewrite loss function
    #         loss /= batch_num
    #         loss_value += loss.item()
    #         loss.backward()
    #
    #     for name, param in self.model.named_parameters():
    #         grad[name] = copy.deepcopy(param.grad.cpu().numpy())
    #
    #     optimizer.step()
    #
    #     # calculate the variance of stochastic gradients
    #     grad2 = {}
    #     grad_square = 0
    #     for x, labels in self.local_training_data:
    #         x, labels = x.to(self.device), labels.to(self.device)
    #         log_probs = self.model(x)
    #         loss = self.criterion(log_probs, labels) * scaled_loss_factor
    #         self.model.zero_grad()
    #         loss.backward()
    #
    #         for name, param in self.model.named_parameters():
    #             grad_square += (param.grad ** 2).sum().item() / batch_num
    #             if name not in grad2:
    #                 grad2[name] = copy.deepcopy(param.grad.cpu().numpy()) / batch_num
    #             else:
    #                 grad2[name] += param.grad.cpu().numpy() / batch_num
    #
    #     sigma = grad_square
    #     grad_delta, model_delta = 0, 0
    #     for name, param in self.model.named_parameters():
    #         sigma -= (grad2[name]**2).sum().item()
    #         grad_delta += np.linalg.norm(grad[name] - grad2[name])
    #         model_delta += np.linalg.norm(w[name] - param.detach().cpu().numpy())
    #     L = grad_delta / model_delta
    #
    #     # calculate zeta
    #     self.model.load_state_dict(w)
    #     self.model.to(self.device)
    #     cum_grad_delta = {}
    #     cum_grad_delta_square = 0
    #     num_of_local_updates = 0
    #     local_update_time = 0
    #     for epoch in range(self.args.epochs):
    #         for x, labels in self.local_training_data:
    #             start_time = time.time()
    #             x, labels = x.to(self.device), labels.to(self.device)
    #             self.model.zero_grad()
    #             log_probs = self.model(x)
    #             loss = self.criterion(log_probs, labels)
    #             loss *= scaled_loss_factor
    #             loss.backward()
    #             end_time = time.time()
    #             time_consumed = end_time - start_time
    #             local_update_time += time_consumed
    #
    #             for name, param in self.model.named_parameters():
    #                 delta_grad = grad[name] - param.grad.cpu().numpy()
    #                 cum_grad_delta_square += (delta_grad**2).sum().item()
    #                 if name not in cum_grad_delta:
    #                     cum_grad_delta[name] = copy.deepcopy(delta_grad)
    #                 else:
    #                     cum_grad_delta[name] += delta_grad
    #
    #             optimizer.step()
    #
    #             num_of_local_updates += 1
    #     cum_grad_delta_square *= num_of_local_updates
    #
    #     return {
    #         'sigma': max(sigma, 0),
    #         'L': L,
    #         'grad': grad,  # dict
    #         'K': num_of_local_updates,
    #         'loss': loss_value,
    #         'local_update_time': local_update_time,
    #         'cum_grad_delta': cum_grad_delta,  # dict
    #         'cum_grad_delta_square': cum_grad_delta_square
    #     }

        # for name, param in self.model.named_parameters():
        #     grad[name] *= scaled_loss_factor
        #     cum_grad_delta[name] *= scaled_loss_factor

        # return {
        #     'sigma': max(sigma, 0) * scaled_loss_factor**2,
        #     'L': L * scaled_loss_factor,
        #     'grad': grad,   # dict
        #     'K': num_of_local_updates * scaled_loss_factor,
        #     'loss': loss_value * scaled_loss_factor,
        #     'local_update_time': local_update_time * scaled_loss_factor,
        #     'cum_grad_delta': cum_grad_delta,   # dict
        #     'cum_grad_delta_square': cum_grad_delta_square * scaled_loss_factor **2
        # }

    # def estimate_parameters(self, w, scaled_loss_factor=1.0):
    #     self.model.load_state_dict(w)
    #     self.model.to(self.device)
    #
    #     scaled_loss_factor = min(scaled_loss_factor, 1.0)
    #     if self.args.client_optimizer == "sgd":
    #         optimizer = torch.optim.SGD(self.model.parameters(), lr=self.args.learning_rate)
    #     else:
    #         optimizer = torch.optim.Adam(
    #             filter(lambda p: p.requires_grad, self.model.parameters()),
    #             lr=self.args.learning_rate,
    #             weight_decay=self.args.weight_decay,
    #             amsgrad=True,
    #         )
    #
    #     batch_num = len(self.local_training_data)
    #     grad = {}
    #     loss_value = 0
    #
    #     # calculate full gradient of the first local update
    #     self.model.zero_grad()
    #     for x, labels in self.local_training_data:
    #         x, labels = x.to(self.device), labels.to(self.device)
    #         log_probs = self.model(x)
    #         loss = self.criterion(log_probs, labels) * scaled_loss_factor  # rewrite loss function
    #         loss /= batch_num
    #         loss_value += loss.item()
    #         loss.backward()
    #
    #     for name, param in self.model.named_parameters():
    #         grad[name] = copy.deepcopy(param.grad.cpu().numpy())
    #
    #     optimizer.step()
    #
    #     # calculate the variance of stochastic gradients
    #     grad2 = {}
    #     grad_square = 0
    #     for x, labels in self.local_training_data:
    #         x, labels = x.to(self.device), labels.to(self.device)
    #         self.model.zero_grad()
    #         log_probs = self.model(x)
    #         loss = self.criterion(log_probs, labels) * scaled_loss_factor
    #         loss.backward()
    #
    #         for name, param in self.model.named_parameters():
    #             grad_square += (param.grad ** 2).sum().item() / batch_num
    #             if name not in grad2:
    #                 grad2[name] = copy.deepcopy(param.grad.cpu().numpy()) / batch_num
    #             else:
    #                 grad2[name] += param.grad.cpu().numpy() / batch_num
    #
    #     sigma = grad_square
    #     grad_delta, model_delta = 0, 0
    #     for name, param in self.model.named_parameters():
    #         sigma -= (grad2[name]**2).sum().item()
    #         grad_delta += ((grad[name] - grad2[name])**2).sum().item()
    #         model_delta += ((w[name] - param.detach().cpu().numpy())**2).sum().item()
    #     L = grad_delta / model_delta
    #
    #     # calculate zeta
    #     cum_grad_delta = {}
    #     cum_grad_delta_square = 0
    #     num_of_local_updates = 0
    #     local_update_time = 0
    #     for epoch in range(self.args.epochs):
    #         for x, labels in self.local_training_data:
    #             start_time = time.time()
    #             x, labels = x.to(self.device), labels.to(self.device)
    #             self.model.zero_grad()
    #             log_probs = self.model(x)
    #             loss = self.criterion(log_probs, labels)  # pylint: disable=E1102
    #             loss.backward()
    #             end_time = time.time()
    #             time_consumed = end_time - start_time
    #             local_update_time += time_consumed
    #
    #             for name, param in self.model.named_parameters():
    #                 delta_grad = grad[name] - param.grad.cpu().numpy()
    #                 cum_grad_delta_square += (delta_grad**2).sum().item()
    #                 if name not in cum_grad_delta:
    #                     cum_grad_delta[name] = copy.deepcopy(delta_grad)
    #                 else:
    #                     cum_grad_delta[name] += delta_grad
    #
    #             optimizer.step()
    #
    #             num_of_local_updates += 1
    #     cum_grad_delta_square *= num_of_local_updates
    #
    #     return {
    #         'sigma': max(sigma, 0),
    #         'L': L,
    #         'grad': grad,
    #         'K': num_of_local_updates,
    #         'loss': loss_value,
    #         'local_update_time': local_update_time,
    #         'cum_grad_delta': cum_grad_delta,
    #         'cum_grad_delta_square': cum_grad_delta_square
    #     }

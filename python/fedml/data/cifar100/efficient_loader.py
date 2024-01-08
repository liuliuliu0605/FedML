import logging

import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR100

from .without_reload import CIFAR100_truncated, CIFAR100_truncated_WO_reload


# generate the non-IID distribution for all methods
def read_data_distribution(filename="./data_preprocessing/non-iid-distribution/CIFAR100/distribution.txt",):
    distribution = {}
    with open(filename, "r") as data:
        for x in data.readlines():
            if "{" != x[0] and "}" != x[0]:
                tmp = x.split(":")
                if "{" == tmp[1].strip():
                    first_level_key = int(tmp[0])
                    distribution[first_level_key] = {}
                else:
                    second_level_key = int(tmp[0])
                    distribution[first_level_key][second_level_key] = int(tmp[1].strip().replace(",", ""))
    return distribution


def read_net_dataidx_map(filename="./data_preprocessing/non-iid-distribution/CIFAR100/net_dataidx_map.txt",):
    net_dataidx_map = {}
    with open(filename, "r") as data:
        for x in data.readlines():
            if "{" != x[0] and "}" != x[0] and "]" != x[0]:
                tmp = x.split(":")
                if "[" == tmp[-1].strip():
                    key = int(tmp[0])
                    net_dataidx_map[key] = []
                else:
                    tmp_array = x.split(",")
                    net_dataidx_map[key] = [int(i.strip()) for i in tmp_array]
    return net_dataidx_map


def record_net_data_stats(y_train, net_dataidx_map):
    net_cls_counts = {}

    for net_i, dataidx in net_dataidx_map.items():
        unq, unq_cnt = np.unique(y_train[dataidx], return_counts=True)
        tmp = {unq[i]: unq_cnt[i] for i in range(len(unq))}
        net_cls_counts[net_i] = tmp
    logging.debug("Data statistics: %s" % str(net_cls_counts))
    return net_cls_counts


class Cutout(object):
    def __init__(self, length):
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1:y2, x1:x2] = 0.0
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask
        return img


def _data_transforms_cifar100():
    CIFAR_MEAN = [0.5071, 0.4865, 0.4409]
    CIFAR_STD = [0.2673, 0.2564, 0.2762]

    train_transform = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
        ]
    )

    train_transform.transforms.append(Cutout(16))

    valid_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(CIFAR_MEAN, CIFAR_STD),])

    return train_transform, valid_transform


def load_cifar100_data(datadir, process_id, synthetic_data_url, private_local_data, resize=32, augmentation=True, data_efficient_load=False):
    train_transform, test_transform = _data_transforms_cifar100()

    is_download = True;

    if data_efficient_load:
        cifar100_train_ds = CIFAR100(datadir, train=True, download=True, transform=train_transform)
        cifar100_test_ds = CIFAR100(datadir, train=False, download=True, transform=test_transform)
    else:
        cifar100_train_ds = CIFAR100_truncated(datadir, train=True, download=is_download, transform=train_transform)
        cifar100_test_ds = CIFAR100_truncated(datadir, train=False, download=is_download, transform=test_transform)

    X_train, y_train = cifar100_train_ds.data, cifar100_train_ds.targets
    X_test, y_test = cifar100_test_ds.data, cifar100_test_ds.targets

    return (X_train, y_train, X_test, y_test, cifar100_train_ds, cifar100_test_ds)


def partition_data(dataset, datadir, partition, n_nets, alpha, process_id, synthetic_data_url, private_local_data, group_num=1, group_alpha=0.5):
    np.random.seed(10)
    logging.info("*********partition data***************")
    X_train, y_train, X_test, y_test, cifar100_train_ds, cifar100_test_ds = load_cifar100_data(datadir, process_id, synthetic_data_url, private_local_data)
    n_train = X_train.shape[0]
    # n_test = X_test.shape[0]
    group_indexes = []


    if partition == "homo":
        total_num = n_train
        idxs = np.random.permutation(total_num)
        batch_idxs = np.array_split(idxs, n_nets)
        net_dataidx_map = {i: batch_idxs[i] for i in range(n_nets)}

    elif partition == "hetero" and group_num == 1:
        min_size = 0
        K = 100
        N = y_train.shape[0]
        logging.info("N = " + str(N))
        net_dataidx_map = {}

        while min_size < 10:
            idx_batch = [[] for _ in range(n_nets)]
            # for each class in the dataset
            for k in range(K):
                idx_k = np.where(y_train == k)[0]
                np.random.shuffle(idx_k)
                proportions = np.random.dirichlet(np.repeat(alpha, n_nets))
                ## Balance
                proportions = np.array([p * (len(idx_j) < N / n_nets) for p, idx_j in zip(proportions, idx_batch)])
                proportions = proportions / proportions.sum()
                proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
                idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
                min_size = min([len(idx_j) for idx_j in idx_batch])

        for j in range(n_nets):
            np.random.shuffle(idx_batch[j])
            net_dataidx_map[j] = idx_batch[j]

    elif partition == "hetero" and group_num > 1:
        # distribute group data
        min_size = 0
        K = 100
        N = y_train.shape[0]
        logging.info("N = " + str(N))
        group_dataidx_map = {}

        while min_size < 1000:
            idx_batch = [[] for _ in range(group_num)]
            # for each class in the dataset
            for k in range(K):
                idx_k = np.where(y_train == k)[0]
                np.random.shuffle(idx_k)
                proportions = np.random.dirichlet(np.repeat(group_alpha, group_num))
                ## Balance
                proportions = np.array([p * (len(idx_j) < N / group_num) for p, idx_j in zip(proportions, idx_batch)])
                proportions = proportions / proportions.sum()
                proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
                idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
                min_size = min([len(idx_j) for idx_j in idx_batch])

        for group_id in range(group_num):
            np.random.shuffle(idx_batch[group_id])
            group_dataidx_map[group_id] = np.array(idx_batch[group_id])

        # assign number of clients to each group
        client_num_in_groups = [n_nets * len(group_dataidx_map[group_id]) // N for group_id in range(group_num)]
        remain_client_num = n_nets - sum(client_num_in_groups)
        while remain_client_num > 0:
            client_num_in_groups[remain_client_num - 1] += 1
            remain_client_num -= 1

        # distribute client data
        net_dataidx_map = {}
        client_start_idx = np.cumsum(client_num_in_groups)
        for group_id in range(group_num):

            min_size = 0
            n = len(group_dataidx_map[group_id])
            logging.info("n_{} = {}".format(group_id, n))

            while min_size < 10:
                idx_batch = [[] for _ in range(client_num_in_groups[group_id])]
                # for each class in the dataset
                for k in range(K):
                    idx_k = np.where(y_train[group_dataidx_map[group_id]] == k)[0]
                    np.random.shuffle(idx_k)
                    proportions = np.random.dirichlet(np.repeat(alpha, client_num_in_groups[group_id]))
                    proportions = np.array([p * (len(idx_j) < n / client_num_in_groups[group_id]) for p, idx_j in zip(proportions, idx_batch)])
                    proportions = proportions / proportions.sum()
                    proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
                    idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
                    min_size = min([len(idx_j) for idx_j in idx_batch])
                # print([len(idx_j) for idx_j in idx_batch])
                # print(client_num_in_groups, client_num_in_groups[group_id], n)

            for j in range(client_num_in_groups[group_id]):
                np.random.shuffle(idx_batch[j])
                client_id = j
                if group_id >= 1:
                    client_id += client_start_idx[group_id-1]
                net_dataidx_map[client_id] = group_dataidx_map[group_id][idx_batch[j]]
                group_indexes.append(group_id)

    elif partition == "hetero-fix":
        dataidx_map_file_path = "./data_preprocessing/non-iid-distribution/CIFAR100/net_dataidx_map.txt"
        net_dataidx_map = read_net_dataidx_map(dataidx_map_file_path)

    if partition == "hetero-fix":
        distribution_file_path = "./data_preprocessing/non-iid-distribution/CIFAR100/distribution.txt"
        traindata_cls_counts = read_data_distribution(distribution_file_path)
    else:
        traindata_cls_counts = record_net_data_stats(y_train, net_dataidx_map)

    return X_train, y_train, X_test, y_test, net_dataidx_map, group_indexes, traindata_cls_counts, cifar100_train_ds, cifar100_test_ds


# for centralized training
def get_dataloader(
    dataset,
    datadir,
    train_bs,
    test_bs,
    dataidxs=None,
    data_efficient_load=False,
    full_train_dataset=None,
    full_test_dataset=None,
):
    return get_dataloader_CIFAR100(
        datadir,
        train_bs,
        test_bs,
        dataidxs,
        data_efficient_load=data_efficient_load,
        full_train_dataset=full_train_dataset,
        full_test_dataset=full_test_dataset,
    )


# for local devices
def get_dataloader_test(dataset, datadir, train_bs, test_bs, dataidxs_train, dataidxs_test):
    return get_dataloader_test_CIFAR100(datadir, train_bs, test_bs, dataidxs_train, dataidxs_test)


def get_dataloader_CIFAR100(
    datadir,
    train_bs,
    test_bs,
    dataidxs=None,
    data_efficient_load=False,
    full_train_dataset=None,
    full_test_dataset=None,
):
    transform_train, transform_test = _data_transforms_cifar100()

    if data_efficient_load:
        dl_obj = CIFAR100_truncated_WO_reload
        train_ds = dl_obj(
            datadir, dataidxs=dataidxs, train=True, transform=transform_train, full_dataset=full_train_dataset
        )
        test_ds = dl_obj(datadir, train=False, transform=transform_test, full_dataset=full_test_dataset)
    else:
        dl_obj = CIFAR100_truncated
        train_ds = dl_obj(datadir, dataidxs=dataidxs, train=True, transform=transform_train, download=True)
        test_ds = dl_obj(datadir, train=False, transform=transform_test, download=True)

    train_dl = data.DataLoader(dataset=train_ds, batch_size=train_bs, shuffle=True, drop_last=True)
    test_dl = data.DataLoader(dataset=test_ds, batch_size=test_bs, shuffle=False, drop_last=True)

    return train_dl, test_dl


def get_dataloader_test_CIFAR100(datadir, train_bs, test_bs, dataidxs_train=None, dataidxs_test=None):
    dl_obj = CIFAR100_truncated

    transform_train, transform_test = _data_transforms_cifar100()

    train_ds = dl_obj(datadir, dataidxs=dataidxs_train, train=True, transform=transform_train, download=True,)
    test_ds = dl_obj(datadir, dataidxs=dataidxs_test, train=False, transform=transform_test, download=True,)

    train_dl = data.DataLoader(dataset=train_ds, batch_size=train_bs, shuffle=True, drop_last=True)
    test_dl = data.DataLoader(dataset=test_ds, batch_size=test_bs, shuffle=False, drop_last=True)

    return train_dl, test_dl


def load_partition_data_distributed_cifar100(
    process_id,
    dataset,
    data_dir,
    partition_method,
    partition_alpha,
    client_number,
    batch_size,
    data_efficient_load=True,
):
    (
        X_train,
        y_train,
        X_test,
        y_test,
        net_dataidx_map,
        group_indexes,
        traindata_cls_counts,
        cifar100_train_ds,
        cifar100_test_ds,
    ) = partition_data(dataset, data_dir, partition_method, client_number, partition_alpha)
    class_num = len(np.unique(y_train))
    logging.info("traindata_cls_counts = " + str(traindata_cls_counts))
    train_data_num = sum([len(net_dataidx_map[r]) for r in range(client_number)])

    # get global test data
    if process_id == 0:
        train_data_global, test_data_global = get_dataloader(
            dataset,
            data_dir,
            batch_size,
            batch_size,
            data_efficient_load=True,
            full_train_dataset=cifar100_train_ds,
            full_test_dataset=cifar100_test_ds,
        )
        logging.info("train_dl_global number = " + str(len(train_data_global)))
        logging.info("test_dl_global number = " + str(len(test_data_global)))
        train_data_local = None
        test_data_local = None
        local_data_num = 0
    else:
        # get local dataset
        dataidxs = net_dataidx_map[process_id - 1]
        local_data_num = len(dataidxs)
        logging.info("rank = %d, local_sample_number = %d" % (process_id, local_data_num))
        # training batch size = 64; algorithms batch size = 32
        train_data_local, test_data_local = get_dataloader(
            dataset,
            data_dir,
            batch_size,
            batch_size,
            dataidxs,
            data_efficient_load=True,
            full_train_dataset=cifar100_train_ds,
            full_test_dataset=cifar100_test_ds,
        )
        logging.info(
            "process_id = %d, batch_num_train_local = %d, batch_num_test_local = %d"
            % (process_id, len(train_data_local), len(test_data_local))
        )
        train_data_global = None
        test_data_global = None
    return (
        train_data_num,
        train_data_global,
        test_data_global,
        local_data_num,
        train_data_local,
        test_data_local,
        class_num,
    )


def efficient_load_partition_data_cifar100(
    dataset,
    data_dir,
    partition_method,
    partition_alpha,
    client_number,
    batch_size,
    process_id=0,
    synthetic_data_url="",
    private_local_data="",
    n_proc_in_silo=0,
    data_efficient_load=True,
    group_num=1,
    group_alpha=0.5
):
    (
        X_train,
        y_train,
        X_test,
        y_test,
        net_dataidx_map,
        group_indexes,
        traindata_cls_counts,
        cifar100_train_ds,
        cifar100_test_ds,
    ) = partition_data(dataset, data_dir, partition_method, client_number, partition_alpha, process_id, synthetic_data_url, private_local_data,
                       group_num=group_num, group_alpha=group_alpha)
    class_num = len(np.unique(y_train))
    logging.info("traindata_cls_counts = " + str(traindata_cls_counts))
    train_data_num = sum([len(net_dataidx_map[r]) for r in range(client_number)])

    train_data_global, test_data_global = get_dataloader(
        dataset,
        data_dir,
        batch_size,
        batch_size,
        data_efficient_load=True,
        full_train_dataset=cifar100_train_ds,
        full_test_dataset=cifar100_test_ds,
    )
    logging.info("train_dl_global number = " + str(len(train_data_global)))
    logging.info("test_dl_global number = " + str(len(test_data_global)))
    test_data_num = len(test_data_global)

    # get local dataset
    data_local_num_dict = dict()
    train_data_local_dict = dict()
    test_data_local_dict = dict()

    for client_idx in range(client_number):
        dataidxs = net_dataidx_map[client_idx]
        local_data_num = len(dataidxs)
        data_local_num_dict[client_idx] = local_data_num
        logging.info("client_idx = %d, local_sample_number = %d" % (client_idx, local_data_num))

        # training batch size = 64; algorithms batch size = 32
        train_data_local, test_data_local = get_dataloader(
            dataset,
            data_dir,
            batch_size,
            batch_size,
            dataidxs,
            data_efficient_load=True,
            full_train_dataset=cifar100_train_ds,
            full_test_dataset=cifar100_test_ds,
        )
        logging.info(
            "client_idx = %d, batch_num_train_local = %d, batch_num_test_local = %d"
            % (client_idx, len(train_data_local), len(test_data_local))
        )
        train_data_local_dict[client_idx] = train_data_local
        test_data_local_dict[client_idx] = test_data_local
    return (
        train_data_num,
        test_data_num,
        train_data_global,
        test_data_global,
        data_local_num_dict,
        train_data_local_dict,
        test_data_local_dict,
        class_num,
        group_indexes
    )

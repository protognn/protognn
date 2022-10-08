import torch_geometric as pyg
import torch_geometric.utils as pygutils
from os.path import join as opjoin
import scipy.sparse as sp
from torch import Tensor
import torch
import numpy as np


def rand_train_test_idx(label, train_prop=.5, valid_prop=.25, ignore_negative=True):
    """ randomly splits label into train/valid/test splits """
    if ignore_negative:
        labeled_nodes = torch.where(label != -1)[0]
    else:
        labeled_nodes = label

    n = labeled_nodes.shape[0]
    train_num = int(n * train_prop)
    valid_num = int(n * valid_prop)

    perm = torch.as_tensor(np.random.permutation(n))

    train_indices = perm[:train_num]
    val_indices = perm[train_num:train_num + valid_num]
    test_indices = perm[train_num + valid_num:]

    if not ignore_negative:
        return train_indices, val_indices, test_indices

    train_idx = labeled_nodes[train_indices]
    valid_idx = labeled_nodes[val_indices]
    test_idx = labeled_nodes[test_indices]

    return train_idx, valid_idx, test_idx



def mask_to_index(mask: Tensor) -> Tensor:
    r"""Converts a mask to an index representation.

    Args:
        mask (Tensor): The mask.
    """
    return mask.nonzero(as_tuple=False).view(-1)


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def index_to_mask(index, size):
    mask = torch.zeros(size, dtype=torch.bool, device=index.device)
    mask[index] = 1
    return mask


def resplit(dataset, data, full_sup, num_classes, num_nodes,  split):
    if dataset in ["penn94", "reed98", "amherst41", "cornell5"]:
        if dataset in ["penn94"]:
            #print(data[6].size())
            data[3] = data[6][:, split]
            data[4] = data[7][:, split]
            data[5] = data[8][:, split]
        else:
            train_index, val_index, test_index = rand_train_test_idx(data[2],  train_prop=.5, valid_prop=.25, ignore_negative=True)
            data[3] = index_to_mask(train_index, size=num_nodes)
            data[4] = index_to_mask(val_index, size=num_nodes)
            data[5] = index_to_mask(test_index, size=num_nodes)
    return data


def load_data(config):
    graph = get_pyg_dataset(dataroot=config['path']['dataroot'], dataset=config['data']['dataset'])
    data = graph.data

    labels = data.y
    features = data.x
    try:
        idx_train = data.train_mask
        idx_val = data.val_mask
        idx_test = data.test_mask
    except:
        idx_train, idx_val, idx_test = rand_train_test_idx(labels)

    edge_index = data.edge_index
    if config['data']['to_undirected']:
        edge_index = pygutils.to_undirected(edge_index)
    if config['data']['add_slflp']:
        edge_index = pygutils.add_self_loops(edge_index)[0]

    return [edge_index, features, labels, idx_train, idx_val, idx_test, idx_train, idx_val, idx_test]


def dummy_normalization(mx):
    if isinstance(mx, np.ndarray) or isinstance(mx, sp.csr.csr_matrix):
        pass
    elif isinstance(mx, sp.lil.lil_matrix):
        mx = np.asarray(mx.todense())
    else:
        raise NotImplementedError
    return mx


def get_pyg_dataset(dataroot, dataset):

    if dataset in ["penn94", "reed98", "amherst41", "cornell5", "johnshopkins55"]:
        graph = pyg.datasets.LINKXDataset(root=opjoin(dataroot, dataset), name=dataset)
    else:
        raise NotImplementedError
    return graph


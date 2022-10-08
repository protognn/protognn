import torch

import numpy as np

from sklearn.metrics import roc_auc_score, f1_score
import torch.nn.functional as F


def get_compatibility_loss(x, prototypes, labels, num_classes):
    # x = nxd
    # proto = c x d
    dim = x.size(-1)
    dots = []
    prototypes = prototypes.view(num_classes,-1, dim) #.max(dim=1)[0]
    for i in range(prototypes.size(1)):
        prototypes_i = prototypes[:,i]
        dots_i = torch.cdist(x, prototypes_i, p=2)
        dots.append(dots_i.unsqueeze(1))
    dots = torch.cat(dots, dim=1).max(dim=1)[0]
    attn = dots #.softmax(dim=1) # n x c
    positives = torch.gather(input=attn[:,:], dim=1, index=labels[:,None]).squeeze()
    negatives = attn[F.one_hot(labels)==0]
    comp_loss = torch.sum(positives) + torch.logsumexp(-negatives, dim=0)
    comp_loss = comp_loss / (x.size(0)*prototypes.size(0))
    return comp_loss


def eval_acc(y_true, y_pred):
    acc_list = []
    y_true = y_true.detach().cpu().numpy()
    y_pred = y_pred.argmax(dim=-1, keepdim=True).detach().cpu().numpy()

    y_pred = np.squeeze(y_pred)
    correct = y_true[y_true!=-1] == y_pred[y_true !=-1]


    return float(np.sum(correct)) / len(correct)


def eval_rocauc(y_true, y_pred):
    """ adapted from ogb
    https://github.com/snap-stanford/ogb/blob/master/ogb/nodeproppred/evaluate.py"""
    rocauc_list = []
    y_true = y_true.detach().cpu().numpy()

    def one_hot(a, num_classes):
        return np.squeeze(np.eye(num_classes)[a.reshape(-1)])
    y_true = one_hot(y_true, np.max(y_true) + 1)
    # if y_true.shape[1] == 1:
    #     # use the predicted class for single-class classification
    #     y_pred = F.softmax(y_pred, dim=-1)[:, 1].unsqueeze(1).cpu().numpy()
    # else:
    y_pred = y_pred.detach().cpu().numpy()


    for i in range(y_true.shape[1]):
        # AUC is only defined when there is at least one positive data.
        if np.sum(y_true[:, i] == 1) > 0 and np.sum(y_true[:, i] == 0) > 0:
            is_labeled = y_true[:, i] == y_true[:, i]
            score = roc_auc_score(y_true[is_labeled, i], y_pred[is_labeled, i])

            rocauc_list.append(score)

    if len(rocauc_list) == 0:
        raise RuntimeError(
            'No positively labeled data available. Cannot compute ROC-AUC.')

    return sum(rocauc_list) / len(rocauc_list)


@torch.no_grad()
def evaluate(model, edge_index, features, labels, idx_train, idx_val, idx_test, eval_func,):
    model.eval()

    out = model(edge_index, features, labels)

    train_acc = eval_func(
        labels[idx_train], out[idx_train])
    valid_acc = eval_func(
        labels[idx_val], out[idx_val])
    test_acc = eval_func(
        labels[idx_test], out[idx_test])

    return train_acc, valid_acc, test_acc


def get_orthogonal_regularization_loss(s):
    # Orthogonality regularization.
    s = s.unsqueeze(0) if s.dim() == 2 else s
    k = s.size(-1)
    ss = torch.matmul(s.transpose(1, 2), s)
    i_s = torch.eye(k).type_as(ss)
    ortho_loss = torch.norm(ss / torch.norm(ss, dim=(-1, -2), keepdim=True) - i_s / torch.norm(i_s), dim=(-1, -2))
    ortho_loss = torch.mean(ortho_loss)
    return ortho_loss


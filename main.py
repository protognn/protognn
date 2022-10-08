from config import ex
import os
import torch.nn.functional as F
from copy import deepcopy as dcopy
from os.path import join as opjoin
from utils import load_data,  resplit
from initializer import init_logger, init_seed, init_optimizer
import torch
from models import LINKX_PROTO
import numpy as np
from worker import get_orthogonal_regularization_loss,evaluate,eval_acc, eval_rocauc, get_compatibility_loss


@ex.automain
def main(_run, _config, _log):
    '''
    _config: dictionary; its keys and values are the variables setting in the cfg function
    _run: run object defined by Sacred, can be used to record hashable values and get some information, e.g. run id, for a run
    _log: logger object provided by Sacred, but is not very flexible, we can define loggers by oureselves
    '''
    
    config = dcopy(_config)  # We need this step because Sacred does not allow us to change _config object
                        # But sometimes we need to add some key-value pairs to config
    torch.cuda.set_device(config['gpu_id'])

    #save_source(_run)  # Source code are saved by running this line
    init_seed(config['seed'])
    logger = init_logger(log_root=_run.observers[0].dir, file_name='log.txt')

    output_folder_path = opjoin(_run.observers[0].dir, config['path']['output_folder_name'])
    os.makedirs(output_folder_path, exist_ok=True)

    best_acc_list = []

    device = torch.device(f"cuda:{config['gpu_id']}" if torch.cuda.is_available() else 'cpu')

    data = load_data(config=config)
    data = [each.to(device)  for each in data]

    split_iterator = range(config['data']['random_split']['num_splits']) \
        if config['data']['random_split']['use'] \
        else range(1)

    config['adj'] = data[0]

    eval_func = eval_acc

    for i in split_iterator:

        output_folder = opjoin(output_folder_path, str(i))
        os.makedirs(output_folder, exist_ok=True)

        if config['data']['random_split']['use']:
            data = resplit(dataset=config['data']['dataset'],
                           data=data,
                           full_sup=config['data']['full_sup'],
                           num_classes=torch.unique(data[2]).shape[0],
                           num_nodes=data[1].shape[0],
                           split=i)

        edge_index, features, labels, idx_train, idx_val, idx_test = data[0], data[1],data[2],data[3],data[4], data[5]
        config['idx_train'] = idx_train.to(device)

        if config['arch']['backbone'] == 'linkx':
            model = LINKX_PROTO( config=config, in_channels=config['data']['num_features'], hidden_channels=config['arch']['num_hiddens'], out_channels=config['data']['num_classes'],  num_nodes=features.size()[0] ).to(device)

        if i == 0:
            logger.info(model)
        optimizer = init_optimizer(params=model.parameters(),
                                   optim_type=config['optim']['type'],
                                   lr=config['optim']['lr'],
                                   weight_decay=config['optim']['weight_decay'],
                                   momentum=config['optim']['momemtum'])

        def train():
            model.train()
            optimizer.zero_grad()
            output, new_labels, new_idx_train, prototypes, x_comp, prototypes_comp, old_idx = model(edge_index, features, labels)
            
            if prototypes_comp is not None:
                comploss = get_compatibility_loss(x_comp[old_idx], prototypes_comp, labels[old_idx],
                                                   config['data']['num_classes'])
                ortholoss = get_orthogonal_regularization_loss(prototypes_comp)
            else:
                comploss = 0
                ortholoss = 0
            if config['optim']['comp_weight'] > 0:
                loss = F.nll_loss(output[new_idx_train], new_labels[new_idx_train].squeeze()) + config['optim']['comp_weight']* comploss  + config['optim']['ortho_weight']*ortholoss
            else:
                loss = F.nll_loss(output[new_idx_train], new_labels[new_idx_train].squeeze())  + config['optim']['ortho_weight']*ortholoss
            loss.backward()
            optimizer.step()


        logger.info(f'split_seed: {i: 04d}')
        best_val_acc = test_acc = -1
        for epoch in range(1, config['optim']['epoch'] + 1):

            train()
            train_acc, val_acc, tmp_test_acc = evaluate(model, edge_index, features, labels,idx_train, idx_val, idx_test, eval_func)
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                test_acc = tmp_test_acc
            logger.info(f'Epoch: {epoch:03d},Train: {train_acc:.4f}, '
                  f'Val: {best_val_acc:.4f}, Test: {test_acc:.4f}')
        best_acc_list.append(test_acc)
    logger.info('********************* STATISTICS *********************')
    np.set_printoptions(precision=4, suppress=True)
    logger.info(f"\n"
                f"Best test acc: {best_acc_list}\n"
                f"Mean: {np.mean(best_acc_list)}\t"
                f"Std: {np.std(best_acc_list)}\n"
                )




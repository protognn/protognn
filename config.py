# pylint: disable=unused-variable

'''
This is a config.py file of a standard sarcred format
We do not need to modify too much here
'''
import os
from os.path import join as opjoin
import sacred
from sacred import Experiment
from sacred.observers import FileStorageObserver
from sacred.utils import apply_backspaces_and_linefeeds


sacred.SETTINGS['CONFIG']['READ_ONLY_CONFIGS'] = False
sacred.SETTINGS.CAPTURE_MODE = 'no'

ex = Experiment('exp')

ex.captured_out_filter = apply_backspaces_and_linefeeds


source_root = './'
source_files = list(filter(lambda filename: filename.endswith('.py'), os.listdir('./')))
for source_file in source_files:
    ex.add_source_file(source_file)


@ex.config
def cfg():
    ex_name = ex.path
    ex_type = ''
    fastmode = False
    seed = 42

    use_gpu = True
    gpu_id = 0
    save_model = False


    features_classes = {
        "cornell5": (4735, 2),
       }


    data = {
        'dataset': 'cornell5',
        'add_slflp': False,
        'random_split':{
                        'use': True, # if false, only run with the given split
                        'num_splits': 3,
                       },
        'full_sup': True, # if true, use fully-supervised setting
        "to_undirected":True,
    }

    data['num_features'], data['num_classes'] = features_classes.get(data['dataset'])
    optim = {
        'type': 'adam', #'adam', 'rmsprop', 'sgd'
        'epoch':2000,
        'lr': 0.005,
        'weight_decay': 0.0005,
        'momemtum': 0.9, # momentum is only applicable for sgd
        'comp_weight':0.01,
        'ortho_weight': 0.1,

    }

    proto = {
        'ce_loss':True,
        'num': 10,
        'n_iter':1,
        'conv_iter':5,
        'eps':1e-8,
        'hidden_dim':64,
        'init_strategy': 'fix_mu', #'random', 'sample', 'fix_mu'
    }
    linkx = {'num_layers': 1}
    arch = {
        'backbone': 'linkx',
        'implement': 'pyg',
        'num_hiddens': 64,
        'proto':True,
        }

    record_grad = False

    path = {
        'log_dir': './runs',
        'dataroot': '../data',
        'output_folder_name': 'output'
    }

    cmt_items=["hd", "type",  "wd", "lr"]


@ex.config_hook
def add_observer(config, command_name, logger):
    '''
    config: dictionary; its keys and values are the variables setting in 3the cfg function
    typically we do not need to use command_name and logger here
    '''
    sanity_check(config)
    os.makedirs(config['path']['log_dir'], exist_ok=True) 
    exp_cmt = get_comment(config)
    observer = FileStorageObserver.create(opjoin(config['path']['log_dir'], config['ex_name'], config['data']['dataset'], config['ex_type'], exp_cmt)) 
    ex.observers.append(observer)
    return config


def sanity_check(config):
    if config['data']['random_split']['use']:
        assert config['data']['random_split']['num_splits'] >= 1

def get_comment(config):

    cmt_items = config['cmt_items']

    assert type(cmt_items) == list
    exp_cmt = []

    if 'type' in cmt_items:
        exp_cmt.append(f"type_{config['arch']['backbone']}", )
    if 'hd' in cmt_items:
        exp_cmt.append(f"hidden_{config['arch']['num_hiddens']}", )
    if 'lr' in cmt_items:
        exp_cmt.append(f"lr_{config['optim']['lr']:.1e}", )
    if 'wd' in cmt_items:
        exp_cmt.append(f"wd_{config['optim']['weight_decay']}", )
    exp_cmt = '_'.join(exp_cmt)

    return exp_cmt



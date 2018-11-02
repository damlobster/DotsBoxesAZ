from functools import partial

from utils.utils import get_cuda_devices_list, DotDict
from dots_boxes.dots_boxes_game import BoxesState, nn_batch_builder
from dots_boxes.dots_boxes_nn import SimpleNN
from nn import ResNetZero

import logging
import logging.config

BoxesState.init_static_fields(dims=(3, 3))

simple33 = DotDict({
    "data_root": "data/simple2710",
    "hdf_file": "data/simple2710/sp_data.hdf",
    "tensorboard_log": "./data/tboard/simple2710",
    "game": {
        "clazz": BoxesState,
        "init": partial(BoxesState.init_static_fields, ((3,3),)),
    },
    "self_play": {
        "n_workers": 11,
        "num_games": 1000,
        "reuse_mcts_tree": True,
        "noise": (1.0, 0.25),  # alpha, coeff
        "nn_batch_size": 48,
        "nn_batch_timeout": 0.01,
        "nn_batch_builder": nn_batch_builder,
        "pytorch_devices": ["cuda:1", "cuda:0"], #get_cuda_devices_list(),
        "mcts": {
            "mcts_num_read": 800,
            "mcts_cpuct": 2.0,
            "temperature": {0: 1.0, 6: 1e-50},  # from 8th move we greedly take move with most visit count
            "max_async_searches": 64,
        }
    },
    "elo":{
        "hdf_file": "data/simple2710/elo_data.hdf",
        "n_games": 10,
        "n_workers": 10,
        "games_per_workers":1
    },
    "nn": {
        "train_params": {
            "nb_epochs": 9,
            "train_split": 0.9,
            "train_batch_size": 512,
            "val_batch_size": 1024,
            "lr": 1e-4,
            "lr_scheduler": {"max_lr_factor": 10, "step_size": 2000},
            "adam_params": {
                "betas":(0.9, 0.999),
                "weight_decay":1e-5,
            },
        },
        "model_class":SimpleNN,
        "pytorch_device": "cuda:1",
        "chkpts_filename": "data/simple2710/model_gen{}.pt",
        "model_parameters":{
            "resnet": {
                "in_channels": 3,
                "nb_channels": 256,
                "kernel_size": 3,
                "nb_blocks": 20
            },
            "policy_head": {
                "in_channels": 256,
                "nb_actions": 32,
            },
            "value_head": {
                "in_channels": 256,
                "nb_actions": 32,
                "n_hidden": 128
            },
        }
    }
})

resnet20 = DotDict({
    "data_root": "data/resnet3110",
    "hdf_file": "data/resnet3110/sp_data.hdf",
    "tensorboard_log": "data/tboard/resnet3310",
    "game": {
        "clazz": BoxesState,
        "init": partial(BoxesState.init_static_fields, ((3, 3),)),
    },
    "self_play": {
        "num_games": 1000,
        "n_workers": 12,
        "games_per_workers": 10,
        "reuse_mcts_tree": True,
        "noise": (1.0, 0.25),  # alpha, coeff
        "nn_batch_size": 48,
        "nn_batch_timeout": 0.05,
        "nn_batch_builder": nn_batch_builder,
        "pytorch_devices": ["cuda:1", "cuda:0"],  # get_cuda_devices_list(),
        "mcts": {
            "mcts_num_read": 800,
            "mcts_cpuct": 2.0,
            # from 6th move we greedly take move with most visit count
            "temperature": {0: 1.0, 6: 1e-50},
            "max_async_searches": 64,
        }
    },
    "elo": {
        "hdf_file": "data/resnet3110/elo_data.hdf",
        "n_games": 10,
        "n_workers": 10,
        "games_per_workers": 1,
        "self_play_override":{
            "reuse_mcts_tree": False,
            "noise": (0.0, 0.0),
        }
    },
    "nn": {
        "train_params": {
            "nb_epochs": 9,
            "train_split": 0.9,
            "train_batch_size": 512,
            "val_batch_size": 1024,
            "lr": 1e-4,
            "adam_params": {
                "betas": (0.9, 0.999),
                "weight_decay": 1e-5,
            },
        },
        "model_class": ResNetZero,
        "pytorch_device": "cuda:1",
        "chkpts_filename": "data/resnet3110/model_gen{}.pt",
        "model_parameters": {
            "resnet": {
                "in_channels": 3,
                "nb_channels": 100,
                "kernel_size": 3,
                "nb_blocks": 20
            },
            "policy_head": {
                "in_channels": 100,
                "activation_map": (4,4),
                "inner_channels":64,
                "nb_actions": 32,
            },
            "value_head": {
                "in_channels": 100,
                "activation_map": (4,4),
                "n_hidden0": 64,
                "n_hidden1": 32
            },
        }
    }
})

params = resnet20

DEFAULT_LOGGING = {
    'version': 1,
    'disable_existing_loggers': True,
    'formatters': { 
        'standard': { 
            'format': '%(message)s'
        },
    },
    'handlers': { 
        'default': { 
            'class': 'logging.StreamHandler',
            'formatter': 'standard',
            'stream'  : 'ext://sys.stderr'
        },
    },
    'loggers': {
        'mcts': {
            'handlers': ['default'],
            'level': 'INFO',
        },
        'nn': {
            'handlers': ['default'],
            'level': 'INFO',
        },
        'self_play': {
            'handlers': ['default'],
            'level': 'INFO',
        },
        'utils.proxies':{
            'handlers': ['default'],
            'level': 'INFO'
        }
    }
}
logging.config.dictConfig(DEFAULT_LOGGING)
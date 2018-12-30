from datetime import datetime
from functools import partial

from utils.utils import get_cuda_devices_list, DotDict
from dots_boxes.dots_boxes_game import BoxesState, nn_batch_builder
from dots_boxes.dots_boxes_nn import SymmetriesGenerator, SimpleNN
from nn import ResNetZero, GenerationLrScheduler

import logging
import logging.config

#BoxesState.init_static_fields(dims=(3, 3))


simple = DotDict({
    "data_root": "data/_exp_",
    "hdf_file": "data/_exp_/sp_data.hdf",
    "tensorboard_log": "data/tboard/_exp_",
    "game": {
        "clazz": BoxesState,
        "init": partial(BoxesState.init_static_fields, ((3, 3),)),
    },
    "self_play": {
        "num_games": 4000,
        "n_workers": 12,
        "games_per_workers": 10,
        "reuse_mcts_tree": True,
        "noise": (1.0, 0.25),  # alpha, coeff
        "nn_batch_size": 48,
        "nn_batch_timeout": 0.05,
        "nn_batch_builder": nn_batch_builder,
        "pytorch_devices": ["cuda:1", "cuda:0"], #get_cuda_devices_list(),
        "mcts": {
            "mcts_num_read": 200,
            "mcts_cpuct": 4.0,
            "temperature": {0: 1.0, 8: 1e-50},  # from 8th move we greedly take move with most visit count
            "max_async_searches": 64,
        }
    },
    "elo": {
        "hdf_file": "data/_exp_/elo_data.hdf",
        "n_games": 10,
        "n_workers": 10,
        "games_per_workers": 1,
        "self_play_override":{
            "reuse_mcts_tree": False,
            "noise": (0.0, 0.0),
            "mcts":{
                "mcts_num_read":1600
            }
        }
    },
    "nn": {
        "model_class":SimpleNN,
        "pytorch_device": "cuda:0",
        "chkpts_filename": "data/_exp_/model_gen{}.pt",
        "train_params": {
            "nb_epochs": 5,
            "train_split": 0.9,
            "train_batch_size": 2048,
            "val_batch_size": 2048,
            "lr_scheduler": GenerationLrScheduler({0: 1e-3, 20:1e-4}),
            "lr": 1e-4,
            "optimizer_momentum":0.9,
            "weight_decay":1e-4,
        },
        "model_parameters": None
    }
})

resnet20 = DotDict({
    "data_root": "data/_exp_",
    "hdf_file": "data/_exp_/sp_data.hdf",
    "tensorboard_log": "data/tboard/_exp_",
    "game": {
        "clazz": BoxesState,
        "init": partial(BoxesState.init_static_fields, ((3, 3),)),
    },
    "self_play": {
        "num_games": 1920,
        "n_workers": 12,
        "games_per_workers": 40,
        "reuse_mcts_tree": True,
        "noise": (0.8, 0.25),  # alpha, coeff
        "nn_batch_size": 48,
        "nn_batch_timeout": 0.05,
        "nn_batch_builder": nn_batch_builder,
        "pytorch_devices": ["cuda:1", "cuda:0"],  # get_cuda_devices_list(),
        "mcts": {
            "mcts_num_read": 800,
            "mcts_cpuct": (1.25, 19652), # CPUCT, CPUCT_BASE
            "temperature": {0: 1, 9: 0.02},
            "max_async_searches": 64,
        }
    },
    "elo": {
        "hdf_file": "data/_exp_/elo_data.hdf",
        "n_games": 20,
        "n_workers": 10,
        "games_per_workers": 2,
        "self_play_override":{
            "reuse_mcts_tree": False,
            "noise": (0.0, 0.0),
            "mcts":{
                "mcts_num_read":800
            }
        }
    },
    "nn": {
        "model_class": ResNetZero,
        "pytorch_device": "cuda:0",
        "chkpts_filename": "data/_exp_/model_gen{}.pt",
        "train_params": {
            "pos_average": False,
            "symmetries": SymmetriesGenerator(), #lambda boards, pi, z: [boards], [pi], [z]
            "nb_epochs": 4,
            "max_samples_per_gen": 90*4096, # approx samples for 10 generations
            "train_split": 0.9,
            "train_batch_size": 4096,
            "val_batch_size": 2048,
            "lr_scheduler": GenerationLrScheduler({0: 1e-1, 20:1e-2, 40:1e-3}),
            "lr": 0.1,
            "optimizer_params": {
                "momentum":0.9,
                "weight_decay":1e-4,
            }
        },
        "model_parameters": {
            "resnet": {
                "pad_layer0": True,
                "in_channels": 3,
                "nb_channels": 64,
                "inner_channels": None,
                "kernel_size": 3,
                "nb_blocks": 20,
                "n_groups": 1
            },
            "policy_head": {
                "in_channels": 64,
                "inner_channels":2,
                "fc_in": 2*16,
                "nb_actions": 32,
            },
            "value_head": {
                "in_channels": 64,
                "inner_channels": 1,
                "fc_in": 1*16,
                "fc_inner": 8
            },
        }
    }
})

resnet55 = DotDict({
    "data_root": "data/_exp_",
    "hdf_file": "data/_exp_/sp_data.hdf",
    "tensorboard_log": "data/tboard/_exp_",
    "game": {
        "clazz": BoxesState,
        "init": partial(BoxesState.init_static_fields, ((5, 5),)),
    },
    "self_play": {
        "num_games": 10,
        "n_workers": 10,
        "games_per_workers": 1,
        "reuse_mcts_tree": True,
        "noise": (1.0, 0.25),  # alpha, coeff
        "nn_batch_size": 48,
        "nn_batch_timeout": 0.05,
        "nn_batch_builder": nn_batch_builder,
        "pytorch_devices": ["cuda:1", "cuda:0"],  # get_cuda_devices_list(),
        "mcts": {
            "mcts_num_read": 800,
            "mcts_cpuct": 4.0,
            # from 6th move we greedly take move with most visit count
            "temperature": {0: 1, 6: 0.2, 9:0.05}, #, 15: 1e-50},
            "max_async_searches": 64,
        }
    },
    "elo": {
        "hdf_file": "data/_exp_/elo_data.hdf",
        "n_games": 20,
        "n_workers": 10,
        "games_per_workers": 2,
        "self_play_override":{
            "reuse_mcts_tree": False,
            "noise": (0.0, 0.0),
            "mcts":{
                "mcts_num_read":800
            }
        }
    },
    "nn": {
        "model_class": ResNetZero,
        "pytorch_device": "cuda:1",
        "chkpts_filename": "data/_exp_/model_gen{}.pt",
        "train_params": {
            "pos_average": True,
            "symmetries": SymmetriesGenerator(), #lambda boards, pi, z: [boards], [pi], [z]
            "nb_epochs": 2,
            "train_split": 0.9,
            "train_batch_size": 2048,
            "val_batch_size": 2048,
            "lr_scheduler": GenerationLrScheduler({0: 1e-3, 20:5e-4,}),
            "lr": None,
            "adam_params": {
                "betas": (0.9, 0.999),
                "weight_decay": 1e-4,
            },
        },
        "model_parameters": {
            "resnet": {
                "in_channels": 3,
                "nb_channels": 128,
                "kernel_size": 3,
                "nb_blocks": 10
            },
            "policy_head": {
                "in_channels": 128,
                "inner_channels":4,
                "fc_in": 36*4,
                "nb_actions": 72,
            },
            "value_head": {
                "in_channels": 128,
                "inner_channels": 1,
                "fc_in": 1*36,
                "fc_inner": 18
            },
        }
    }
})

# configuration to use
params = resnet20
params.game.init()

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
            'level': 'WARNING',
        },
        'nn': {
            'handlers': ['default'],
            'level': 'WARNING',
        },
        'dots_boxes_nn': {
            'handlers': ['default'],
            'level': 'WARNING',
        },
        'dots_boxes_game': {
            'handlers': ['default'],
            'level': 'WARNING',
        },
        'self_play': {
            'handlers': ['default'],
            'level': 'WARNING',
        },
        'utils.proxies':{
            'handlers': ['default'],
            'level': 'WARNING'
        }
    }
}
logging.config.dictConfig(DEFAULT_LOGGING)

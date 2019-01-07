from datetime import datetime
from functools import partial

from utils.utils import get_cuda_devices_list, DotDict
from dots_boxes.dots_boxes_game import BoxesState, nn_batch_builder
from dots_boxes.dots_boxes_nn import SymmetriesGenerator, SimpleNN
from nn import ResNetZero, GenerationLrScheduler

import logging
import logging.config


simple = DotDict({
    "data_root": "data/_exp_",
    "hdf_file": "data/_exp_/sp_data.hdf",
    "tensorboard_log": "data/tboard/_exp_",
    "game": {
        "clazz": BoxesState,
        "init": partial(BoxesState.init_static_fields, ((3, 3),)),
    },
    "self_play": {
        "num_games": 2000,
        "n_workers": 20,
        "games_per_workers": 25,
        "reuse_mcts_tree": True,
        "noise": (0.8, 0.25),  # alpha, coeff
        "nn_batch_size": 48,
        "nn_batch_timeout": 0.05,
        "nn_batch_builder": nn_batch_builder,
        "pytorch_devices": ["cuda:1", "cuda:0"],  # get_cuda_devices_list(),
        "mcts": {
            "mcts_num_read": 800,
            "mcts_cpuct": (1.25, 19652),  # CPUCT, CPUCT_BASE
            "temperature": {0: 1.0, 12: 0.02},
            "max_async_searches": 64,
        }
    },
    "elo": {
        "hdf_file": "data/_exp_/elo_data.hdf",
        "n_games": 20,
        "n_workers": 10,
        "games_per_workers": 2,
        "self_play_override": {
            "reuse_mcts_tree": False,
            "noise": (0.0, 0.0),
            "mcts": {
                "mcts_num_read": 1200
            }
        }
    },
    "nn": {
        "model_class": SimpleNN,
        "pytorch_device": "cuda:0",
        "chkpts_filename": "data/_exp_/model_gen{}.pt",
        "train_params": {
            "pos_average": True,
            "symmetries": SymmetriesGenerator(),
            "nb_epochs": 10,
            "max_samples_per_gen":100*4096,  # approx samples for 10 generations
            "train_split": 0.9,
            "train_batch_size": 4096,
            "val_batch_size": 4096,
            "lr_scheduler": GenerationLrScheduler({0: 1e-2, 20: 1e-3, 50: 1e-4}),
            "lr": 1e-2,
            "optimizer_params": {
                "momentum": 0.9,
                "weight_decay": 1e-4,
            }
        },
        "model_parameters": None
    }
})

resnet = DotDict({
    "data_root": "data/_exp_",
    "hdf_file": "data/_exp_/sp_data.hdf",
    "tensorboard_log": "data/tboard/_exp_",
    "game": {
        "clazz": BoxesState,
        "init": partial(BoxesState.init_static_fields, ((3, 3),)),
    },
    "self_play": {
        "num_games": 2000,
        "n_workers": 20,
        "games_per_workers": 50,
        "reuse_mcts_tree": True,
        "noise": [
            0.8,
            0.25
        ],
        "nn_batch_size": 48,
        "nn_batch_timeout": 0.05,
        "nn_batch_builder": nn_batch_builder,
        "pytorch_devices": ["cuda:1", "cuda:0"],
        "mcts": {
            "mcts_num_read": 800,
            "mcts_cpuct": [1.25, 19652],
            "temperature": { 0: 1, 12: 0.02 },
            "max_async_searches": 64
        }
    },
    "elo": {
        "hdf_file": "data/_exp_/elo_data.hdf",
        "n_games": 20,
        "n_workers": 20,
        "games_per_workers": 1,
        "self_play_override": {
            "reuse_mcts_tree": False,
            "noise": [0.0, 0.0],
            "mcts": {
                "mcts_num_read": 1200
            }
        }
    },
    "nn": {
        "model_class": ResNetZero,
        "pytorch_device": "cuda:0",
        "chkpts_filename": "data/_exp_/model_gen{}.pt",
        "train_params": {
            "pos_average": True,
            "symmetries": SymmetriesGenerator(),
            "nb_epochs": 10,
            "max_samples_per_gen": 100*4096,
            "train_split": 0.9,
            "train_batch_size": 4096,
            "val_batch_size": 4096,
            "lr_scheduler": GenerationLrScheduler({0: 0.1, 30: 0.01, 50: 0.001}),
            "lr": 0.1,
            "optimizer_params": {
                "momentum": 0.9,
                "weight_decay": 0.0001
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
                "inner_channels": 16,
                "fc_in": 16*16,
                "nb_actions": 32
            },
            "value_head": {
                "in_channels": 64,
                "inner_channels": 16,
                "fc_in": 16*16,
                "fc_inner": 8
            }
        }
    }
})

# configuration to use
params = resnet
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
            'stream': 'ext://sys.stderr'
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
        'utils.proxies': {
            'handlers': ['default'],
            'level': 'WARNING'
        }
    }
}
logging.config.dictConfig(DEFAULT_LOGGING)

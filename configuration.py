import logging
logging.basicConfig(level=logging.INFO)

from utils.utils import get_cuda_devices_list, DotDict
from dots_boxes.dots_boxes_game import BoxesState, nn_batch_builder
from dots_boxes.dots_boxes_nn import SimpleNN
from nn import ResNetZero

BoxesState.init_static_fields(dims=(3, 3))
params = simple33


simple33 = DotDict({
    "data_root": "data/simple2710",
    "hdf_file": "data/simple2710/sp_data.hdf",
    "train_log": "data/tboard/simple2710",
    "game": {
        "clazz": BoxesState,
        "init": lambda: BoxesState.init_static_fields((3,3)),
    },
    "self_play": {
        "n_workers": 9,
        "num_games": 1000,
        "reuse_mcts_tree": True,
        "noise": (1.0, 0.25),  # alpha, coeff
        "nn_batch_size": 48,
        "nn_batch_timeout": 0.01,
        "nn_batch_builder": nn_batch_builder,
        "pytorch_devices": ["cuda:1", "cuda:0"], #get_cuda_devices_list(),
        "mcts": {
            "mcts_num_read": 800,
            "mcts_cpuct": 1.0,
            "temperature": {0: 1.0, 5: 1e-50},  # from 8th move we greedly take move with most visit count
            "max_async_searches": 64,
        }
    },
    "nn": {
        "train_params": {
            "nb_epochs": 50,
            "train_split": 0.9,
            "train_batch_size": 512,
            "val_batch_size": 1024,
            "lr": 0.001,
            "lr_scheduler": {"max_lr_factor": 8, "step_size": 400},
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
    "data_root": "data/resnet20",
    "hdf_file": "data/resnet20/sp_data.hdf",
    "tensorboard_log": "data/tboard/resnet20",
    "game": {
        "clazz": BoxesState
    },
    "self_play": {
        "n_workers": 11,
        "num_games": 1000,
        "reuse_mcts_tree": True,
        "noise": (1.0, 0.25),  # alpha, coeff
        "nn_batch_size": 48,
        "nn_batch_timeout": 0.01,
        "nn_batch_builder": nn_batch_builder,
        "pytorch_devices": ["cuda:1", "cuda:0"],  # get_cuda_devices_list(),
        "mcts": {
            "mcts_num_read": 800,
            "mcts_cpuct": 1.0,
            # from 8th move we greedly take move with most visit count
            "temperature": {0: 1.0, 5: 1e-50},
            "max_async_searches": 64,
        }
    },
    "nn": {
        "train_params": {
            "nb_epochs": 50,
            "train_split": 0.9,
            "train_batch_size": 512,
            "val_batch_size": 1024,
            "lr": 0.001,
            #"lr_scheduler": (10, 0.1),
            "adam_params": {
                "betas": (0.9, 0.999),
                "weight_decay": 1e-5,
            },
        },
        "model_class": ResNetZero,
        "pytorch_device": "cuda:1",
        "chkpts_filename": "data/resnet20/model_gen{}.pt",
        "model_parameters": {
            "resnet": {
                "in_channels": 3,
                "nb_channels": 128,
                "kernel_size": 3,
                "nb_blocks": 20
            },
            "policy_head": {
                "in_channels": 128,
                "nb_actions": BoxesState.NB_ACTIONS,
            },
            "value_head": {
                "in_channels": 128,
                "nb_actions": BoxesState.NB_ACTIONS,
                "n_hidden": 128
            },
        }
    }
})

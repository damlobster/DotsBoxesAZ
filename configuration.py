import logging
logging.basicConfig(level=logging.INFO)

from utils.utils import get_cuda_devices_list, DotDict
from dots_boxes.dots_boxes_game import BoxesState, nn_batch_builder
from dots_boxes.dots_boxes_nn import SimpleNN

BoxesState.set_board_dim((3,3))

root = "./data/"
exp = "simple_2310"
params = DotDict({
    "data_root": root+exp,
    "hdf_file": root+exp+"/sp_data.hdf",
    "tensorboard_log": root+"tboard/"+exp,
    "game": {
        "clazz": BoxesState
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
            #"lr_scheduler": (10, 0.1),
            "adam_params": {
                "betas":(0.9, 0.999),
                "weight_decay":1e-5,
            },
        },
        "model_class":SimpleNN,
        "pytorch_device": "cuda:1",
        "chkpts_filename": root+exp+"/model_gen{}.pt",
        "model_parameters":{
            "resnet": {
                "in_channels": 3,
                "nb_channels": 256,
                "kernel_size": 3,
                "nb_blocks": 20
            },
            "policy_head": {
                "in_channels": 256,
                "nb_actions": BoxesState.NB_ACTIONS,
            },
            "value_head": {
                "in_channels": 256,
                "nb_actions": BoxesState.NB_ACTIONS,
                "n_hidden": 128
            },
        }
    }
})
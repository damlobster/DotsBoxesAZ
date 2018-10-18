import logging
logging.basicConfig(level=logging.INFO)

from dots_boxes.dots_boxes_game import BoxesState, nn_batch_builder


BoxesState.set_board_dim((3,3))


BoxesState.set_board_dim((3,3))
params = utils.DotDict({
    "hdf_file": "./data/selfplay/simple.hdf",
    "game_class": {
        "clazz": BoxesState
    },
    "self_play": {
        "num_games": 2000,
        "reuse_mcts_tree": True,
        "noise": (1.0, 0.25),  # alpha, coeff
        "nn_batch_size": 64,
        "nn_batch_timeout": 0.001,
        "nn_batch_builder": nn_batch_builder,
        "mcts": {
            "mcts_num_read": 1000,
            "mcts_cpuct": 1.0,
            "temperature": {0: 1.0, 5: 1e-50},  # from 8th move we greedly take move with most visit count
            "max_async_searches": 64,
        }
    },
    "nn": {
        "train_params": {
            "nb_epochs": 50,
            "train_split": 0.9,
            "train_batch_size": 256,
            "val_batch_size": 512,
            "lr": 0.01,
            "lr_scheduler": (10, 0.1),
            "adam_betas": (0.9, 0.999),
            "lambda_l2": 1e-4,
        },
        "model_class":SimpleNN,
        "pytorch_device": "cuda:1",
        "chkpts_filename":"./data/model_chkpts/simple_{}.pt",
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
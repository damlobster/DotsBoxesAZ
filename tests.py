import logging
logging.basicConfig(level=logging.INFO)

import asyncio
import sys
import numpy as np
import pickle
import multiprocessing as mp
import torch

from self_play import SelfPlay, generate_games
from nn import NeuralNetWrapper, ResNetZero
from dots_boxes.dots_boxes_game import BoxesState, moves_to_string, nn_batch_builder
from dots_boxes.dots_boxes_nn import SimpleNN
import utils.utils as utils
from utils.proxies import AsyncBatchedProxy

BoxesState.set_board_dim((3,3))
params = utils.DotDict({
    "game": {
        "board_size": BoxesState.BOARD_DIM,
    },
    "self_play": {
        "num_games": 1,
        "reuse_mcts_tree": True,
        "noise": (1.0, 0.25),  # alpha, coeff
        "nn_batch_size": 128,
        "nn_batch_timeout": 0.02,
        "nn_batch_builder": nn_batch_builder,
        "mcts": {
            "mcts_num_read": 1000,
            "mcts_cpuct": 1.0,
            "temperature": {0: 1.0, 5: 1e-50},  # from 8th move we greedly take move with most visit count
            "max_async_searches": 32,
        }
    },
    "nn": {
        "pytorch_device": "cuda:1",
        "train_params": {
            "nb_epochs": 50,
            "train_split": 0.9,
            "train_batch_size": 50,
            "val_batch_size": 512,
            "lr": 0.01,
            "lr_scheduler": (200, 0.5),
            "adam_betas": (0.9, 0.999),
            "lambda_l2": 1e-4,
        },
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
})

resnet = False

def play_game(generation=None):
    if generation is None:
        raise ValueError("You should specify a generation!")

    # change parameters
    params.self_play.mcts.temperature = {0:1.0, 3:1e-50}
    params.self_play.noise = (1.0, 0.0) # remove noise
    params.self_play.mcts.mcts_num_read = 100

    model = ResNetZero(params) if resnet else SimpleNN()
    model.load_parameters(
        "./data/model_chkpts/{}_{}.pt".format("resnet" if resnet else "simple", generation))

    loop = asyncio.get_event_loop()
    nn = NeuralNetWrapper(model, params)
    batched_nn = AsyncBatchedProxy(nn, 32, loop=loop)
    sp = SelfPlay(batched_nn, params, loop)
    game_state = BoxesState()
    sp.play_games(game_state, 1, show_progress=True)
    moves, visit_counts = sp.get_games_moves()

    for i in range(1, len(moves)+1):
        print(moves_to_string(moves[:i], visit_counts[i-1]))
    loop.close()


def check_randomness():
    async def nn(state):
        return np.ones(state.get_actions_size()), 0

    for i in range(10):
        sp = SelfPlay(nn, params)
        game_state = BoxesState()
        sp.play_game(game_state)
        moves, _ = sp.get_games_moves()
        print(moves)

def selfplay(n_games, nn):
    game_state = BoxesState()
    sp = SelfPlay(nn, params)
    sp.play_games(game_state, n_games, show_progress=False)
    sys.stdout.flush()
    return sp.get_training_data()

async def random_nn(state):
    p = np.random.uniform(0, 1, (state.shape[0], 32))
    s = p.sum(axis=1)
    return (p, np.array([0]*state.shape[0]))

def generate_random_games(n_games):
    generate_games("./data/selfplay/test.h5", 0, random_nn, int(n_games), params, n_workers=None, games_per_workers=None)
"""    with mp.Pool(mp.cpu_count()) as pool:
        results = [pool.apply_async(selfplay, args=(10, random_nn))
                    for _ in range(200)]
        data = [sample for p in results for sample in p.get()]
        with open("./data/selfplay/selfplay0.pkl", "wb") as f:
            pickle.dump(list(data), f)
"""

def train_nn(generation=None, file=None, to_idx="1e12", epochs=None):
    print("Generation is the next one for which we train the model", flush=True)
    assert generation is not None
    generation = int(generation)
    if epochs is not None:
        params.nn.train_params.nb_epochs = int(epochs)
        
    ds = utils.PickleDataset("./data/selfplay/", file=file, to_idx=int(float(to_idx)))
    model = ResNetZero(params)  if resnet else SimpleNN()
    if generation != 1:
        model.load_parameters("./data/model_chkpts/{}_{}.pt".format("resnet" if resnet else "simple", generation-1))
    wrapper = NeuralNetWrapper(model, params)
    wrapper.train(ds)
    wrapper.save_model_parameters(
        "./data/model_chkpts/{}_{}.pt".format("resnet" if resnet else "simple", generation))


def predict_nn(generation, from_idx="0", to_idx="10"):
    assert generation is not None
    valid_actions = BoxesState().get_valid_moves(as_indices=True)

    ds = utils.PickleDataset("./data/selfplay/", file="selfplay{}.pkl".format(int(generation)-1), from_idx=int(float(from_idx)), to_idx=int(float(to_idx)))
    model = ResNetZero(params) if resnet else SimpleNN()
    model.load_parameters(
        "./data/model_chkpts/{}_{}.pt".format("resnet" if resnet else "simple", generation))

    pv = list(zip(*model(torch.tensor([s[0] for s in ds]))))
    for i, (_, pi, z) in enumerate(ds):
        p, v = pv[i]
        p = torch.exp(p)
        print("*"*70)
        print("*"*70)
        print("pi= [", ", ".join(map(lambda v: "{:.3f}".format(v), pi[valid_actions])), "]")
        print("p = [", ", ".join(map(lambda v: "{:.3f}".format(v), p[valid_actions])), "]")
        #print("crossentropy=", -pi.T @ np.log(p.tolist()))
        print("*"*50)
        print("z=", z[0])
        print("v=", v.item())


def selfplay_nn(generation, n_game=1):
    print("Generation is the next one for which we generate samples")
    assert generation is not None
    model = ResNetZero(params) if resnet else SimpleNN()
    model.load_parameters(
        "./data/model_chkpts/{}_{}.pt".format("resnet" if resnet else "simple", int(generation)-1))
    nn = NeuralNetWrapper(model, params)

    results = selfplay(int(n_game), nn.predict_from_game)
    with open("./data/selfplay/selfplay{}.pkl".format(generation), "wb") as f:
        pickle.dump(list(results), f)


def async_selfplay(generation, n_games=1000):
    print("Generation is the one for which we generate samples")
    assert generation is not None

    model = ResNetZero(params) if resnet else SimpleNN()
    model.load_parameters(
        "./data/model_chkpts/{}_{}.pt".format("resnet" if resnet else "simple", int(generation)))
    nn = NeuralNetWrapper(model, params)

    game_state = BoxesState()

    loop = asyncio.get_event_loop()    
    batched_nn = AsyncBatchedProxy(nn, 64, loop=loop)
    sp = SelfPlay(batched_nn, params, loop)
    sp.play_games(game_state, int(n_games), show_progress=True)

    results = sp.get_training_data()
    with open("./data/selfplay/selfplay{}.pkl".format(generation), "wb") as f:
        pickle.dump(list(results), f)

    loop.close()
    print("Finished !!!!", flush=True)


if __name__ == '__main__':
    if len(sys.argv) == 1:
        print("Function name missing")
    else:
        function = sys.argv[1]
        args = []
        if len(sys.argv)>2:
            args = sys.argv[2:]
        globals()[function](*args)

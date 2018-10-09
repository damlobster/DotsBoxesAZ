import asyncio
import sys
import numpy as np
import pickle
import multiprocessing as mp
import torch

from self_play import SelfPlay
from nn import NeuralNetWrapper, ResNetZero
from dots_boxes.dots_boxes_game import BoxesState, moves_to_string
from dots_boxes.dots_boxes_nn import SimpleNN
import utils
from utils import AsyncBatchedProxy

params = utils.DotDict({
    "game": {
        "board_size": BoxesState.BOARD_DIM,
    },
    "self_play": {
        "num_games": 1,
        "reuse_mcts_tree": True,
        "noise": (1.0, 0.25),  # alpha, coeff
        "mcts": {
            "mcts_num_read": 1000,
            "mcts_cpuct": 1.0,
            "temperature": {0: 1.0, 5: 1e-50},  # {0:1.0}
            "max_async_searches": 32,
        }
    },
    "nn": {
        "pytorch_device": "cuda:1",
        "train_params": {
            "nb_epochs": 50,
            "train_split": 0.8,
            "train_batch_size": 256,
            "val_batch_size": 1024,
            "lr": {0:0.15, 10:0.1, 20:0.01},
            "adam_betas": (0.9, 0.999),
            "lambda_l2": 0.0 #1e-4
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

def play_game(generation):
    if generation is None:
        raise ValueError("You should specify a generation!")
    
    # change parameters
    params.self_play.mcts.temperature = {0:1.0, 3:1e-50}
    params.self_play.noise = (1.0, 0.0)
    params.self_play.mcts.mcts_num_read = 100

    model = ResNetZero(params) if resnet else SimpleNN()
    #model.load_parameters("./temp/tests_nn_model_{}.pkl".format(generation))
    nn = NeuralNetWrapper(model, params)
    sp = SelfPlay(nn.predict_from_game, params)
    game_state = BoxesState()
    sp.play_game(game_state)
    moves, visit_counts = sp.get_games_moves()

    for i in range(1, len(moves)+1):
        print(moves_to_string(moves[:i], visit_counts[i-1]))


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
    sp.play_games(game_state, n_games, show_progress=True)
    return sp.get_training_data()

async def random_nn(state):
        return (np.ones(state.get_actions_size()), 0)

def generate_random_games():
    with mp.Pool(mp.cpu_count()) as pool:
        for i in range(1):
            results = [pool.apply_async(selfplay, args=(1, random_nn))
                       for _ in range(12)]
            data = [sample for p in results for sample in p.get()]
            with open("./data/selfplay{}.pkl".format(i), "wb") as f:
                pickle.dump(list(data), f)


def train_nn(generation="1", to_idx="1e12"):
    print("Generation is the next one for which we train the model", flush=True)
    assert generation is not None
    generation = int(generation)
    ds = utils.PickleDataset("./data/", file="selfplay{}.pkl".format(generation-1), to_idx=int(to_idx))
    print(ds[0])
    model = ResNetZero(params)  if resnet else SimpleNN()
    #model.load_parameters("./temp/tests_nn_model_{}.pkl".format(generation-1))
    wrapper = NeuralNetWrapper(model, params)
    wrapper.train(ds)
    wrapper.save_model_parameters("./temp/tests_nn_model_{}.pkl".format(generation))


def predict_nn(generation):
    assert generation is not None
    valid_actions = BoxesState().get_valid_moves(as_indices=True)

    ds = utils.PickleDataset("./data/", from_idx=10000, to_idx=10100)
    model = ResNetZero(params) if resnet else SimpleNN()
    model.load_parameters("./temp/tests_nn_model_{}.pkl".format(generation))

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
    model.load_parameters("./temp/tests_nn_model_{}.pkl".format(int(generation)-1))
    nn = NeuralNetWrapper(model, params)

    results = selfplay(int(n_game), nn.predict_from_game)
    with open("./data/selfplay{}.pkl".format(generation), "wb") as f:
        pickle.dump(list(results), f)


def async_selfplay(generation, n_games=1000):
    print("Generation is the next one for which we generate samples")
    assert generation is not None

    model = ResNetZero(params) if resnet else SimpleNN()
    model.load_parameters(
        "./temp/tests_nn_model_{}.pkl".format(int(generation)-1))
    nn = NeuralNetWrapper(model, params)

    build_X = lambda states_batch: \
        np.concatenate(tuple(gs.game_state.get_features() for gs in states_batch))

    batched_nn = AsyncBatchedProxy(nn, build_X, 64)

    game_state = BoxesState()
    sp = SelfPlay(batched_nn, params)

    loop = asyncio.get_event_loop()
    loop.run_until_complete(asyncio.gather(batched_nn.batch_runner(),
        sp.play_games(game_state, int(n_games), show_progress=True)))
    loop.close()

    results = sp.get_training_data()
    with open("./data/async_selfplay{}.pkl".format(generation), "wb") as f:
        pickle.dump(list(results), f)

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

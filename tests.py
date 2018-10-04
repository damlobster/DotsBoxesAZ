import sys
import numpy as np
import pickle
import multiprocessing as mp
import torch

from self_play import SelfPlay
from nn import NeuralNetWrapper, ResNetZero
from dots_boxes.dots_boxes_game import BoxesState, moves_to_string
import utils

params = utils.DotDict({
    "self_play": {
        "num_games": 1,
        "reuse_mcts_tree": True,
        "noise": (1.0, 0.25),  # alpha, coeff
        "mcts":{
            "mcts_num_read": 1000,
            "mcts_cpuct": 1.0,
            "temperature": {0: 1.0, 5: 1e-50},  # {0:1.0}
        }
    },
    "nn": {
        "train_params": {
            "nb_epochs": 10,
            "train_split": 0.8,
            "train_batch_size": 8,
            "val_batch_size": 1024,
            "lr": 0.2,
            "adam_betas": (0.9, 0.999),
            "lambda_l2": 0.0002
        },
        "resnet": {
            "in_channels": 3,
            "nb_channels": 128,
            "kernel_size": 3,
            "nb_blocks": 5
        },
        "policy_head": {
            "in_channels": 128,
            "in_fc": 64,
            "nb_actions": 32
        },
        "value_head": {
            "in_channels": 128,
            "in_fc": 32,
            "n_hidden": 32
        },
    }
})

def test():
  sp = SelfPlay(lambda state: (np.ones(state.get_actions_size()), 0), params)
  game_state = BoxesState()
  sp.play_game(game_state)
  moves, visit_counts = sp.get_games_moves()

  for i in range(1, len(moves)+1):
    print(moves_to_string(moves[:i], visit_counts[i-1]))


def check_randomness():
  for i in range(20):
    sp = SelfPlay(lambda state: (np.ones(state.get_actions_size()), 0), params)
    game_state = BoxesState()
    sp.play_game(game_state)
    moves, _ = sp.get_games_moves()
    print(moves)


def worker(n_games, j):
  print(str(j))
  game_state = BoxesState()
  sp = SelfPlay(lambda state: (np.ones(state.get_actions_size()), 0), params)
  sp.play_games(game_state, n_games, show_progress=True)
  return sp.get_training_data()

def generate_games():
  with mp.Pool(mp.cpu_count()) as pool:
    for i in range(1):
      results = [pool.apply_async(worker, args=(10, j)) for j in range(10)]
      data = [p.get() for p in results]
      with open("./data/selfplay{}.pkl".format(i), "wb") as f:
        pickle.dump(list(data), f)

def train_nn():
  ds = utils.PickleDataset("./data/", size_limit=int(500))
  model = ResNetZero(params.nn)
  print(model)
  wrapper = NeuralNetWrapper(model, params.nn)
  wrapper.train(ds)
  wrapper.save_model_parameters("./temp/tests_nn_model.pkl")


def predict_nn():
  ds = utils.PickleDataset("./data/", size_limit=int(10))
  model = ResNetZero(params.nn)
  #model.load_parameters("./temp/tests_nn_model.pkl")
  
  for board, pi, z in ds:
    p, v = model(torch.tensor([board]))
    print("*"*70)
    print("pi=", pi)
    print("p =",p[0].tolist())
    print("*"*50)
    print("z=", z[0])
    print("v=", v.item())


if __name__ == '__main__':
  if len(sys.argv)==1:
    print("Function name missing")
  else:
    function = sys.argv[1]
    globals()[function]()

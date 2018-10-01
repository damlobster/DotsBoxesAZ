import torch
import utils
from game import GameState
from nn import NeuralNetWrapper, ResBlock, ResNet, PolicyHead, ValueHead

params = utils.DotDict({
  "nn": {
    "resnet": {
        "in_channels": 3,
        "nb_channels": 128, 
        "kernel_size": 3, 
        "nb_blocks": 20
    },
    "policy_head": {
        "in_channels": 128,
        "nb_actions": 32
    },
    "value_head": {
        "in_channels": 128,
        "n_hidden": 64
    },
    "train_params":{
      "batch_size": 256,
      "lr": 0.2,
    }
  }
})

class DotsNN(NeuralNetWrapper):
  def __init__(self, checkpoint=None):
    self.resnet = None
    self.policy_head = None
    self.value_head = None
    self.cache = {}

  def build_model(self):
    cfg = params.nn
    self.resnet = ResNet(**cfg.resnet)
    self.policy_head = PolicyHead(**cfg.policy_head)
    self.value_head = ValueHead(**cfg.value_head)

  def _forward(self, x):
    x = self.resnet.forward(x)
    policy = self.policy_head.forward(x)
    value = self.value_head.forward(x)
    return policy, value

  def predict(self, game_state):
    if game_state in self.cache:
      return self.cache[game_state]

    pv = self._forward(game_state.get_features())
    self.cache[game_state] = pv
    return pv

  def train(self, X, policies, values):
    pass

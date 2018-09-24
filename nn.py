from game import GameState
import numpy as np
import torch

class NeuralNet():
  @classmethod
  def predict(self, game_state):
    return np.random.random(game_state.get_action_size()), np.random.random()

  def train(self, X, policies, values):
    pass
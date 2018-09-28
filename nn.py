import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

import utils
from game import GameState

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
        "train_params": {
            "batch_size": 256,
            "lr": 0.2,
            "adam_betas": (0.9, 0.999)
        }
    }
})

class ResBlock(nn.Module):
  def __init__(self, nb_channels, kernel_size):
    super(ResBlock, self).__init__()
    self.conv1 = nn.Conv2d(nb_channels, nb_channels, kernel_size,
                          padding=(kernel_size-1)//2)
    self.bn1 = nn.BatchNorm2d(nb_channels)
    self.conv2 = nn.Conv2d(nb_channels, nb_channels, kernel_size,
                          padding=(kernel_size-1)//2)
    self.bn2 = nn.BatchNorm2d(nb_channels)

    def forward(self, x):
      y = self.bn1(self.conv1(x))
      y = F.relu(y)
      y = self.bn2(self.conv2(y))
      y += x
      y = F.relu(y)
      return y

class ResNet(nn.Module):
  def __init__(self, in_channels, nb_channels, kernel_size, nb_blocks):
    super(ResNet, self).__init__()
    self.conv0 = nn.Conv2d(in_channels, nb_channels, kernel_size=1)
    self.resblocks = nn.Sequential(
        # A bit of fancy Python
        *(ResBlock(nb_channels, kernel_size) for _ in range(nb_blocks))
    )
    #self.avg = nn.AvgPool2d(kernel_size=28)
    #self.fc = nn.Linear(nb_channels, 10)

  def forward(self, x):
    x = F.relu(self.conv0(x))
    x = self.resblocks(x)
    #x = F.relu(self.avg(x))
    #x = x.view(x.size(0), -1)
    #x = self.fc(x)
    return x

class PolicyHead(nn.Module):
  def __init__(self, in_channels, nb_actions):
    super(PolicyHead, self).__init__()
    self.conv0 = nn.Conv2d(in_channels, 2, kernel_size=1)
    self.bn0 = nn.BatchNorm2d(2)
    self.fc = nn.Linear(2*in_channels, nb_actions)

  def forward(self, x):
    x = self.conv0(x)
    x = F.relu(self.bn0(x))
    x = self.fc(x)
    p = F.log_softmax(x)
    return p

class ValueHead(nn.Module):
  def __init__(self, in_channels, n_hidden):
    super(ValueHead, self).__init__()
    self.conv0 = nn.Conv2d(in_channels, 1, kernel_size=1)
    self.bn0 = nn.BatchNorm2d(1)
    self.fc0 = nn.Linear(in_channels, n_hidden)
    self.fc1 = nn.Linear(n_hidden, 1)

  def forward(self, x):
    x = self.conv0(x)
    x = F.relu(self.bn0(x))
    x = F.relu(self.fc0(x))
    x = F.relu(self.fc1(x))
    v = F.tanh(x)
    return v  

class NeuralNetWrapper():
  def __init__(self, game, model, params):
    self.cache = {}
    self.params = params
    self.device = torch.device(
        params.pytorch_device if torch.cuda.is_available() else "cpu")
    self.game = game
    self.model = model.to(device)

  def predict(self, game_state: GameState):
    if game_state in self.cache:
      return self.cache[game_state]

    x = torch.tensor(game_state.get_features(),
                     dtype=torch.float32, device=self.device)
    pv = self.model.forward(x).cpu()
    self.cache[game_state] = pv
    return pv

  def train(self, samples):
    params = self.params.train_params
    self.cache = {}

    optimizer = torch.optim.Adam(self.model.parameters(), lr=params.lr, betas=params.adam_betas)

    for epoch in range(self.params.nb_epochs):
      pass

  def loss(self, v, z, p, pi):
    loss_v = (z - v).pow(2).sum()
    loss_pi = (pi.t() @ p.log()).sum()
    loss = (loss_v + loss_pi) / v.size(0)
    for p in self.model.parameters():
      loss += self.params.lambda_l2 * p.pow(2).sum()
    return loss

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import torch.utils.data as data

import utils

from game import GameState


class ResNet(nn.Module):
    def __init__(self, in_channels, nb_channels, kernel_size, nb_blocks):
        super(ResNet, self).__init__()
        self.conv0 = _create_conv_layer(in_channels, nb_channels, kernel_size)
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


class ResBlock(nn.Module):
    def __init__(self, nb_channels, kernel_size):
        super(ResBlock, self).__init__()
        self.conv1 = _create_conv_layer(nb_channels, nb_channels, kernel_size)
        self.bn1 = nn.BatchNorm2d(nb_channels)
        self.conv2 = _create_conv_layer(nb_channels, nb_channels, kernel_size)
        self.bn2 = nn.BatchNorm2d(nb_channels)

    def forward(self, x):
        x = self.bn1(self.conv1(x))
        x = F.relu(x)
        x = self.bn2(self.conv2(x))
        x += x
        x = F.relu(x)
        return x


def _create_conv_layer(in_channels, out_channels, kernel_size):
    if kernel_size % 2 == 0:
        pad = nn.ConstantPad2d((0, kernel_size//2, 0, kernel_size//2), 0.0)
        conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=0)
        return nn.Sequential(pad, conv)
    else:
        return nn.Conv2d(in_channels, out_channels, kernel_size,
                         padding=(kernel_size-1)//2)


class PolicyHead(nn.Module):
    def __init__(self, in_channels, nb_actions):
        super(PolicyHead, self).__init__()
        self.conv0 = nn.Conv2d(in_channels, 2, kernel_size=1)
        self.bn0 = nn.BatchNorm2d(2)
        self.fc = nn.Linear(nb_actions, nb_actions)

    def forward(self, x):
        x = self.conv0(x)
        x = F.relu(self.bn0(x))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        p = F.log_softmax(x, dim=1)
        return p


class ValueHead(nn.Module):
    def __init__(self, in_channels, nb_actions, n_hidden):
        super(ValueHead, self).__init__()
        self.conv0 = nn.Conv2d(in_channels, 1, kernel_size=1)
        self.bn0 = nn.BatchNorm2d(1)
        self.fc0 = nn.Linear(nb_actions//2, n_hidden)
        self.fc1 = nn.Linear(n_hidden, 1)

    def forward(self, x):
        x = self.conv0(x)
        x = self.bn0(F.relu(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc0(x))
        x = F.relu(self.fc1(x))
        v = torch.tanh(x)
        return v


class ResNetZero(nn.Module):
    def __init__(self, params):
        super(ResNetZero, self).__init__()
        self.params = params.nn
        self.resnet = ResNet(**params.nn.resnet)
        self.value_head = ValueHead(**params.nn.value_head)
        self.policy_head = PolicyHead(**params.nn.policy_head)

    def forward(self, x):
        x = self.resnet(x)
        p = self.policy_head(x)
        v = self.value_head(x)
        return (p, v)

    def save_parameters(self, filename):
        print("Model saved to:", filename)
        torch.save(self.state_dict(), filename)

    def load_parameters(self, filename):
        print("Model loaded from:", filename)
        self.load_state_dict(torch.load(filename, map_location=lambda storage, loc: storage))


class NeuralNetWrapper():
    def __init__(self, model, params):
        self.params = params
        self.device = torch.device(
            params.nn.pytorch_device if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)

    def predict_sync(self, X):
        self.model.train(False)

        x = torch.tensor(X, dtype=torch.float32, device=self.device)
        p, v = self.model.forward(x)
        p, v = torch.exp(p).cpu().detach().numpy(), v.cpu().detach().numpy()
        return (p, v)

    async def predict(self, X):
        return self.predict_sync(X)

    async def predict_from_game(self, game_state):
        return await self.predict([game_state.get_features()])

    async def __call__(self, X):
        return await self.predict(X)

    def train(self, dataset):
        params = self.params.nn.train_params

        optimizer = None

        split_idx = int(len(dataset)*params.train_split)
        train_data, validation_data = data.random_split(
            dataset, [split_idx, len(dataset)-split_idx])
        train_data = data.DataLoader(
            train_data, params.train_batch_size, shuffle=True)
        validation_data = data.DataLoader(
            validation_data, params.val_batch_size)

        for epoch in range(params.nb_epochs):
            if epoch in params.lr:
                print("Update lr: ", params.lr[epoch])
                optimizer = torch.optim.Adam(
                    self.model.parameters(), lr=params.lr[epoch], betas=params.adam_betas)
            self.model.train(True)
            train_loss = 0.0
            for boards, pi, z in train_data:
                # Transfer to GPU
                boards = boards.to(self.device)
                pi = pi.requires_grad_(True).to(self.device)
                z = z.requires_grad_(True).to(self.device)

                p, v = self.model(boards)
                loss_pi = self.loss_pi(p, pi)
                loss_v = self.loss_v(v, z)
                loss_reg = self.loss_reg()
                loss = (loss_v + loss_pi + loss_reg)/len(train_data)
                train_loss += loss.item()
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            self.model.train(False)
            val_loss = 0.0
            for boards, pi, z in validation_data:
                # Transfer to GPU
                boards = boards.to(self.device)
                pi = pi.to(self.device)
                z = z.to(self.device)

                p, v = self.model(boards)
                loss_pi = self.loss_pi(p, pi)
                loss_v = self.loss_v(v, z)
                loss_reg = 0.0 #self.loss_reg()
                loss = (loss_v + loss_pi + loss_reg) / len(validation_data)
                val_loss += loss.item()

            print("Epoch {}, train loss= {:5f}, validation loss= {:5f}".format(
                epoch+1, train_loss, val_loss,))

    def loss_pi(self, p, pi):
        loss_pi = -(pi * p).sum()/pi.size()[0]
        return loss_pi

    def loss_v(self, v, z):
        loss_v = (z - v).pow(2).sum()/z.size()[0]
        return loss_v

    def loss_reg(self):
        loss_reg = torch.tensor(0.0, dtype=torch.float, device=self.device)
        for p in self.model.parameters():
            loss_reg += self.params.nn.train_params.lambda_l2 * p.pow(2).sum()
        return loss_reg

    def save_model_parameters(self, filename):
        self.model.save_parameters(filename)

    def load_model_parameters(self, filename):
        self.model.load_parameters(filename)

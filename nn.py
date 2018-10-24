import asyncio
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import torch.utils.data as data
import pandas as pd

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
        _x = self.bn1(self.conv1(x))
        _x = F.relu(_x)
        _x = self.bn2(self.conv2(_x))
        _x += x
        _x = F.relu(_x)
        return _x


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

class AlphaZeroLoss(nn.Module):
    def __init__(self):
        super(AlphaZeroLoss, self).__init__()

    def forward(self, p, v, pi, z):
        loss_v = (z - v).pow(2).mean()
        loss_pi = -(pi * p).sum(1).mean()
        return loss_v + loss_pi, (loss_pi.item(), loss_v.item()) #loss:variable, (loss_v:float, loss_pi:float)

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
        loop = asyncio.get_event_loop()
        future = loop.run_in_executor(None, self.predict_sync, X)
        return await future

    async def predict_from_game(self, game_state):
        return await self.predict([game_state.get_features()])

    async def __call__(self, X):
        res = await self.predict(X)
        return res

    def train(self, train_dataset, val_dataset, writer, last_batch_idx):
        params = self.params.nn.train_params
        
        train_data = data.DataLoader(
            train_dataset, params.train_batch_size, shuffle=True)
        validation_data = data.DataLoader(
            val_dataset, params.val_batch_size) if val_dataset is not None else None

        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=params.lr, **params.adam_params)
        #lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, *params.lr_scheduler)

        criterion = AlphaZeroLoss()
        batch_i = last_batch_idx + 1
        for epoch in range(params.nb_epochs):
            #lr_scheduler.step() #TODO use batch number

            self.model.train(True)
            n_batches = len(train_data)
            tr_loss = 0
            for boards, pi, z in train_data:
                batch_i += 1
                # Transfer to GPU
                boards = boards.to(self.device)
                pi = pi.requires_grad_(True).to(self.device)
                z = z.requires_grad_(True).to(self.device)

                p, v = self.model(boards)
                loss, (loss_pi, loss_v) = criterion(p, v, pi, z)
                loss = loss/n_batches
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                loss_v, loss_pi = loss_v/n_batches, loss_pi/n_batches
                tr_loss += loss_pi + loss_v
                # write scalar to tensorboard
                writer.add_scalar('data/loss/train/pi', loss_pi, batch_i) #, walltime=batch_i)
                writer.add_scalar('data/loss/train/v', loss_v, batch_i)
                writer.add_scalar('data/loss/train/total', loss_pi + loss_v, batch_i)
                writer.add_scalar("data/lr", params.lr, batch_i)
            
            val_loss = 0.0
            if val_dataset:
                self.model.train(False)
                loss_pi, loss_v = 0.0, 0.0
                for boards, pi, z in validation_data:
                    # Transfer to GPU
                    boards = boards.to(self.device)
                    pi = pi.to(self.device)
                    z = z.to(self.device)

                    p, v = self.model(boards)
                    loss, (loss_pi, loss_v) = criterion(p, v, pi, z)
                    loss_v, loss_pi = loss_v/n_batches, loss_pi/n_batches
                loss = loss.item()
                val_loss += loss
                writer.add_scalar('data/loss/val/pi', loss_pi, batch_i)
                writer.add_scalar('data/loss/val/v', loss_v, batch_i)
                writer.add_scalar('data/loss/val/total', loss, batch_i)

            print("Epoch {}, train loss= {:5f}, validation loss= {:5f}".format(
                epoch, tr_loss, val_loss,), flush=True)
        
        return batch_i

    def save_model_parameters(self, generation=None):
        if generation:
            self.model.save_parameters(filename.format(generation))
        else:
            self.model.save_parameters(filename)

    def load_model_parameters(self, generation, to_device=None):
        self.model.load_parameters(generation, to_device=to_device)

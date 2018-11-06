import logging
logger = logging.getLogger(__name__)

import asyncio
import numpy as np
import os
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
        self.bn0 = nn.BatchNorm2d(nb_channels)
        self.resblocks = nn.Sequential(
            *(ResBlock(nb_channels, kernel_size) for _ in range(nb_blocks))
        )

    def forward(self, x):
        x = self.conv0(x)
        x = F.relu(self.bn0(x))
        x = self.resblocks(x)
        return x


class ResBlock(nn.Module):
    def __init__(self, nb_channels, kernel_size):
        super(ResBlock, self).__init__()
        self.conv1 = _create_conv_layer(nb_channels, nb_channels, kernel_size)
        self.bn1 = nn.BatchNorm2d(nb_channels)
        self.conv2 = _create_conv_layer(nb_channels, nb_channels, kernel_size)
        self.bn2 = nn.BatchNorm2d(nb_channels)

    def forward(self, x):
        _x = self.conv1(x)
        _x = F.relu(self.bn1(_x))
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
    def __init__(self, in_channels, inner_channels, fc_in, nb_actions):
        super(PolicyHead, self).__init__()
        self.conv0 = nn.Conv2d(in_channels, inner_channels, kernel_size=(1,1))
        self.bn0 = nn.BatchNorm2d(inner_channels)
        self.fc = nn.Linear(fc_in, nb_actions)

    def forward(self, x):
        x = self.conv0(x)
        x = F.relu(self.bn0(x))
        x = self.fc(x.view(x.size(0), -1))
        p = F.log_softmax(x, dim=1)
        return p


class ValueHead(nn.Module):
    def __init__(self, in_channels, inner_channels, fc_in, fc_inner):
        super(ValueHead, self).__init__()
        self.conv0 = nn.Conv2d(in_channels, inner_channels, kernel_size=(1,1))
        self.bn0 = nn.BatchNorm2d(inner_channels)
        self.fc0 = nn.Linear(fc_in, fc_inner)
        self.fc1 = nn.Linear(fc_inner, 1)

    def forward(self, x):
        x = self.conv0(x)
        x = F.relu(self.bn0(x))
        x = F.relu(self.fc0(x.view(x.size(0), -1)))
        x = self.fc1(x)
        v = torch.tanh(x)
        return v


class ResNetZero(nn.Module):
    def __init__(self, params):
        super(ResNetZero, self).__init__()
        self.params = params
        self.resnet = ResNet(**params.nn.model_parameters.resnet)
        self.value_head = ValueHead(**params.nn.model_parameters.value_head)
        self.policy_head = PolicyHead(**params.nn.model_parameters.policy_head)

    def forward(self, x):
        x = self.resnet(x)
        p = self.policy_head(x)
        v = self.value_head(x)
        return (p, v)

    def save_parameters(self, generation):
        raise ValueError("Do not call!!!")
        #filename = self.params.nn.chkpts_filename
        #fn = filename.format(generation)
        #logger.info("Model saved to: %s", fn)
        #torch.save(self.state_dict(), fn)

    def load_parameters(self, generation, to_device=None):
        filename = self.params.nn.chkpts_filename
        fn = filename.format(generation)
        logger.info("Model loaded from: %s", fn)
        self.load_state_dict(torch.load(fn, map_location='cpu')['model_dict'])
        self.to(to_device)

class AlphaZeroLoss(nn.Module):
    def __init__(self):
        super(AlphaZeroLoss, self).__init__()

    def forward(self, p, v, pi, z):
        loss_v = (z - v).pow(2).mean()
        loss_pi = -(pi * p).sum(1).mean()
        return loss_v + loss_pi, (loss_pi.item(), loss_v.item()) #loss:variable, (loss_v:float, loss_pi:float)

def _err_cb(fut):
    if fut.exception():
        logger.error(fut.exception())
        raise fut.exception()

class NeuralNetWrapper():
    def __init__(self, model, params):
        self.params = params
        self.device = torch.device(
            params.nn.pytorch_device if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device) if model is not None else None

    def set_model(self, model):
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
        future.add_done_callback(_err_cb) # to re-raise if exception occured in executor
        return await future

    async def predict_from_game(self, game_state):
        return await self.predict([game_state.get_features()])

    async def __call__(self, X):
        res = await self.predict(X)
        return res

    def train(self, train_dataset, val_dataset, writer, generation):
        #self.model = nn.DataParallel(self.model, device_ids=[1, 0])
        params = self.params.nn.train_params

        train_data = data.DataLoader(
            train_dataset, params.train_batch_size, shuffle=True, drop_last=True)
        validation_data = data.DataLoader(
            val_dataset, params.val_batch_size) if val_dataset is not None else None

        criterion = AlphaZeroLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=params.lr, **params.adam_params)
        batch_i = 0
        if generation > 0:
            filename = self.params.nn.chkpts_filename.format(generation-1)
            batch_i = load_checkpoint(filename, self.model, optimizer, self.device)

        logger.warning(f"LR={params.lr}")
        writer.add_scalar("lr", params.lr, batch_i)
        for epoch in range(params.nb_epochs):

            self.model.train(True)
            tr_loss = 0
            for boards, pi, z in train_data:
                batch_i += 1
                # Transfer to GPU
                boards = boards.to(self.device)
                pi = pi.requires_grad_(True).to(self.device)
                z = z.requires_grad_(True).to(self.device)

                p, v = self.model(boards)
                loss, (loss_pi, loss_v) = criterion(p, v, pi, z)
                #loss = loss/n_batches
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                loss_v, loss_pi = loss_v, loss_pi
                tr_loss += loss_pi + loss_v
                # write scalars to tensorboard
                writer.add_scalars('loss', {'pi/train': loss_pi, 'v/train':loss_v, 'total/train':loss_pi+loss_v}, batch_i)
            
            val_loss = 0.0
            loss_v = 0.0
            loss_pi = 0.0
            if val_dataset:
                self.model.train(False)
                loss_pi, loss_v = 0.0, 0.0
                for boards, pi, z in validation_data:
                    # Transfer to GPU
                    boards = boards.to(self.device)
                    pi = pi.to(self.device)
                    z = z.to(self.device)

                    p, v = self.model(boards)
                    _, (_loss_pi, _loss_v) = criterion(p, v, pi, z)
                    loss_v += _loss_v
                    loss_pi += _loss_pi

                # get eval loss average
                n_batches = len(validation_data)
                loss_v /= n_batches
                loss_pi /= n_batches
                val_loss = loss_v + loss_pi
                writer.add_scalars('loss', {'pi/eval': loss_pi, 'v/eval':loss_v, 'total/eval':val_loss}, batch_i) #, walltime=batch_i)

            print(f"Epoch {epoch}, train loss= {tr_loss/len(train_data):5f}, validation loss= {val_loss:5f}", flush=True)

        filename = self.params.nn.chkpts_filename.format(generation)
        save_checkpoint(filename, self.model, optimizer, batch_i)
        return batch_i

class GenerationLrScheduler(object):
    def __init__(self, schedule):
        assert schedule is not None
        self.schedule = schedule

    def __call__(self, generation):
        lr = None
        for g in range(generation+1):
            if g in self.schedule:
                lr = self.schedule[g]
        assert lr is not None
        return lr

    def __repr__(self):
        return f"GenerationLrScheduler({self.schedule})"


def save_checkpoint(filename, model, optimizer, last_batch_idx):
    state = {'last_batch_idx': last_batch_idx, 'model_dict': model.state_dict(),
             'optimizer_dict': optimizer.state_dict()}
    torch.save(state, filename)

def load_checkpoint(filename, model, optimizer, to_device):
    start_epoch = 0
    if os.path.isfile(filename):
        logger.info(f"=> loading checkpoint '{filename}'")
        checkpoint = torch.load(filename, map_location='cpu')
        last_batch_idx = checkpoint['last_batch_idx']
        model.load_state_dict(checkpoint['model_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_dict'])
    else:
        raise ValueError(f"=> no checkpoint found at '{filename}'")


    model.to(to_device)
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(to_device)

    return last_batch_idx

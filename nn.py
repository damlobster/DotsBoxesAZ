import Optimizer
import asyncio
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import torch.utils.data as data
from tensorboardX import SummaryWriter
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
        return await future

    async def predict_from_game(self, game_state):
        return await self.predict([game_state.get_features()])

    async def __call__(self, X):
        res = await self.predict(X)
        return res

    def train(self, train_dataset, val_dataset, last_batch_idx):
        params = self.params.nn.train_params
        
        train_log = SummaryWriter(params.tensorboard_log+"/train")
        eval_log = SummaryWriter(params.tensorboard_log+"/eval")

        train_data = data.DataLoader(
            train_dataset, params.train_batch_size, shuffle=True)
        validation_data = data.DataLoader(
            val_dataset, params.val_batch_size) if val_dataset is not None else None

        criterion = AlphaZeroLoss()

        batch_i = last_batch_idx + 1
        lr = params.lr
        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=lr, **params.adam_params)
        lr_scheduler = CyclicLR(optimizer,
                                base_lr=lr,
                                max_lr=lr*params.lr_scheduler.max_lr_factor,
                                step_size=params.lr_scheduler.step_size,
                                last_batch_iteration=batch_i)

        for epoch in range(params.nb_epochs):

            self.model.train(True)
            tr_loss = 0
            for boards, pi, z in train_data:
                scheduler.batch_step()
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
                train_log.add_scalar('loss/pi', loss_pi, batch_i) #, walltime=batch_i)
                train_log.add_scalar('loss/v', loss_v, batch_i)
                train_log.add_scalar('loss/total', loss_pi + loss_v, batch_i)
                train_log.add_scalar("lr", params.lr, batch_i)
            
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
                    _, (loss_pi, loss_v) = criterion(p, v, pi, z)
                    loss_v += loss_v
                    loss_pi += loss_pi

                # get eval loss average
                n_batches = len(validation_data)
                loss_v /= n_batches
                loss_pi /= n_batches
                val_loss = loss_v + loss_pi
                eval_log.add_scalar('loss/pi', loss_pi, batch_i)
                eval_log.add_scalar('loss/v', loss_v, batch_i)
                eval_log.add_scalar('loss/total', val_loss, batch_i)

            print("Epoch {}, train loss= {:5f}, validation loss= {:5f}".format(
                epoch, tr_loss/len(train_data), val_loss), flush=True)

        train_log.export_scalars_to_json(params.tensorboard_log+"train_{}.json".format(last_batch_idx))
        train_log.close()
        train_log.export_scalars_to_json(params.tensorboard_log+"eval_{}.json".format(last_batch_idx))
        eval_log.close()

        return batch_i

    def save_model_parameters(self, generation=None):
        if generation:
            self.model.save_parameters(filename.format(generation))
        else:
            self.model.save_parameters(filename)

    def load_model_parameters(self, generation, to_device=None):
        self.model.load_parameters(generation, to_device=to_device)


class CyclicLR(object):
    """Sets the learning rate of each parameter group according to
    cyclical learning rate policy (CLR). The policy cycles the learning
    rate between two boundaries with a constant frequency, as detailed in
    the paper `Cyclical Learning Rates for Training Neural Networks`_.
    The distance between the two boundaries can be scaled on a per-iteration
    or per-cycle basis.
    Cyclical learning rate policy changes the learning rate after every batch.
    `batch_step` should be called after a batch has been used for training.
    To resume training, save `last_batch_iteration` and use it to instantiate `CycleLR`.
    This class has three built-in policies, as put forth in the paper:
    "triangular":
        A basic triangular cycle w/ no amplitude scaling.
    "triangular2":
        A basic triangular cycle that scales initial amplitude by half each cycle.
    "exp_range":
        A cycle that scales initial amplitude by gamma**(cycle iterations) at each
        cycle iteration.
    This implementation was adapted from the github repo: `bckenstler/CLR`_
    Args:
        optimizer (Optimizer): Wrapped optimizer.
        base_lr (float or list): Initial learning rate which is the
            lower boundary in the cycle for eachparam groups.
            Default: 0.001
        max_lr (float or list): Upper boundaries in the cycle for
            each parameter group. Functionally,
            it defines the cycle amplitude (max_lr - base_lr).
            The lr at any cycle is the sum of base_lr
            and some scaling of the amplitude; therefore
            max_lr may not actually be reached depending on
            scaling function. Default: 0.006
        step_size (int): Number of training iterations per
            half cycle. Authors suggest setting step_size
            2-8 x training iterations in epoch. Default: 2000
        mode (str): One of {triangular, triangular2, exp_range}.
            Values correspond to policies detailed above.
            If scale_fn is not None, this argument is ignored.
            Default: 'triangular'
        gamma (float): Constant in 'exp_range' scaling function:
            gamma**(cycle iterations)
            Default: 1.0
        scale_fn (function): Custom scaling policy defined by a single
            argument lambda function, where
            0 <= scale_fn(x) <= 1 for all x >= 0.
            mode paramater is ignored
            Default: None
        scale_mode (str): {'cycle', 'iterations'}.
            Defines whether scale_fn is evaluated on
            cycle number or cycle iterations (training
            iterations since start of cycle).
            Default: 'cycle'
        last_batch_iteration (int): The index of the last batch. Default: -1
    Example:
        >>> optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
        >>> scheduler = torch.optim.CyclicLR(optimizer)
        >>> data_loader = torch.utils.data.DataLoader(...)
        >>> for epoch in range(10):
        >>>     for batch in data_loader:
        >>>         scheduler.batch_step()
        >>>         train_batch(...)
    .. _Cyclical Learning Rates for Training Neural Networks: https://arxiv.org/abs/1506.01186
    .. _bckenstler/CLR: https://github.com/bckenstler/CLR
    """
    def __init__(self, optimizer, base_lr=1e-3, max_lr=6e-3,
                 step_size=2000, mode='triangular', gamma=1.,
                 scale_fn=None, scale_mode='cycle', last_batch_iteration=-1):

        if not isinstance(optimizer, Optimizer):
            raise TypeError('{} is not an Optimizer'.format(
                type(optimizer).__name__))
        self.optimizer = optimizer

        if isinstance(base_lr, list) or isinstance(base_lr, tuple):
            if len(base_lr) != len(optimizer.param_groups):
                raise ValueError("expected {} base_lr, got {}".format(
                    len(optimizer.param_groups), len(base_lr)))
            self.base_lrs = list(base_lr)
        else:
            self.base_lrs = [base_lr] * len(optimizer.param_groups)

        if isinstance(max_lr, list) or isinstance(max_lr, tuple):
            if len(max_lr) != len(optimizer.param_groups):
                raise ValueError("expected {} max_lr, got {}".format(
                    len(optimizer.param_groups), len(max_lr)))
            self.max_lrs = list(max_lr)
        else:
            self.max_lrs = [max_lr] * len(optimizer.param_groups)

        self.step_size = step_size

        if mode not in ['triangular', 'triangular2', 'exp_range'] \
                and scale_fn is None:
            raise ValueError('mode is invalid and scale_fn is None')

        self.mode = mode
        self.gamma = gamma

        if scale_fn is None:
            if self.mode == 'triangular':
                self.scale_fn = self._triangular_scale_fn
                self.scale_mode = 'cycle'
            elif self.mode == 'triangular2':
                self.scale_fn = self._triangular2_scale_fn
                self.scale_mode = 'cycle'
            elif self.mode == 'exp_range':
                self.scale_fn = self._exp_range_scale_fn
                self.scale_mode = 'iterations'
        else:
            self.scale_fn = scale_fn
            self.scale_mode = scale_mode

        self.batch_step(last_batch_iteration + 1)
        self.last_batch_iteration = last_batch_iteration

    def batch_step(self, batch_iteration=None):
        if batch_iteration is None:
            batch_iteration = self.last_batch_iteration + 1
        self.last_batch_iteration = batch_iteration
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr

    def _triangular_scale_fn(self, x):
        return 1.

    def _triangular2_scale_fn(self, x):
        return 1 / (2. ** (x - 1))

    def _exp_range_scale_fn(self, x):
        return self.gamma**(x)

    def get_lr(self):
        step_size = float(self.step_size)
        cycle = np.floor(1 + self.last_batch_iteration / (2 * step_size))
        x = np.abs(self.last_batch_iteration / step_size - 2 * cycle + 1)

        lrs = []
        param_lrs = zip(self.optimizer.param_groups,
                        self.base_lrs, self.max_lrs)
        for param_group, base_lr, max_lr in param_lrs:
            base_height = (max_lr - base_lr) * np.maximum(0, (1 - x))
            if self.scale_mode == 'cycle':
                lr = base_lr + base_height * self.scale_fn(cycle)
            else:
                lr = base_lr + base_height * \
                    self.scale_fn(self.last_batch_iteration)
            lrs.append(lr)
        return lrs

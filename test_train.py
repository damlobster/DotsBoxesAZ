import numpy as np
import os
import torch
from torch import nn
from torch.nn import functional as F
import torch.utils.data as data
import pandas as pd

from game import GameState
from dots_boxes.dots_boxes_nn import SymmetriesGenerator
from nn import AlphaZeroLoss

def compute_accuracy(v, z):
    correct = ((((z*2).round()).sign()).eq(((v*2).round()).sign())).sum()
    return correct.item(), z.size()[0]

def train(model, optimizer, params, train_dataset, val_dataset, nb_epochs, epoch_i, verbose=10):
    tr_params = params.nn.train_params

    log = pd.DataFrame(columns=["epoch", "train/v", "train/pi", "train/v/acc", "eval/v", "eval/pi", "eval/v/acc"])

    drop_last = True
    train_data = data.DataLoader(
        train_dataset, tr_params.train_batch_size, shuffle=True, drop_last=drop_last)

    validation_data = data.DataLoader(
        val_dataset, tr_params.val_batch_size, shuffle=False, drop_last=drop_last) if val_dataset is not None else None

    criterion = AlphaZeroLoss()

    symGen = SymmetriesGenerator()
    
    print("LR=", tr_params.lr)

    for epoch in range(epoch_i, epoch_i+nb_epochs):
        acc_v = 0

        model.train(True)
        tr_loss_pi = 0
        tr_loss_v = 0
        tr_acc_correct = 0
        tr_acc_total = 0

        n_batches = 0
        for boards, pi, z in train_data:
            n_batches += 1
                
            # Transfer to GPU
            boards = boards.requires_grad_(True).to(params.nn.pytorch_device)
            pi = pi.to(params.nn.pytorch_device)
            z = z.to(params.nn.pytorch_device)

            boards, pi = symGen(boards, pi)

            p, v = model(boards)
            loss, (loss_pi, loss_v) = criterion(p, v, pi, z)
            tr_loss_v += loss_v
            tr_loss_pi += loss_pi
            correct, total = compute_accuracy(v, z)
            tr_acc_correct += correct
            tr_acc_total += total

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        tr_loss_pi = tr_loss_pi / n_batches
        tr_loss_v = tr_loss_v / n_batches
        tr_loss = tr_loss_pi + tr_loss_v


        val_loss_pi = 0
        val_loss_v = 0
        val_acc_correct = 0
        val_acc_total = 0
        if val_dataset:
            model.train(False)

            val_n_batches = 0
            for boards, pi, z in validation_data:
                # Transfer to GPU
                boards = boards.to(params.nn.pytorch_device)
                pi = pi.to(params.nn.pytorch_device)
                z = z.to(params.nn.pytorch_device)

                boards, pi = symGen(boards, pi)

                val_n_batches += 1
                p, v = model(boards)

                correct, total = compute_accuracy(v, z)
                val_acc_correct += correct
                val_acc_total += total
                _, (loss_pi, loss_v) = criterion(p, v, pi, z)
                val_loss_v += loss_v
                val_loss_pi += loss_pi

            val_loss_v /= val_n_batches
            val_loss_pi /= val_n_batches
            val_loss = val_loss_v + val_loss_pi

        log = log.append({
            "epoch":epoch, 
            "train/v": tr_loss_v, 
            "train/pi": tr_loss_pi, 
            "train/v/acc": tr_acc_correct/tr_acc_total, 
            "eval/v": val_loss_v, 
            "eval/pi": val_loss_pi, 
            "eval/v/acc": val_acc_correct/val_acc_total
            }, ignore_index=True)
        if epoch % verbose == 0:
            print(f"Epoch {epoch}, train [loss= {tr_loss:5f}, p={tr_loss_pi:5f}, v={tr_loss_v:5f}, v_acc={tr_acc_correct/tr_acc_total:5f}], validation [loss= {val_loss:5f}, p={val_loss_pi:5f}, v={val_loss_v:5f}, v_acc={val_acc_correct/val_acc_total:5f}]", flush=True)

    log.set_index("epoch", inplace=True)
    return log
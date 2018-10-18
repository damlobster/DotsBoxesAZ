import logging
logging.basicConfig(level=logging.INFO)

import argparse

from functools import partial
import asyncio
import sys
import numpy as np
import pickle
import multiprocessing as mp
import torch

from self_play import SelfPlay, generate_games
from nn import NeuralNetWrapper, ResNetZero
from dots_boxes.dots_boxes_game import BoxesState, moves_to_string, nn_batch_builder
from dots_boxes.dots_boxes_nn import SimpleNN
import utils.utils as utils
from utils.proxies import AsyncBatchedProxy

import configuration
params = configuration.params

def selfplay(generation):
    print("*"*50)
    print("Selfplay start for generation {} ...".format(generation))
    model = params.nn.model_class()
    model.load_parameters(generation)
    nn = NeuralNetWrapper(model, params)
    generate_games(params.hdf_file, generation, nn, params.num_games, params, n_workers=None, games_per_workers=20)


def train_nn(generation, where):
    print("*"*50)
    print("Train neural net for generation {} ...".format(generation))
    ds = utils.HDFStoreDataset(params.hdf_file, features_shape=params.game.clazz.FEATURES_SHAPE, where=where)
    model = params.nn.model_class()
    if generation != 1:
        model.load_parameters(generation-1)
    wrapper = NeuralNetWrapper(model, params)
    wrapper.train(ds)
    wrapper.save_model_parameters(generation)

def run_generation(generation):
    selfplay(generation)
    train_nn(generation+1)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("generation", type=int, help="the current generation")
    parser.add_argument("selfplay_data", type=str, help="path to the selfplay data HDFStore file")
    parser.add_argument("-m", "--model", type=str, default="./data/model_chkpts/model_{}.pt" help="path to the model to use")
    parser.add_argument("-t", "--train", action="store_true", help="train the neural network")
    parser.add_argument("-s", "--selfplay", action="store_true", help="generate selfplay")
    parser.add_argument("-i", "--iterations", type=int, default=1, help="number of iterations")
    parser.parse_args()


if __name__ == '__main__':
    main()
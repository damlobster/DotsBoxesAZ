import logging
logging.basicConfig(level=logging.INFO)

import argparse
import os
import torch.multiprocessing as mp
from functools import partial

import configuration
params = configuration.params

if not os.path.exists(params.data_root):
    os.makedirs(params.data_root)

def _launch_in_process(func, *args):
    with mp.Pool(processes = 1) as pool:
        res = pool.starmap(func, [args])
        return res[0]


def selfplay(generation):
    import configuration
    import time
    from self_play import generate_games
    params = configuration.params

    tick = time.time()
    print("*"*70)
    print("Selfplay start for generation {} ...".format(generation))
    print("Model used:", params.nn.model_class)
    generate_games(params.hdf_file, generation, params.nn.model_class, params.self_play.num_games, params, n_workers=params.self_play.n_workers, games_per_workers=10)
    print("Selfplay finished !!! Generation of {} games took {} sec.".format(params.self_play.num_games, time.time()-tick))


def train_nn(generation, where, last_batch_idx):
    import numpy as np
    import pandas as pd
    import time
    from nn import NeuralNetWrapper
    import utils.utils as utils

    tick = time.time()
    print("-"*70)
    print("Train neural net for generation {} (where={})".format(generation, where))

    # copy new training data to current dataframe (key=data)
    with pd.HDFStore(params.hdf_file, mode="a") as store:
        if '/fresh' in store:
            new_samples = store['/fresh']
            train = new_samples.sample(frac=params.nn.train_params.train_split)
            train = train.assign(training=1)
            val = new_samples[~new_samples.index.isin(train.index)]
            val = val.assign(training = -1)
            new_samples = train.append(val)
            new_samples = new_samples.astype({'training':np.int8})
            store.append('/data', new_samples, format='table')
            del store['/fresh']
        else:
            print("No new training data! Is it normal?")

    train_ds = utils.HDFStoreDataset(params.hdf_file, "data", train=True, features_shape=params.game.clazz.FEATURES_SHAPE, where=where)
    val_ds = utils.HDFStoreDataset(params.hdf_file, "data", train=False, features_shape=params.game.clazz.FEATURES_SHAPE, where=where)
    model = params.nn.model_class(params)
    if generation != 0:
        model.load_parameters(generation-1)
    wrapper = NeuralNetWrapper(model, params)

    last_batch_idx = wrapper.train(train_ds, val_ds, last_batch_idx)
    model.save_parameters(generation)

    print("Training finished in {} sec.".format(time.time()-tick))
    return last_batch_idx


def learn_to_play(from_generation, to_generation, last_batch_idx, start_train=False):
    while from_generation <= to_generation:
        if not start_train:
            selfplay(from_generation)
        else:
            start_train = False

        window_start = 0 if from_generation <= 3 else (from_generation - 3)//2
        where = "generation>={}".format(window_start)
        last_batch_idx = _launch_in_process(train_nn, from_generation, where, last_batch_idx)
        print("Last training batch idx= {}".format(last_batch_idx), flush=True)
        from_generation += 1


if __name__ == '__main__':
    mp.set_start_method('spawn')
    
    parser = argparse.ArgumentParser(description='Launch selfplay/train loop')
    parser.add_argument('from_gen', type=int, help='start from generation nb')
    parser.add_argument('to_gen', type=int, help='stop at from generation nb')
    parser.add_argument('-b', '--batch_idx', dest='batch_idx', type=int, default=0, help='for tensorboard: batch idx')
    parser.add_argument('-t', '--start_train', dest='start_train', action="store_true", default=False, help='if set, start with training')
    args = parser.parse_args()

    learn_to_play(args.from_gen, args.to_gen, args.batch_idx, args.start_train)
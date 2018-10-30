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
    generate_games(params.hdf_file, generation, params.nn.model_class, 
                   params.self_play.num_games, params, 
                   n_workers=params.self_play.n_workers, games_per_workers=10)
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


def compute_elo(params, from_generation, to_generation, last_elo):
    import time
    from self_play import compute_elo

    tick = time.time()
    print("*"*70)
    print("Elo rating start for generations {} vs {} ...".format(from_generation, to_generation))
    print("Model used:", params.nn.model_class)
    compute_elo(params.elo.hdf_file, [params, params], [from_generation, to_generation], (last_elo, last_elo), params.elo.n_games, 
                n_workers=params.self_play.n_workers, games_per_workers=params.elo.games_per_workers)
    print("Elo rating finished !!! Generation of {} games took {} sec.".format(params.elo.n_games, time.time()-tick))

def learn_to_play(from_generation, to_generation, last_batch_idx, last_model_elo=1200, start_train=False):
    while from_generation <= to_generation:
        if not start_train:
            selfplay(from_generation)
        start_train = False

        window_start = 0 if from_generation <= 3 else (from_generation - 3)//2
        where = "generation>={}".format(window_start)
        last_batch_idx = _launch_in_process(train_nn, from_generation, where, last_batch_idx)
        print("Last training batch idx= {}".format(last_batch_idx), flush=True)

        if from_generation > 0:
            compute_elo(params.elo.hdf_file, params, from_generation, (last_model_elo, last_model_elo), 
                n_games=params.elo.n_games, n_workers=params.elo.n_workers)

        from_generation += 1


if __name__ == '__main__':
    mp.set_start_method('spawn')
    
    parser = argparse.ArgumentParser(description='Launch selfplay/train/elo loop')
    parser.add_argument('from_gen', type=int, help='start from generation nb')
    parser.add_argument('to_gen', type=int, help='stop at from generation nb')
    parser.add_argument('-b', '--batch_idx', dest='batch_idx', type=int, default=None, help='for tensorboard: batch idx')
    parser.add_argument('-t', '--start_train', dest='start_train', action="store_true", default=False, help='if set, start with training')
    parser.add_argument('-e', '--elo', dest='elo', type=int, default=None, help='Elo of player from_generation')
    parser.add_argument('-c', '--call', dest='call', default=None, help='run just on function')
    args = parser.parse_args()

    if args.call:
        if args.call == "elo":
            assert args.elo is not None
            compute_elo(params, args.from_gen, args.to_gen, args.elo)
        elif args.call == "selfplay":
            selfplay(args.from_gen)
        elif args.call == "train":
            assert args.batch_idx is not None
            last_batch_idx = _launch_in_process(train_nn, to_generation, "generation>={}".format(from_generation), args.batch_idx)
            print("Last training batch idx= {}".format(last_batch_idx), flush=True)
        else:
            raise ValueError("Command undefined:", args.call)
    else:
        learn_to_play(args.from_gen, args.to_gen, args.batch_idx, args.elo, args.start_train)
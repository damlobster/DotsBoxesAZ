import argparse
import os
import torch
import torch.multiprocessing as mp
from tensorboardX import SummaryWriter
from functools import partial
import configuration
params = configuration.params

if not os.path.exists(params.data_root):
    os.makedirs(params.data_root)

def _launch_in_process(func, *args):
    with mp.Pool(processes = 1) as pool:
        res = pool.starmap(func, [args])
        return res[0]

def selfplay(params, generation):
    import time
    from self_play import generate_games

    tick = time.time()
    print("*"*70)
    print(f"Selfplay start for generation {generation}...")
    print("Model used:", params.nn.model_class, flush=True)
    generate_games(params.hdf_file, generation, params.nn.model_class, 
                   params.self_play.num_games, params, 
                   n_workers=params.self_play.n_workers, games_per_workers=params.self_play.games_per_workers)
    print(f"Selfplay finished. Generation of {params.self_play.num_games} games took {time.time()-tick:.0f} sec.", flush=True)


def train_nn(params, generation, where, writer, last_batch_idx):
    import numpy as np
    import pandas as pd
    import time
    from nn import NeuralNetWrapper
    import utils.utils as utils

    tick = time.time()
    print("-"*70)
    print(f"Train neural net for generation {generation} (where:{where})...", flush=True)

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
            print("No new training data! Is it normal?", flush=True)

    train_ds = utils.HDFStoreDataset(params.hdf_file, "data", train=True, features_shape=params.game.clazz.FEATURES_SHAPE, where=where)
    val_ds = utils.HDFStoreDataset(params.hdf_file, "data", train=False, features_shape=params.game.clazz.FEATURES_SHAPE, where=where)
    model = params.nn.model_class(params)
    if generation != 0:
        model.load_parameters(generation-1)
    wrapper = NeuralNetWrapper(model, params)

    if params.nn.lr_scheduler is not None:
        params.nn.lr = params.nn.lr_scheduler(generation)
    last_batch_idx = wrapper.train(train_ds, val_ds, writer, last_batch_idx)
    model.save_parameters(generation)
    # free cuda cache so that this memory can be used by spawned selfplay processes
    del model, wrapper, train_ds, val_ds
    torch.cuda.empty_cache() 

    print(f"Training finished in {time.time()-tick:.0f} sec. (batch_idx={last_batch_idx})", flush=True)
    return last_batch_idx


def compute_elo(elo_params, player0, player1):
    import time
    from self_play import compute_elo as elo

    params0, gen0, elo0 = player0
    params1, gen1, elo1 = player1

    tick = time.time()
    print("-"*70)
    print("Computation of Elo ratings...")
    elo0, elo1 = elo(elo_params, [params0, params1], [gen0, gen1], (elo0, elo1))
    print(f"Elo ratings computation finished in {time.time()-tick:.0f} sec.!", flush=True)
    return elo0, elo1

def learn_to_play(params, from_generation, to_generation, last_batch_idx, last_model_elo=1200, start_train=False):
    while from_generation <= to_generation:
        if not start_train:
            selfplay(params, from_generation)
        start_train = False

        writer = SummaryWriter(params.tensorboard_log)
        where = f"generation>={max(0, min((from_generation-4)//2, 20))}"
        params.nn.train_params.lr = params.nn.train_params.lr_scheduler(from_generation)
        last_batch_idx = train_nn(params, from_generation, where, writer, last_batch_idx)
        print(f"Last training batch idx= {last_batch_idx}", flush=True)

        if from_generation > 0:
            player0 = (params, from_generation-1, last_model_elo)
            player1 = (params, from_generation, last_model_elo)
            _, last_model_elo = compute_elo(params.elo, player0, player1)
            writer.add_scalar('elo', last_model_elo, last_batch_idx)

        writer.close()
        from_generation += 1

def main(parser):
    global params
    if args.params:
        params = eval(args.params)
    else:
        params = configuration.params

    if args.call:
        eval(args.call)
    else:
        learn_to_play(params, args.from_gen, args.to_gen, args.batch_idx, args.elo, args.start_train)

if __name__ == '__main__':
    mp.set_start_method('spawn')
    
    parser = argparse.ArgumentParser(description='Launch selfplay/train/elo loop')
    parser.add_argument('from_gen', type=int, help='start from generation nb')
    parser.add_argument('to_gen', type=int, help='stop at from generation nb')
    parser.add_argument('-b', '--batch_idx', dest='batch_idx', type=int, default=0, help='for tensorboard: batch idx')
    parser.add_argument('-t', '--start_train', dest='start_train', action="store_true", default=False, help='if set, start with training')
    parser.add_argument('-e', '--elo', dest='elo', type=int, default=1200, help='Elo of last generation player')
    parser.add_argument('-c', '--call', dest='call', default=None, help='call a function')
    parser.add_argument('-p', '--params', dest='params', default='configuration.params', help='parameters dotdict to use, must be a global in configuration.py file')
    args = parser.parse_args()

    main(args)
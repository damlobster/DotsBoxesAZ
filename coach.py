import argparse
import json
import os
import torch
import torch.multiprocessing as mp
from tensorboardX import SummaryWriter
from functools import partial
import configuration
params = None

def selfplay(params, generation):
    """Run a batch of selfplay games.
    All the parameters are specified in the params argument (see configuration.py for details).
    
    Arguments:
        params {DotDict} -- dictionnary holding the parameters of the selfplay, MCTS, neural net
        generation {int} -- the generation for which the games will be generated
    """

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


def train_nn(params, generation, where, writer):
    """Train the neural net with the selfplay generated game positions
    
    Arguments:
        params {DotDict} -- the parameters
        generation {int} -- the generation for which we train the nn
        where {str} -- a HDFStore compatible query, eg. 'generation>=4 and generation<=10'
        writer {tensorboardx.SummaryWriter} -- where to write the training curves
    
    Returns:
        [int] -- the last training batch index
    """

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

    avg = params.nn.train_params.pos_average
    train_ds = utils.HDFStoreDataset(params.hdf_file, "data", train=True, features_shape=params.game.clazz.FEATURES_SHAPE, where=where, pos_average=avg)
    val_ds = utils.HDFStoreDataset(params.hdf_file, "data", train=False, features_shape=params.game.clazz.FEATURES_SHAPE, where=where, pos_average=avg)
    model = params.nn.model_class(params)
    wrapper = NeuralNetWrapper(model, params)

    if params.nn.lr_scheduler is not None:
        params.nn.lr = params.nn.lr_scheduler(generation)
    last_batch_idx = wrapper.train(train_ds, val_ds, writer, generation)

    # free cuda cache so that this memory can be used by spawned selfplay processes
    del model, wrapper, train_ds, val_ds
    torch.cuda.empty_cache() 

    print(f"Training finished in {time.time()-tick:.0f} sec. (batch_idx={last_batch_idx})", flush=True)
    return last_batch_idx


def compute_elo(elo_params, player0, player1):
    """Evaluate the relative performance of two players by making them player against each other.
    
    Arguments:
        elo_params {DotDict} -- the paramters of the selfplay for computing Elo score
        player0 {(DotDict, int, int)} -- a tuple containing: the parameters of the 1st player, its the generation and its elo old score
        player1 {(DotDict, int, int)} -- same as 1st player
    
    Returns:
        [(int, int)] -- the new Elo scores of the players
    """

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

def learn_to_play(params, from_generation, to_generation, last_model_elo=1200, start_train=False):
    """Reinforcement learing loop.
    1. Selfplay with last generation network
    2. Train the network with the data generated so far
    3. Compute Elo score
    4. Increment generation and go to 1.
    
    Arguments:
        params {DotDict} -- the parameters to use
        from_generation {int} -- starting generation
        to_generation {int} -- stop at this generation (included)
    
    Keyword Arguments:
        last_model_elo {int} -- the Elo score of the last model (new model start with this value) (default: {1200})
        start_train {bool} -- if true, skip the selfplay for the first generation (default: {False})
    """

    writer = SummaryWriter(params.tensorboard_log)
    cfg = json.dumps(params, indent=4, default=lambda o: str(o)).replace(" ", "&nbsp;").replace("\n", "  \n")
    writer.add_text("params", cfg)
    while from_generation <= to_generation:
        if not start_train:
            selfplay(params, from_generation)
        start_train = False

        where = f"generation>={max(0, min((from_generation-4)//2, 20))}"
        params.nn.train_params.lr = params.nn.train_params.lr_scheduler(from_generation)
        last_batch_idx = train_nn(params, from_generation, where, writer)
        print(f"Last training batch idx= {last_batch_idx}", flush=True)

        if from_generation > 0:
            player0 = (params, from_generation-1, last_model_elo)
            player1 = (params, from_generation, last_model_elo)
            _, last_model_elo = compute_elo(params.elo, player0, player1)
            writer.add_scalar('elo', last_model_elo, last_batch_idx)

        from_generation += 1
    writer.close()


def main(args):
    global params
    if args.params:
        params = eval(args.params)
    else:
        params = configuration.params

    params.rewrite_str("_exp_", args.exp)
    if not os.path.exists(params.data_root):
        os.makedirs(params.data_root)

    if args.call:
        eval(args.call)
    else:
        learn_to_play(params, args.from_gen, args.to_gen, args.elo, args.start_train)

if __name__ == '__main__':
    mp.set_start_method('spawn')
    
    parser = argparse.ArgumentParser(description='Launch selfplay/train/elo loop')
    parser.add_argument('from_gen', type=int, help='start from generation nb')
    parser.add_argument('to_gen', type=int, help='stop at from generation nb')
    parser.add_argument('exp', type=str, help='experiment tag')
    parser.add_argument('-t', '--start_train', dest='start_train', action="store_true", default=False, help='if set, start with training')
    parser.add_argument('-e', '--elo', dest='elo', type=int, default=1200, help='Elo of last generation player')
    parser.add_argument('-c', '--call', dest='call', default=None, help='call a function')
    parser.add_argument('-p', '--params', dest='params', default='configuration.params', help='parameters dotdict to use, must be a global in configuration.py file')
    args = parser.parse_args()

    main(args)
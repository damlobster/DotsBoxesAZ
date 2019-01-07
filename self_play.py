import asyncio
import sys
import copy
import random
from functools import partial
import torch.multiprocessing as mp
import configuration
import logging
logger = logging.getLogger(__name__)

import numpy as np
import pandas as pd
import math

from utils.utils import DotDict, elo_rating2
import mcts


class SelfPlay(object):

    def __init__(self, nn, params):
        self.played_games = []
        self.params = params
        self.nn = nn
        self.player_change_callback = lambda player: None

    async def get_next_move(self, root_node, nb_mcts_searches, temperature, dirichlet):
        visit_counts = await mcts.UCT_search(root_node, nb_mcts_searches, self.nn, 
                                             self.params.self_play.mcts.mcts_cpuct,
                                             self.params.self_play.mcts.max_async_searches, dirichlet)
        # apply temperature
        probs = (visit_counts/visit_counts.max()) ** (1/temperature)
        probs = probs / probs.sum()
        # sample next move
        move = np.random.choice(probs.shape[0], 1, p=probs)[0]

        # debug 
        if(logger.isEnabledFor(logging.DEBUG)):
            msg = []
            for i in range(visit_counts.shape[0]):
                if round(probs[i], 3) > 0:
                    o = '\033[92m' if i==move else "\033[93m" if probs[move]<probs[i] else "\033[94m" if probs[i]>0.1 else ""
                    c = '\033[0m' if i==move or probs[i]>0 else ""
                    msg.append(f"{o}{visit_counts[i]}:{probs[i]:.3f}{c}")

            logger.debug(" ".join(msg))
            logger.debug("move= %s", move)
        # end debug
        return move

    async def play_game(self, game_state, idx):
        temperature = None

        moves_sequence = []
        root_node = mcts.create_root_uct_node(game_state)
        i = -1
        while not root_node.is_terminal:
            i += 1
            self.player_change_callback(root_node.game_state.to_play)
            params = self.params
            if i in params.self_play.mcts.temperature:
                temperature = params.self_play.mcts.temperature[i]

            nb_valid_moves = len(root_node.game_state.get_valid_moves(as_indices=True))
            n_searches = min(4*math.factorial(nb_valid_moves), params.self_play.mcts.mcts_num_read)
            move = await self.get_next_move(root_node, n_searches, temperature, params.self_play.noise)
            
            moves_sequence.append(root_node)
            root_node = mcts.init_mcts_tree(root_node, move, reuse_tree=params.self_play.reuse_mcts_tree)

        moves_sequence.append(root_node) #add the terminal node

        self.played_games.append(
            (idx, moves_sequence, root_node.game_state.get_result()))

    async def play_games(self, game_state, games_idxs, show_progress=False):
        for idx in games_idxs:
            if show_progress:
                print(".", end="", flush=True)
            await self.play_game(game_state, idx)

    def get_games_moves(self):
        moves = []
        visit_counts = []
        for _, moves_seq, _ in self.played_games:
            for node in moves_seq[1:]:
                moves.append(node.move)
                visit_counts.append(node.child_number_visits)
        vc = np.asarray(visit_counts, dtype=float)
        return moves, vc

    def set_player_change_callback(self, cb):
        self.player_change_callback = cb

    def get_datasets(self, generation, with_features=True):
        game_idxs = []
        moves_idxs = []
        moves = []
        players = []
        features = []
        policies = []
        values = []
        stats = []
        for game_idx, moves_seq, z in self.played_games:
            terminal = moves_seq[-1]
            winner = terminal.game_state.just_played
            for move_i, node in enumerate(moves_seq[:-1]):
                game_idxs.append(game_idx)
                moves_idxs.append(move_i)
                moves.append(node.move)
                players.append(node.game_state.to_play)
                values.append(z if node.game_state.to_play == winner else -z)
                stats.append(node.get_tree_stats())
                vs = node.child_number_visits.sum()
                policies.append(node.child_number_visits / (vs or 1.0))
                if with_features:
                    features.append(node.game_state.get_features().ravel())

        players = np.asarray(players, dtype=np.int8)
        if isinstance(generation, (list, tuple)):
            gen = np.zeros(len(moves_idxs), dtype=np.int16)
            gen[players==0] = generation[0]
            gen[players==1] = generation[1]
            generation = gen
        else:
            generation = np.asarray([generation]*len(moves_idxs), dtype=np.int16)

        df = pd.DataFrame(generation, columns=['generation'])
        games_idxs = np.asarray(game_idxs, dtype=np.int16)
        df = df.join(pd.DataFrame(games_idxs, columns=['game_idx'], index=df.index))
        moves_idxs = np.asarray(moves_idxs, dtype=np.int16)
        df = df.join(pd.DataFrame(moves_idxs, columns=['move_idx'], index=df.index))

        moves = np.asarray(moves)
        moves[moves==None] = -1
        moves = moves.astype(np.int16)
        df = df.join(pd.DataFrame(moves, columns=['move'], index=df.index))

        df = df.join(pd.DataFrame(players, columns=['player'], index=df.index))

        if with_features:
            features = np.stack(features, axis=0)
            df = df.join(pd.DataFrame(features, columns=list("x_"+str(i) for i in range(features.shape[1])), index=df.index))

        policies = np.stack(policies, axis=0)
        df = df.join(pd.DataFrame(policies, columns=list("pi_"+str(i) for i in range(policies.shape[1])), index=df.index))

        values = np.asarray(values)[:, np.newaxis]
        df = df.join(pd.DataFrame(values, columns=['z'], index=df.index))

        stats_df = pd.DataFrame.from_records(stats, columns=['max_deepness', 'tree_size', 'terminal_count', 'q_value'], index=df.index)
        df = df.join(stats_df.astype({'max_deepness':np.int16, 'tree_size':np.int32, 'terminal_count':np.int32, 'q_value':np.float32}))

        df.set_index(["generation", "game_idx", "move_idx"], inplace=True)

        return df


def _fut_cb(fut):
    if fut.exception():
        print(fut.exception(), flush=True)
        raise fut.exception()


_env_ = None
def _worker_init(hdf_file_name, devices, lock, generations, nn_class, params, player_change_callback=None):
    global _env_
    import multiprocessing as mp    
    import time
    import self_play
    from nn import NeuralNetWrapper
    from utils.proxies import AsyncBatchedProxy

    pid = mp.current_process().pid

    _env_ = DotDict({})

    if not isinstance(nn_class, (list,tuple)):
        nn_class, generations, params = (nn_class,), (generations,), (params,)
    else:
        _env_.compare_models = True
    assert len(nn_class) == len(generations) == len(params)

    pytorch_device = devices[pid%len(devices)]
    players_params = {}
    models = {}
    for i in range(len(generations)):
        models[i] = nn_class[i](params[i])
        if generations[i] != 0:
            models[i].load_parameters(generations[i]-(1 if len(nn_class)==1 else 0), to_device=pytorch_device)
        players_params[i] = params[i]

    if len(models) == 1:
        models[1] = models[0]
        players_params[1] = players_params[0]
        generations = [generations[0], generations[0]]

    players_params[0].nn.pytorch_device = pytorch_device
    players_params[1].nn.pytorch_device = pytorch_device

    # shuffle the players based on the pid, important for computing the Elo score
    if pid % 2 != 0:
        tmp = models[0]
        models[0] = models[1]
        models[1] = tmp
        tmp = players_params[0]
        players_params[0] = players_params[1]
        players_params[1] = players_params[0]
        generations = list(reversed(generations))

    _env_.models = models
    _env_.players_params = DotDict(players_params)
    _env_.generations = generations
    _env_.params = players_params[0]
    _env_.player_change_callback = player_change_callback
    _env_.name = 'w%i' % pid
    _env_.hdf_file_name = hdf_file_name
    _env_.hdf_lock = lock

    logger.info("Worker %s uses device %s", _env_.name, _env_.params.nn.pytorch_device)

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)            
    
    sp_params = _env_.params.self_play
    _env_.nn_wrapper = NeuralNetWrapper(None, _env_.params)
    _env_.nnet = AsyncBatchedProxy(_env_.nn_wrapper, batch_size=sp_params.nn_batch_size, 
                                   timeout=sp_params.nn_batch_timeout, 
                                   batch_builder=sp_params.nn_batch_builder,
                                   cache_size = 0 if _env_.compare_models else 400000)
    _env_.tasks = []
    fut = asyncio.ensure_future(_env_.nnet.run(), loop=loop)
    fut.add_done_callback(_fut_cb) # re-raise exception if occured in nnet
    _env_.tasks.append(fut)


def _player_change_callback(player):
    _env_.sp.params = _env_.players_params[player]
    _env_.nn_wrapper.set_model(_env_.models[player])


def _worker_run(games_idxs):
    global _env_
    import self_play
    from dots_boxes.dots_boxes_game import BoxesState
    from utils.utils import write_to_hdf
    import time
    loop = asyncio.get_event_loop()

    tick = time.time()
    try:
        _env_.sp = self_play.SelfPlay(_env_.nnet, _env_.params)
        _env_.sp.set_player_change_callback(_player_change_callback)
        loop.run_until_complete(_env_.sp.play_games(BoxesState(), games_idxs, show_progress=False))
    except Exception as e:
        print(e, flush=True)
        raise e
    tack = time.time()

    df = _env_.sp.get_datasets(_env_.generations, not _env_.compare_models)
    if not _env_.compare_models:
        df["training"] = np.zeros(len(df.index), dtype=np.int8)

    with _env_.hdf_lock:
        write_to_hdf(_env_.hdf_file_name, "fresh", df)

    tock = time.time()

    logger.warning("Worker %s played %d games (%d samples) in %.0fs (save=%.3fs)", 
        _env_.name, len(games_idxs), len(df.index), tock-tick, tock-tack)


def _worker_teardown(not_used):
    import time
    loop = asyncio.get_event_loop()
    try:
        for  t in _env_.tasks:
            t.cancel()
        loop.run_until_complete(asyncio.sleep(0.01))
        asyncio.get_event_loop().close()
    except Exception as e:
        logger.exception("An error occured during workers teardown.")
        raise e
    time.sleep(0.1)

def _err_cb(err):
    if isinstance(err, BaseException):
        logger.error("Got an exception from a worker: %s", err)
        raise err

def generate_games(hdf_file_name, generation, nn_class, n_games, params, n_workers=None, games_per_workers=10):
    nw = n_workers if n_workers else mp.cpu_count() - 1
    gpw = games_per_workers if games_per_workers else max(1, n_games//nw)
    games_idxs = np.array_split(np.arange(n_games), n_games//gpw)
    nw = min(nw, len(games_idxs))

    lock = mp.Lock()
    devices = params.self_play.pytorch_devices
    with mp.Pool(nw, initializer=_worker_init, initargs=(hdf_file_name, devices, lock, generation, nn_class, params)) as pool:
        try:    
            tasks = [pool.apply_async(_worker_run, (idxs,), error_callback=_err_cb) for idxs in games_idxs]
            [t.wait() for t in tasks]
        except Exception as e:
            logger.exception("An error occured")
            raise e
        pool.map(_worker_teardown, range(nw))


def compute_elo(elo_params, params, generations, elos):
    nw = elo_params.n_workers if elo_params.n_workers else mp.cpu_count() - 1
    gpw = elo_params.games_per_workers if elo_params.games_per_workers else max(1, elo_params.n_games//nw)
    games_idxs = np.array_split(np.arange(elo_params.n_games), elo_params.n_games//gpw)
    nw = min(nw, len(games_idxs))

    lock = mp.Lock()
    params = copy.deepcopy(params)
    params[0].self_play.merge(elo_params.self_play_override)
    params[1].self_play.merge(elo_params.self_play_override)
    devices = params[0].self_play.pytorch_devices
    nn_classes = list(p.nn.model_class for p in params)
    with mp.Pool(nw, initializer=_worker_init, initargs=(elo_params.hdf_file, devices, lock, generations, nn_classes, params)) as pool:
        try:
            tasks = [pool.apply_async(_worker_run, (idxs,), error_callback=_err_cb) for idxs in games_idxs]
            [t.wait() for t in tasks]
        except Exception as e:
            logger.exception("An error occured")
            raise e
        pool.map(_worker_teardown, range(nw))

    # compute new elo score
    with pd.HDFStore(elo_params.hdf_file, mode="a") as store:
        games = store.get("/fresh")
        del store["/fresh"]
        store.put("elo{}vs{}".format(*generations), games, format="table", append=False)

    winners = games[games.z==1].sort_index(level=["move_idx"]).groupby(level=["game_idx"]).head(1)
    n0 = sum(winners.index.get_level_values("generation")==generations[0])
    n1 = len(winners)-n0
    elo0, elo1 = elo_rating2(elos[0], elos[1], n0, n1, K=30)

    print(f"{params[0].nn.model_class.__name__} generation {generations[0]}: wins={n0}, elo={elos[0]} -> {elo0}")
    print(f"{params[1].nn.model_class.__name__} generation {generations[1]}: wins={n1}, elo={elos[1]} -> {elo1}")
    
    return elo0, elo1, n1/len(winners)

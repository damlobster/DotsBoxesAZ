import asyncio
import logging
import sys
import random
from functools import partial
import torch.multiprocessing as mp
mpl = mp.log_to_stderr()
mpl.setLevel(logging.WARNING)

import numpy as np
import pandas as pd

from utils.utils import DotDict, elo_rating2
from utils.proxies import AsyncBatchedProxy
import mcts


class SelfPlay(object):

    def __init__(self, nn, params):
        self.played_games = []
        self.params = params
        self.nn = nn
        self.player_change_callback = lambda sp_self, player: None

    async def get_next_move(self, root_node, nb_mcts_searches, temperature, dirichlet):
        visit_counts = await mcts.UCT_search(root_node, nb_mcts_searches, self.nn, 
                                             self.params.self_play.mcts.mcts_cpuct,
                                             self.params.self_play.mcts.max_async_searches, dirichlet)
        
        # apply temperature
        probs = (visit_counts/visit_counts.max()) ** (1/temperature)
        probs = probs / (1.000001*probs.sum()) # TODO: p/p.sum() is sometimes > 1
        sampled = np.random.multinomial(1, np.append(probs, 0), 1)
        move = np.argmax(sampled)
        return move

    async def play_game(self, game_state, idx):
        temperature = None

        moves_sequence = []
        root_node = mcts.create_root_uct_node(game_state)
        i = -1
        while not root_node.is_terminal:
            i += 1
            self.player_change_callback(root_node.game_state.next_player)
            params = self.params
            if i in params.self_play.mcts.temperature:
                temperature = params.self_play.mcts.temperature[i]

            move = await self.get_next_move(root_node, params.self_play.mcts.mcts_num_read, 
                                            temperature, params.self_play.noise)
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
            for move_i, node in reversed(list(enumerate(moves_seq[:-1]))): #! remove last move in training data
                if node.game_state.player != node.game_state.next_player:
                    z = -z # flip sign of z if player has changed
                game_idxs.append(game_idx)
                moves_idxs.append(move_i)
                moves.append(node.move)
                players.append(node.game_state.player)
                values.append(z)
                stats.append(node.get_tree_stats())
                vs = node.child_number_visits.sum()
                policies.append(node.child_number_visits / (vs if vs > 0.0 else 1.0))
                if with_features:
                    features.append(node.game_state.get_features().ravel())

        players = np.asarray(players, dtype=np.int8)        
        if isinstance(generation, (list, tuple)):
            gen = np.zeros(len(game_idxs), dtype=np.int16)
            gen[players==0] = generation[0]
            gen[players==1] = generation[1]
            generation = gen
        else:
            generation = np.asarray([generation]*len(game_idxs), dtype=np.int16)

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


_env_ = None
def _worker_init(hdf_file_name, devices, lock, generations, nn_class, params, player_change_callback=None):
    global _env_
    import time
    import self_play
    from nn import NeuralNetWrapper
    import multiprocessing as mp
    pid = mp.current_process().pid

    _env_ = DotDict({})

    if not isinstance(nn_class, (list,tuple)):
        nn_class, generations, params = (nn_class,), (generations,), (params,)
    else:
        _env_.one_generation = True
    assert len(nn_class) == len(generations) == len(params)

    players_params = {}
    models = {}
    for i in range(len(generations)):
        models[i] = nn_class[i](params[i])
        models[i].load_parameters(generations[i])
        players_params[i] = params[i]

    if len(models) == 1:
        models[1] = models[0]
        players_params[1] = players_params[0]
        generations = [generations[0], generations[0]]

    # shuffle the players based on the pid, important for computing the Elo score
    if pid % 2 != 0:
        tmp = models[0]
        models[0] = models[1]
        models[1] = tmp
        tmp = players_params[0]
        players_params[0] = players_params[1]
        players_params[1] = players_params[0]
        generations = list(reversed(generations))

    pytorch_device = devices[pid%len(devices)]
    players_params[0].nn.pytorch_device = pytorch_device
    players_params[1].nn.pytorch_device = pytorch_device

    _env_.models = models
    _env_.players_params = DotDict(players_params)
    _env_.generations = generations
    _env_.params = players_params[0]
    _env_.player_change_callback = player_change_callback
    _env_.name = 'w%i' % pid
    _env_.hdf_file_name = hdf_file_name
    _env_.hdf_lock = lock

    print("Worker {} uses device {}".format(_env_.name, _env_.params.nn.pytorch_device), flush=True)

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)            
    
    sp_params = _env_.params.self_play
    _env_.nn_wrapper = NeuralNetWrapper(None, _env_.params)
    _env_.nnet = AsyncBatchedProxy(_env_.nn_wrapper, batch_size=sp_params.nn_batch_size, 
                                   timeout=sp_params.nn_batch_timeout, 
                                   batch_builder=sp_params.nn_batch_builder)
    _env_.tasks = []
    _env_.tasks.append(asyncio.ensure_future(_env_.nnet.run(), loop=loop))


def _player_change_callback(player):
    _env_.sp.params = _env_.players_params[player]
    _env_.nn_wrapper.set_model(_env_.models[player])


def _worker_run(games_idxs):
    global _env_
    import self_play
    from dots_boxes.dots_boxes_game import BoxesState
    from utils.utils import write_to_hdf
    import time

    tick = time.time()

    loop = asyncio.get_event_loop()
    try:
        _env_.sp = self_play.SelfPlay(_env_.nnet, _env_.params)
        _env_.sp.set_player_change_callback(_player_change_callback)
        loop.run_until_complete(_env_.sp.play_games(BoxesState(), games_idxs, show_progress=False))
    except Exception as e:
        print(e, flush=True)
        raise e

    tack = time.time()

    df = _env_.sp.get_datasets(_env_.generations, not _env_.one_generation)
    if not _env_.one_generations:
        df["training"] = np.zeros(len(df.index), dtype=np.int8)

    with _env_.hdf_lock:
        write_to_hdf(_env_.hdf_file_name, "fresh", df)

    tock = time.time()

    print("Worker {} played {} games ({} samples) in {:.0f}s (save={:.3f}s)".format(
        _env_.name, len(games_idxs), len(df.index), tock-tick, tock-tack), flush=True)


def _worker_teardown(not_used):
    import time
    loop = asyncio.get_event_loop()
    for t in _env_.tasks:
        t.cancel()
    loop.run_until_complete(asyncio.sleep(0.01))
    asyncio.get_event_loop().close()
    time.sleep(0.1)

def _err_cb(err):
    print("Got an exception from a worker:", err, flush=True)
    raise err

def generate_games(hdf_file_name, generation, nn_class, n_games, params, n_workers=None, games_per_workers=10):
    nw = n_workers if n_workers else mp.cpu_count() - 1
    gpw = games_per_workers if games_per_workers else max(1, n_games//nw)
    games_idxs = np.array_split(np.arange(n_games), n_games//gpw)
    nw = min(nw, len(games_idxs))

    lock = mp.Lock()
    devices = params.self_play.pytorch_devices
    with mp.Pool(nw, initializer=_worker_init, initargs=(hdf_file_name, devices, lock, generation, nn_class, params)) as pool:
        tasks = [pool.apply_async(_worker_run, (idxs,), error_callback=_err_cb) for idxs in games_idxs]
        [t.wait() for t in tasks]
        pool.map(_worker_teardown, range(nw))


def compute_elo(hdf_file_name, params, generations, elos, n_games, n_workers=None, games_per_workers=10):
    nw = n_workers if n_workers else mp.cpu_count() - 1
    gpw = games_per_workers if games_per_workers else max(1, n_games//nw)
    games_idxs = np.array_split(np.arange(n_games), n_games//gpw)
    nw = min(nw, len(games_idxs))

    lock = mp.Lock()
    devices = params[0].self_play.pytorch_devices
    nn_classes = list(p.nn.model_class for p in params)
    with mp.Pool(nw, initializer=_worker_init, initargs=(hdf_file_name, devices, lock, generations, nn_classes, params)) as pool:
        tasks = [pool.apply_async(_worker_run, (idxs,), error_callback=_err_cb) for idxs in games_idxs]
        [t.wait() for t in tasks]
        pool.map(_worker_teardown, range(nw))

    # compute new elo score
    with pd.HDFStore(hdf_file_name, mode="a") as store:
        games = store.get("/fresh")
        del store["/fresh"]
        store.put("elo{}vs{}".format(*generations), games, format="table", append=False)

    winners = games[games.z==1].sort_index(level=["move_idx"]).groupby(level=["game_idx"]).head(1)
    n0 = sum(winners.index.get_level_values("generation")==generations[0])
    n1 = len(winners)-n0
    elo0, elo1 = elo_rating2(elos[0], elos[1], n0, n1, K=30)

    print("Generation {} won {} games and generation {} won {} games".format(generations[0], n0, generations[1], n1))
    print("Old Elo scores: gen{}= {:.1f}, gen{}= {:.1f}".format(generations[0], elos[0], generations[1], elos[1]), flush=True)
    print("New Elo scores: gen{}= {:.1f}, gen{}= {:.1f}".format(generations[0], elo0, generations[1], elo1), flush=True)

    return elo0, elo1

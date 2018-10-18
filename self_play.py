import asyncio
import logging
import sys
import random
from functools import partial
import multiprocessing as mp
#mpl = mp.log_to_stderr()
#mpl.setLevel(logging.INFO)

import numpy as np
import pandas as pd

import utils.utils as utils
from utils.proxies import PipedProxy, pipe_proxy_init, AsyncBatchedProxy
import mcts


class SelfPlay(object):

    def __init__(self, nn, params):
        self.played_games = []
        self.params = params.self_play
        self.nn = nn

    async def get_next_move(self, root_node, nb_mcts_searches, temperature, dirichlet):
        visit_counts = await mcts.UCT_search(root_node, nb_mcts_searches, self.nn, 
                                             self.params.mcts.mcts_cpuct,
                                             self.params.mcts.max_async_searches, dirichlet)
        
        # apply temperature
        probs = (visit_counts/visit_counts.max()) ** (1/temperature)
        probs = probs / (1.000001*probs.sum()) # TODO: p/p.sum() is sometimes > 1
        sampled = np.random.multinomial(1, np.append(probs, 0), 1)
        move = np.argmax(sampled)
        return move

    async def play_game(self, game_state, idx):
        params = self.params
        temperature = None

        moves_sequence = []
        root_node = mcts.create_root_uct_node(game_state)
        i = -1
        while not root_node.is_terminal:
            i += 1
            if i in params.mcts.temperature:
                temperature = params.mcts.temperature[i]
            move = await self.get_next_move(root_node, params.mcts.mcts_num_read, 
                                            temperature, params.noise)
            
            moves_sequence.append(root_node)
            root_node = mcts.init_mcts_tree(root_node, move, reuse_tree=params.reuse_mcts_tree)

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
        for moves_seq, _ in self.played_games:
            for node in moves_seq[1:]:
                moves.append(node.move)
                visit_counts.append(node.child_number_visits)
        vc = np.asarray(visit_counts, dtype=float)
        return moves, vc

    def get_datasets(self, generation):
        game_idxs = []
        moves_idxs = []
        moves = []
        players = []
        features = []
        policies = []
        values = []
        stats = []
        for game_idx, moves_seq, result in self.played_games:
            for move_i, node in reversed(list(enumerate(moves_seq[:-1]))):
                game_idxs.append(game_idx)
                moves_idxs.append(move_i)
                moves.append(node.move)
                players.append(node.game_state.player)
                features.append(node.game_state.get_features().ravel())
                policies.append(node.child_number_visits /
                                node.child_number_visits.sum())
                values.append(result)
                stats.append(node.get_tree_stats())
                if node.game_state.player != node.game_state.next_player:
                    result = result * -1
        
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

        players = np.asarray(players, dtype=np.int8)
        df = df.join(pd.DataFrame(players, columns=['player'], index=df.index))

        features = np.stack(features, axis=0)
        df = df.join(pd.DataFrame(features, columns=list("feature_"+str(i) for i in range(features.shape[1])), index=df.index))

        policies = np.stack(policies, axis=0)
        df = df.join(pd.DataFrame(policies, columns=list("policy_"+str(i) for i in range(policies.shape[1])), index=df.index))

        values = np.asarray(values)[:, np.newaxis]
        df = df.join(pd.DataFrame(values, columns=['value'], index=df.index))

        stats_df = pd.DataFrame.from_records(stats, columns=['moves_nb', 'max_deepness', 'tree_size', 'terminal_count', "q_value"], index=df.index)
        #stats_df = stats_df.tree_size.astype(np.int64)
        df = df.join(stats_df)

        df.set_index(["generation", "game_idx", "move_idx"], inplace=True)

        return df


_env_ = None
def _worker_init(hdf_file_name, generation, nn_class, params):
    global _env_
    _env_ = params
    _env_.name = 'w%i' % mp.current_process().pid
    _env_.hdf_file_name, _env_.generation = hdf_file_name, generation
    _env_.nn_class =  nn_class

def _worker_set_device(device):
    import time
    global _env_
    _env_.nn.pytorch_device = device
    time.sleep(0.1)
    print("Worker {} uses device {}".format(_env_.name, device))

def _worker_run(games_idxs):
    global _env_
    
    import self_play
    from dots_boxes.dots_boxes_game import BoxesState
    import utils.utils as utils
    from nn import NeuralNetWrapper
    import time

    tic = time.time()
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    tasks = []

    sp_params = _env_.self_play
    nn = NeuralNetWrapper(_env_.nn_class(_env_), _env_)
    nn.load_model_parameters(_env_.generation-1)
    nn = AsyncBatchedProxy(nn, batch_size=sp_params.nn_batch_size, timeout=sp_params.nn_batch_timeout, batch_builder=sp_params.nn_batch_builder)
    tasks.append(asyncio.ensure_future(nn.run(), loop=loop))

    try:
        sp = self_play.SelfPlay(nn, _env_)
        loop.run_until_complete(sp.play_games(BoxesState(), games_idxs, show_progress=False))
    except e:
        print(e, flush=True)
        raise e
    finally:
        for t in tasks:
            t.cancel()
        loop.run_until_complete(asyncio.sleep(0.001))
        loop.close()

    tac = time.time()
    df = sp.get_datasets(_env_.generation)
    toc = time.time()

    print("Worker {} played {} games ({} samples) in {:.2f}s (df gen+saving={:.3f}s)".format(
        _env_.name, len(games_idxs), len(df.index), toc-tic, toc-tac), flush=True)

    return df

def generate_games(hdf_file_name, generation, nn_class, n_games, params, n_workers=None, devices=["cuda:1", "cuda:0"], games_per_workers=10):
    nw = n_workers if n_workers else mp.cpu_count() - 1
    gpw = games_per_workers if games_per_workers else max(1, n_games//nw)
    games_idxs = np.array_split(np.arange(n_games), n_games//gpw)
    nw = min(nw, len(games_idxs))

    def save_df(df):
        utils.write_to_hdf(hdf_file_name, df)
    def err_cb(err):
        raise err
    with mp.Pool(nw, initializer=_worker_init, initargs=(hdf_file_name, generation, nn_class, params)) as pool:
        pool.map(_worker_set_device, list(devices[i%len(devices)] for i in range(nw)))
        tasks = [pool.apply_async(_worker_run, (idxs,), callback=save_df, error_callback=err_cb) for idxs in games_idxs]
        [t.wait() for t in tasks]
        pool.close()

"""
_params = None
def _selfplay_worker_init(params):
    global _params
    _params = params
    pipe_proxy_init()

def _selfplay_worker(generation, games_idxs):
    import self_play
    from dots_boxes.dots_boxes_game import BoxesState
    import utils.proxies
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    sp = self_play.SelfPlay(utils.proxies.pipe_proxy, _params)
    loop.run_until_complete(
        sp.play_games(BoxesState(), games_idxs, show_progress=True))
    loop.close()
    return sp.get_datasets(generation)


def generate_games(hdf_file_name, generation, nn, n_games, params, n_workers=None, games_per_workers=None):
    def write_to_hdf(result):
        data, stats = result
        with pd.HDFStore(hdf_file_name, mode="a") as store:
            features = np.stack(data.features.values, axis=0)
            features = pd.DataFrame(features, columns=range(48), index=data.index)
            store.append("data/features",features, format="table")

            policies = np.stack(data.policy.values, axis=0)
            policies = pd.DataFrame(policies, columns=range(32), index=data.index)
            store.append("data/policy", pd.DataFrame(policies, columns=range(32)), format="table")
            store.append("data/value", data["value"], format="table")
            store.append("stats", stats, format="table")

    def err_cb(err):
        raise err
        
    nw = n_workers if n_workers else mp.cpu_count() - 1
    gpw = games_per_workers if games_per_workers else max(1, n_games//nw)
    games_idxs = np.array_split(np.arange(n_games), n_games//gpw)
    
    loop = asyncio.get_event_loop()
    asyncio.set_event_loop(loop)
    tasks = []

    sp_params = params.self_play
    nn = AsyncBatchedProxy(nn, batch_size=sp_params.nn_batch_size, timeout=sp_params.nn_batch_timeout, batch_builder=sp_params.nn_batch_builder)
    tasks.append(asyncio.ensure_future(nn.run(), loop=loop))

    with mp.Pool(nw, initializer=_selfplay_worker_init, initargs=(params,)) as pool:
        nn_worker = utils.proxies.PipedProxy(nn, loop, pool)
        tasks.append(asyncio.ensure_future(nn_worker.run()))
        
        try:
            for idxs in games_idxs:
                pool.apply_async(_selfplay_worker, args=(generation, list(idxs)),
                                callback=write_to_hdf, error_callback=err_cb)
                            
            pool.close()
            workers_task = loop.run_in_executor(None, pool.join)
            loop.run_until_complete(asyncio.gather(workers_task))
        except Exception:
            pool.terminate()
            raise
        finally:
            for t in tasks:
                t.cancel()
            loop.run_until_complete(asyncio.sleep(0.001))
            loop.close()
"""

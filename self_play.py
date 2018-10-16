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

import utils
from utils.proxies import PipedProxy, pipe_proxy_init, AsyncBatchedProxy
import mcts


class SelfPlay(object):

    def __init__(self, nn, params, asyncio_loop=None):
        self.played_games = []
        self.params = params.self_play
        self.loop = asyncio_loop
        self.nn = nn

    def get_next_move(self, root_node, nb_mcts_searches, temperature, dirichlet):
        visit_counts = mcts.UCT_search(root_node, nb_mcts_searches, self.nn, 
                                             self.params.mcts.mcts_cpuct, self.loop, 
                                             self.params.mcts.max_async_searches, dirichlet)
        
        # apply temperature
        probs = (visit_counts/visit_counts.max()) ** (1/temperature)
        probs = probs / (1.000001*probs.sum()) # TODO: p/p.sum() is sometimes > 1
        sampled = np.random.multinomial(1, np.append(probs, 0), 1)
        move = np.argmax(sampled)
        return move

    def play_game(self, game_state, idx):
        params = self.params
        temperature = None

        moves_sequence = []
        root_node = mcts.create_root_uct_node(game_state)
        i = -1
        while not root_node.is_terminal:
            i += 1
            if i in params.mcts.temperature:
                temperature = params.mcts.temperature[i]
            move = self.get_next_move(
                root_node, params.mcts.mcts_num_read, temperature, params.noise)
            
            moves_sequence.append(root_node)
            root_node = mcts.init_mcts_tree(root_node, move, reuse_tree=params.reuse_mcts_tree)

        moves_sequence.append(root_node) #add the terminal node

        self.played_games.append(
            (idx, moves_sequence, root_node.game_state.get_result()))

    def play_games(self, game_state, games_idxs, show_progress=False):
        for idx in games_idxs:
            if show_progress:
                print(".", end="", flush=True)
            self.play_game(game_state, idx)

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
        features = []
        policies = []
        values = []
        stats = []
        for game_idx, moves_seq, result in self.played_games:
            for move_i, node in reversed(list(enumerate(moves_seq[:-1]))):
                game_idxs.append(game_idx)
                moves_idxs.append(move_i)
                features.append(node.game_state.get_features().ravel())
                policies.append(node.child_number_visits /
                                node.child_number_visits.sum())
                values.append(result)
                stats.append(node.get_tree_stats())
                if node.game_state.player != node.game_state.next_player:
                    result = result * -1
        
        data = {"generation": [generation]*len(game_idxs), "game_idx": game_idxs, 
                "move_idx": moves_idxs, "features": features, 
                "policy": policies, "value": values}
        data_df = pd.DataFrame(data).set_index(["generation", "game_idx", "move_idx"])

        mcts_stats_df = pd.DataFrame.from_records(stats, data_df.index, columns=[
                                                  'moves_nb', 'max_deepness', 'tree_size', 'terminal_count'])
        return data_df, mcts_stats_df

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
    game_state = BoxesState()
    sp = self_play.SelfPlay(utils.proxies.pipe_proxy, _params)
    sp.play_games(game_state, games_idxs, show_progress=True)
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
    sp_params = params.self_play
    nn = AsyncBatchedProxy(nn, batch_size=sp_params.nn_batch_size, timeout=sp_params.nn_batch_timeout, loop=loop, batch_builder=sp_params.nn_batch_builder)
    asyncio.ensure_future(nn.run(), loop=loop)

    with mp.Pool(nw, initializer=_selfplay_worker_init, initargs=(params,)) as pool:
        nn_worker = utils.proxies.PipedProxy(nn, loop, pool)
        proxy_task = asyncio.ensure_future(nn_worker.run(), loop=loop)
        
        try:
            for idxs in games_idxs:
                pool.apply_async(_selfplay_worker, args=(generation, list(idxs)),
                                callback=write_to_hdf, error_callback=err_cb)
                            
            pool.close()
            workers_task = loop.run_in_executor(None, pool.join)
            loop.run_until_complete(asyncio.gather(workers_task, proxy_task))
        except Exception:
            pool.terminate()
            raise
        finally:
            loop.close()

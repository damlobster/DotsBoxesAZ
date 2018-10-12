import logging
import sys
import random
from functools import partial
import multiprocessing as mp

import numpy as np
import pandas as pd

import utils
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

    def play_game(self, game_state):
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
            (moves_sequence, root_node.game_state.get_result()))

    def play_games(self, game_state, n_iters, show_progress=False):
        for _ in range(n_iters):
            if show_progress:
                print(".", end="", flush=True)
            self.play_game(game_state)

    @DeprecationWarning
    def get_training_data(self, with_stats=True):
        features = []
        policies = []
        values = []
        stats = []
        for moves_seq, result in self.played_games:
            for node in reversed(moves_seq[:-1]):
                features.append(node.game_state.get_features())
                policies.append(node.child_number_visits /
                                node.child_number_visits.sum())
                values.append(result)
                stats.append(node.get_tree_stats())
                if node.game_state.player != node.game_state.next_player:
                    result = result * -1
                
        return features, policies, values, stats

    def get_games_moves(self):
        moves = []
        visit_counts = []
        for moves_seq, _ in self.played_games:
            for node in moves_seq[1:]:
                moves.append(node.move)
                visit_counts.append(node.child_number_visits)
        vc = np.asarray(visit_counts, dtype=float)
        return moves, vc

    def get_datasets(self, generation, start_idx=0):
        game_idxs = []
        moves_idxs = []
        features = []
        policies = []
        values = []
        stats = []
        for game_i, (moves_seq, result) in enumerate(self.played_games):
            for move_i, node in reversed(enumerate(moves_seq[:-1])):
                game_idxs.append(start_idx + game_i)
                moves_idxs.append(move_i)
                features.append(node.game_state.get_features())
                policies.append(node.child_number_visits /
                                node.child_number_visits.sum())
                values.append(result)
                stats.append(node.get_tree_stats())
                if node.game_state.player != node.game_state.next_player:
                    result = result * -1
                
        index = {"generation":generation ,"game_idx":game_idxs, "move_idx":move_idxs}
        data = {"features":features, "policy":policies, "value":values}
        data_df = pd.DataFrame(data, index)
        mcts_stats_df = pd.from_records(stats, index)
        return data_df, mcts_stats_df

def _selfplay_worker(generation, n_games, nn_proxy, params, start_idx):
    game_state = BoxesState()
    sp = SelfPlay(nn, params)
    sp.play_games(game_state, n_games, show_progress=False)
    sys.stdout.flush()
    return sp.get_datasets(generation, start_idx)

def generate_games(generation, nn_proxy, n_games, params, n_workers=None, games_per_workers=None):
    nw = mp.cpu_count()-1 if n_workers is None else n_workers
    gpw = max(1, n_games//nw) if games_per_workers is None else games_per_workers
    assert n_games % gpw == 0, "%d is not a multiple of %d" % (n_games, gpw)

    with mp.Pool(nw) as pool:
        results = [pool.apply_async(selfplay, args=(generation, nn, gpw, params, start_idx)) 
                   for start_idx in range(0, n_games, gpw)]
        datas, stats = zip(*[r.get() for r in results])

        return pd.concat(datas, copy=False), pd.concat(stats, copy=False)

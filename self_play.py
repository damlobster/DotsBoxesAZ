from nn import NeuralNetWrapper
import mcts
from utils.proxies import PipeProxyServer, PipeProxyClient
from utils.utils import DotDict, elo_rating2
import math
import pandas as pd
import numpy as np
import asyncio
import sys
import copy
import random
from functools import partial
import torch
import torch.multiprocessing as mp
import configuration
import logging
logger = logging.getLogger(__name__)


class SelfPlay(object):

    def __init__(self, players):
        self.played_games = []
        self.players = players

    async def get_next_move(self, root_node, nb_mcts_searches, temperature, dirichlet):
        nn = self.players.get_nn(root_node.game_state.next_player)
        params = self.players.get_params(root_node.game_state.next_player)
        visit_counts = await mcts.UCT_search(root_node, nb_mcts_searches, nn,
                                             params.self_play.mcts.mcts_cpuct,
                                             params.self_play.mcts.max_async_searches, dirichlet)
        # apply temperature
        probs = (visit_counts/visit_counts.max()) ** (1/temperature)
        probs = probs / probs.sum()  # TODO: p/p.sum() is sometimes > 1
        move = np.random.choice(probs.shape[0], 1, p=probs)[0]
        if(logger.isEnabledFor(logging.DEBUG)):
            msg = []
            for i in range(visit_counts.shape[0]):
                if round(probs[i], 3) > 0:
                    o = '\033[92m' if i == move else "\033[93m" if probs[move] < probs[i] else "\033[94m" if probs[i] > 0.1 else ""
                    c = '\033[0m' if i == move or probs[i] > 0 else ""
                    msg.append(f"{o}{visit_counts[i]}:{probs[i]:.3f}{c}")

            logger.debug(" ".join(msg))
            logger.debug("move= %s", move)
        return move

    async def play_game(self, game_state, idx):
        temperature = None

        moves_sequence = []
        root_node = mcts.create_root_uct_node(game_state)
        i = -1
        while not root_node.is_terminal:
            i += 1
            params = self.players.get_params(root_node.game_state.next_player)
            if i in params.self_play.mcts.temperature:
                temperature = params.self_play.mcts.temperature[i]

            nb_valid_moves = len(
                root_node.game_state.get_valid_moves(as_indices=True))
            searches = min(4*math.factorial(nb_valid_moves),
                           params.self_play.mcts.mcts_num_read)
            move = await self.get_next_move(root_node, searches, temperature, params.self_play.noise)

            moves_sequence.append(root_node)
            root_node = mcts.init_mcts_tree(
                root_node, move, reuse_tree=params.self_play.reuse_mcts_tree)

        moves_sequence.append(root_node)  # add the terminal node

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

    def get_datasets(self, with_features=True):
        game_idxs = []
        moves_idxs = []
        moves = []
        players = []
        features = []
        policies = []
        values = []
        stats = []
        for game_idx, moves_seq, z in self.played_games:
            # ! remove last move in training data
            for move_i, node in reversed(list(enumerate(moves_seq[:-1]))):
                if node.game_state.player != node.game_state.next_player:
                    z = -z  # flip sign of z if player has changed
                game_idxs.append(game_idx)
                moves_idxs.append(move_i)
                moves.append(node.move)
                players.append(node.game_state.player)
                values.append(z)
                stats.append(node.get_tree_stats())
                vs = node.child_number_visits.sum()
                policies.append(node.child_number_visits /
                                (vs if vs > 0.0 else 1.0))
                if with_features:
                    features.append(node.game_state.get_features().ravel())

        players = np.asarray(players, dtype=np.int8)
        gen = np.zeros(len(game_idxs), dtype=np.int16)
        gen[players == 0] = self.players.get_generation(0)
        gen[players == 1] = self.players.get_generation(1)

        df = pd.DataFrame(gen, columns=['generation'])
        games_idxs = np.asarray(game_idxs, dtype=np.int16)
        df = df.join(pd.DataFrame(games_idxs, columns=[
                     'game_idx'], index=df.index))
        moves_idxs = np.asarray(moves_idxs, dtype=np.int16)
        df = df.join(pd.DataFrame(moves_idxs, columns=[
                     'move_idx'], index=df.index))

        moves = np.asarray(moves)
        moves[moves == None] = -1
        moves = moves.astype(np.int16)
        df = df.join(pd.DataFrame(moves, columns=['move'], index=df.index))

        df = df.join(pd.DataFrame(players, columns=['player'], index=df.index))

        if with_features:
            features = np.stack(features, axis=0)
            df = df.join(pd.DataFrame(features, columns=list("x_"+str(i)
                                                             for i in range(features.shape[1])), index=df.index))

        policies = np.stack(policies, axis=0)
        df = df.join(pd.DataFrame(policies, columns=list("pi_"+str(i)
                                                         for i in range(policies.shape[1])), index=df.index))

        values = np.asarray(values)[:, np.newaxis]
        df = df.join(pd.DataFrame(values, columns=['z'], index=df.index))

        stats_df = pd.DataFrame.from_records(stats, columns=[
                                             'max_deepness', 'tree_size', 'terminal_count', 'q_value'], index=df.index)
        df = df.join(stats_df.astype(
            {'max_deepness': np.int16, 'tree_size': np.int32, 'terminal_count': np.int32, 'q_value': np.float32}))

        df.set_index(["generation", "game_idx", "move_idx"], inplace=True)

        return df


def _fut_cb(fut):
    if fut.exception():
        print(fut.exception(), flush=True)
        raise fut.exception()


class Players:
    def __init__(self):
        self.players = []

    def add_player(self, generation, nn, params):
        self.players.append((generation, nn, params))

    def switch_players(self):
        self.players = list(reversed(self.players))

    def get_generation(self, id):
        return self.players[id if len(self.players) > 1 else 0][0]

    def get_nn(self, id):
        return self.players[id if len(self.players) > 1 else 0][1]

    def get_params(self, id):
        return self.players[id if len(self.players) > 1 else 0][2]


class SelfPlayProcess(mp.Process):

    def __init__(self, process_id, games_idxs, hdf_file_name, lock, players):
        super(SelfPlayProcess, self).__init__()
        self.process_id = process_id
        self.games_idxs = games_idxs
        self.hdf_file_name = hdf_file_name
        self.lock = lock
        self.players = players

    def run(self):
        import multiprocessing as mp
        import time
        import self_play
        from dots_boxes.dots_boxes_game import BoxesState
        from utils.utils import write_to_hdf

        loop = asyncio.get_event_loop()

        try:
            sp = self_play.SelfPlay(self.players)

            for idxs in self.games_idxs:
                tick = time.time()
                loop.run_until_complete(sp.play_games(
                    BoxesState(), idxs, show_progress=False))
                df = sp.get_datasets(with_features=True)
                df["training"] = np.zeros(len(df.index), dtype=np.int8)

                tack = time.time()
                with self.lock:
                    write_to_hdf(self.hdf_file_name, "fresh", df)

                tock = time.time()
                logger.info("Worker %s played %d games (%d samples) in %.0fs (save=%.3fs)",
                            self.process_id, len(idxs), len(df.index), tock-tick, tock-tack)

        except Exception as e:
            print(e, flush=True)
            raise e


def generate_games(hdf_file_name, generation, model, n_games, params, n_workers=None, games_batch_size=10):
    nw = n_workers if n_workers else mp.cpu_count() - 1
    gpw = games_batch_size if games_batch_size else max(1, n_games//nw)
    games_idxs = np.array_split(np.arange(n_games), nw)

    lock = mp.Lock()

    nn = NeuralNetWrapper(model, params)
    proxy = PipeProxyServer(lambda args: nn.predict_sync(
        torch.cat(list(g.get_features() for g in args[0]))))

    processes = []
    for process_id, idxs in enumerate(games_idxs):
        player = Players()
        player.add_player(0, generation, proxy.create_client(), params)
        process = SelfPlayProcess(process_id, np.array_split(idxs, len(idxs)/games_batch_size),
                                  hdf_file_name, lock, player)
        processes.append(process)

    for process in processes:
        process.start()

    while len(mp.connection.wait(processes, 0)) < nw:
        proxy.handle_requests()


def compute_elo(elo_params, models, params, generations, elos):
    nw = elo_params.n_workers if elo_params.n_workers else mp.cpu_count() - 2
    gpw = elo_params.games_per_workers if elo_params.games_per_workers else max(
        1, elo_params.n_games//nw)
    games_idxs = np.array_split(np.arange(elo_params.n_games), nw)

    lock = mp.Lock()

    params = copy.deepcopy(params)
    params[0].self_play.merge(elo_params.self_play_override)
    params[1].self_play.merge(elo_params.self_play_override)

    players = Players()
    proxies = []
    for i in range(len(models)):
        p = params[i].self_play.merge(elo_params.self_play_override)
        nn = NeuralNetWrapper(models[i], params[i])
        proxy = PipeProxyServer(lambda args: nn.predict_sync(
            torch.cat(list(g.get_features() for g in args[0]))))
        players.add_player(generations[i], proxy.create_client(), p)
        proxies.append(proxy)

    processes = []
    for process_id, idxs in enumerate(games_idxs):
        players.switch_players()
        process = SelfPlayProcess(process_id, np.array_split(idxs, len(idxs)/gpw),
                                  elo_params.hdf_file, lock, players)
        processes.append(process)

    for process in processes:
        process.start()

    while len(mp.connection.wait(processes, 0)) < nw:
        proxies[0].handle_requests()
        proxies[1].handle_requests()

    # compute new elo score
    with pd.HDFStore(elo_params.hdf_file, mode="a") as store:
        games = store.get("/fresh")
        del store["/fresh"]
        store.put("elo{}vs{}".format(*generations),
                  games, format="table", append=False)

    winners = games[games.z == 1].sort_index(
        level=["move_idx"]).groupby(level=["game_idx"]).head(1)
    n0 = sum(winners.index.get_level_values("generation") == generations[0])
    n1 = len(winners)-n0
    elo0, elo1 = elo_rating2(elos[0], elos[1], n0, n1, K=30)

    print(
        f"{params[0].nn.model_class.__name__} generation {generations[0]}: wins={n0}, elo={elos[0]} -> {elo0}")
    print(
        f"{params[1].nn.model_class.__name__} generation {generations[1]}: wins={n1}, elo={elos[1]} -> {elo1}")

    return elo0, elo1, n1/len(winners)

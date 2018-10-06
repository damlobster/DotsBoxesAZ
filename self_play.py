import numpy as np
import random
import utils
import mcts
import sys


class SelfPlay(object):

    def __init__(self, nn, params):
        self.nn = nn
        self.played_games = []
        self.params = params.self_play

    async def get_next_move(self, root_node, nb_mcts_searches, temperature, dirichelet):
        visit_counts = await mcts.UCT_search_async(
            root_node, nb_mcts_searches, self.nn, self.params.mcts.mcts_cpuct)
        probs = (visit_counts/visit_counts.max()) ** (1/temperature)

        alpha, coeff = dirichelet
        probs = probs / probs.sum()
        valid_actions = root_node.game_state.get_valid_moves()
        valid_actions[valid_actions == 0] = 1e-60
        noise = np.random.dirichlet(valid_actions*alpha, 1).ravel()
        new_probs = (1-coeff)*probs + coeff*noise

        move = np.argmax(np.random.multinomial(
            1, new_probs / new_probs.sum(), 1))
        return move

    async def play_game(self, game_state):
        params = self.params
        temperature = None

        moves_sequence = []
        root_node = mcts.create_root_uct_node(game_state)
        i = -1
        while not root_node.is_terminal:
            i += 1
            if i in params.mcts.temperature:
                temperature = params.mcts.temperature[i]
            move = await self.get_next_move(
                root_node, params.mcts.mcts_num_read, temperature, params.noise)
            moves_sequence.append(root_node)
            next_node = None
            if params.reuse_mcts_tree:
                next_node = root_node.children[move]
                del(root_node.children)
            else:
                next_node = mcts.create_root_uct_node(
                    root_node.children[move].game_state)
            root_node = next_node

        moves_sequence.append(root_node)
        self.played_games.append(
            (moves_sequence, root_node.game_state.get_result()))

    async def play_games(self, game_state, n_iters, show_progress=False):
        for _ in range(n_iters):
            if show_progress:
                print(".", end="", flush=True)
            await self.play_game(game_state)

    def get_training_data(self):
        features = []
        policies = []
        values = []
        for moves_seq, result in self.played_games:
            for node in reversed(moves_seq[:-1]):
                features.append(node.game_state.get_features())
                policies.append(node.child_number_visits /
                                node.child_number_visits.sum())
                values.append(result)
                if node.game_state.player != node.game_state.next_player:
                    result = result * -1

        return list(zip(features, policies, values))

    def get_games_moves(self):
        moves = []
        visit_counts = []
        for moves_seq, _ in self.played_games:
            for node in moves_seq[1:]:
                moves.append(node.move)
                visit_counts.append(node.child_number_visits)
        vc = np.asarray(visit_counts, dtype=float)
        return moves, vc

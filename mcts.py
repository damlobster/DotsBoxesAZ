from utils.utils import DictWithDefault
from game import GameState
from collections import namedtuple
import numpy as np
import time
import math
from functools import partial
import collections
import asyncio
import logging
logger = logging.getLogger(__name__)


tree_stats_enabled = True
TreeStats = namedtuple(
    'TreeStats', ['max_deepness', 'tree_size', 'terminal_count', 'q_value'])

VIRTUAL_LOSS = 1


class TreeRoot():
    def __init__(self, game_state):
        self.first_node = None
        self.child_total_value = collections.defaultdict(float)
        self.child_number_visits = collections.defaultdict(int)
        self.deepness_correction = 0
        self.deepness = 0
        self.max_deepness = 0
        self.terminal_states_count = 0
        self.tree_size = 0

    def get_tree_stats(self):
        max_deepness = self.max_deepness - self.deepness_correction
        q = self.first_node.total_value / (1 + self.first_node.number_visits)
        return TreeStats(max_deepness, int(self.tree_size), self.terminal_states_count, q if isinstance(q, float) else q[0])


class UCTNode():
    __slots__ = ['game_state', 'move', 'is_expanded', 'parent', 'children',
                 'child_priors', 'child_total_value', 'child_number_visits',
                 'is_terminal', "deepness"]

    CPUCT = 1.25
    CPUCT_BASE = 19652

    def __init__(self, game_state: GameState, move: int, parent):
        self.game_state = game_state
        self.move = move
        self.parent = parent
        self.is_expanded = False
        self.is_terminal = game_state.get_result() is not None
        self.children = DictWithDefault(lambda move: UCTNode(
            self.game_state.play(move), move, parent=self))
        self.child_priors = np.zeros(
            game_state.get_actions_size(), dtype=np.float32)
        self.child_total_value = np.zeros(
            game_state.get_actions_size(), dtype=np.float32)
        self.child_number_visits = np.zeros(
            game_state.get_actions_size(), dtype=np.int32)

        if tree_stats_enabled:
            self.deepness = parent.deepness + 1

    @property
    def number_visits(self):
        return self.parent.child_number_visits[self.move]

    @number_visits.setter
    def number_visits(self, value):
        self.parent.child_number_visits[self.move] = value

    @property
    def total_value(self):
        return self.parent.child_total_value[self.move]

    @total_value.setter
    def total_value(self, value):
        self.parent.child_total_value[self.move] = value

    def children_ucb_score(self):
        pb_c = math.log((self.number_visits + UCTNode.CPUCT_BASE +
                         1) / UCTNode.CPUCT_BASE) + UCTNode.CPUCT
        pb_c *= math.sqrt(self.number_visits) / (self.child_number_visits + 1)

        prior_score = pb_c * self.child_priors
        value_score = self.child_total_value / (1 + self.child_number_visits)
        return prior_score + value_score

    def best_child(self):
        invalid_moves = 1 - self.game_state.get_valid_moves()
        return np.argmax(-1e12*invalid_moves + self.children_ucb_score())

    def select_leaf(self):
        current = self
        search_path = [current]
        while current.is_expanded and not current.is_terminal:
            current.total_value -= VIRTUAL_LOSS
            best_move = current.best_child()
            current = current.children[best_move]
            search_path.append(current)

        return current, search_path

    def expand(self, child_priors):
        self.is_expanded = True
        self.child_priors = child_priors

    def backup(self, search_path: list, value_estimate: float):
        to_play = self.game_state.to_play
        for i, node in enumerate(search_path):
            v = value_estimate if node.game_state.just_played == to_play else -value_estimate
            node.total_value += v + VIRTUAL_LOSS
            node.number_visits += 1

        if tree_stats_enabled:
            # root is the TreeRoot, self is the leaf down the tree
            root = search_path[0].parent
            root.terminal_states_count += self.is_terminal
            root.max_deepness = max(root.max_deepness, self.deepness)

    def get_tree_stats(self):
        assert isinstance(
            self.parent, TreeRoot), "Must be called on the first node of the tree"
        self.parent.number_visits = self.number_visits
        return self.parent.get_tree_stats()

    def __repr__(self):
        string = []
        string.append("*"*15)
        string.append("Node: " + str(self.game_state.hash))
        string.append("Move: " + str(self.move))
        string.append("#visits: " + str(self.number_visits))
        string.append("Expanded: " + str(self.is_expanded))
        string.append("Terminal: " + str(self.is_terminal))
        string.append("Total value: " + str(self.total_value))
        string.append(str(self.game_state))
        return "\n".join(string)

    def __hash__(self):
        return self.game_state.__hash__()


def create_root_uct_node(game_state):
    root = TreeRoot(game_state)
    node = UCTNode(game_state, move=None, parent=root)
    root.first_node = node
    return node


def init_mcts_tree(previous_node, move, reuse_tree=True):
    next_node = None

    if reuse_tree:
        next_node = previous_node.children[move]
        nb_visits = previous_node.child_number_visits[move]
        root = TreeRoot(previous_node.game_state)
        next_node.parent = root
        root.first_node = next_node
        root.deepness_correction = next_node.deepness
        root.tree_size = nb_visits
        del previous_node.children
    else:
        next_node = create_root_uct_node(
            previous_node.children[move].game_state)
        next_node.move = move

    return next_node


async def UCT_search(root_node: UCTNode, num_reads, async_nn, cpuct=(1.25, 19652), max_pending_evals=64, dirichlet=(0.0, 0.0), time_limit=None):
    async def _search():
        leaf, search_path = root_node.select_leaf()
        if not leaf.is_terminal:
            child_priors, value_estimate = await async_nn(leaf.game_state)
            # mask invalid moves and renormalize
            child_priors = child_priors * \
                leaf.game_state.get_valid_moves(as_indices=False)
            priors_sum = child_priors.sum()
            if priors_sum > 0 and priors_sum != 1.0:
                child_priors /= priors_sum
        else:
            child_priors, value_estimate = np.zeros(
                leaf.game_state.get_actions_size()), leaf.game_state.get_result()

        leaf.expand(child_priors)
        leaf.backup(search_path, value_estimate)

    end_time = time.time() + 120  # by default 2 minutes time limit
    if time_limit:
        end_time = time.time() + time_limit

    UCTNode.CPUCT, UCTNode.CPUCT_BASE = cpuct

    if not root_node.is_expanded:
        await _search()

    # add noise to root_node.child_priors
    alpha, coeff = dirichlet

    cpsum = root_node.child_priors.sum()
    if cpsum != 0:
        probs = root_node.child_priors / root_node.child_priors.sum()
    else:
        probs = np.zeros(len(root_node.child_priors))

    if alpha > 0:
        valid_actions = root_node.game_state.get_valid_moves()
        valid_actions[valid_actions == 0] = 1e-60
        noise = np.random.dirichlet(valid_actions*alpha, 1).ravel()
        noise *= root_node.game_state.get_valid_moves()
    else:
        noise = 0.0
    root_node.child_priors = (1-coeff)*probs + coeff*noise

    max_pend = min(max_pending_evals, len(
        root_node.game_state.get_valid_moves()))
    pending = set()
    for i in range(num_reads):
        if time.time() > end_time:
            break
        if len(pending) >= max_pend:
            _, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
            max_pend = max_pending_evals
        fut = asyncio.ensure_future(_search())
        fut.add_done_callback(_err_cb)
        pending.add(fut)

    if len(pending) > 0:
        done, pending = await asyncio.wait(pending)

    return root_node.child_number_visits


def _err_cb(fut):
    if fut.exception():
        print(fut.exception(), flush=True)
        raise fut.exception()


def print_mcts_tree(node: UCTNode, max_level=10, prefix=" "):
    if max_level < 0:
        return

    def get_3_max(arr, min=False):
        _min = max(arr) == 0
        s = []
        for idx in np.argsort(arr)[::1 if _min else -1][:3]:
            s.append(f"{idx}->{arr[idx]:.4f}")
        return "; ".join(s)

    gs = node.game_state
    print(
        f"{prefix[:-1]}{node.move} ({node.number_visits}/{node.total_value}) -> {gs.to_play} {gs.get_result()}")
    print(f"{prefix[:-1]} - child values:{get_3_max(node.child_total_value)}")
    print(f"{prefix[:-1]} - child visits:{get_3_max(node.child_number_visits)}")
    print(f"{prefix[:-1]} - priors:{get_3_max(node.child_priors)}")
    print(f"{prefix[:-1]} - ucb:{get_3_max(node.children_ucb_score())}")
    for n in node.children.values():
        print_mcts_tree(n, max_level-1, prefix + "   |")

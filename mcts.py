import asyncio
import collections
from functools import partial
import math
import numpy as np
from collections import namedtuple

from game import GameState
from utils.utils import DictWithDefault

tree_stats_enabled = True
TreeStats = namedtuple('TreeStats', ['max_deepness', 'tree_size', 'terminal_count', 'q_value'])

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

    CPUCT = 1.0

    def __init__(self, game_state: GameState, move: int, parent):
        self.game_state = game_state
        self.move = move
        self.parent = parent
        self.is_expanded = False
        self.is_terminal = game_state.get_result() is not None
        self.children = DictWithDefault(lambda move: UCTNode(self.game_state.play(move), move, parent=self))
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

    def child_Q(self):
        return self.child_total_value / (1 + self.child_number_visits)

    def child_U(self):
        return math.sqrt(self.number_visits) * (
            self.child_priors / (1 + self.child_number_visits))

    def best_child(self):
        invalid_moves = 1 - self.game_state.get_valid_moves()
        return np.argmax(-1e12*invalid_moves + self.child_Q() + UCTNode.CPUCT * self.child_U())

    def select_leaf(self):
        current = self
        while current.is_expanded and not current.is_terminal:
            current.number_visits += 1
            current.total_value -= 1
            best_move = current.best_child()
            current = current.children[best_move]
        current.number_visits += 1
        current.total_value -= 1
        return current
    
    def expand(self, child_priors):
        if not self.is_terminal:
            self.is_expanded = True
            self.child_priors = child_priors

    def backup(self, value_estimate: float):
        current = self
        v = value_estimate
        while not isinstance(current, TreeRoot):
            v = v if current.game_state.player == current.game_state.next_player else -v
            current.total_value += v + 1
            current = current.parent
        
        if tree_stats_enabled:
            # current is the TreeRoot: update tree stats
            # self is the leaf down the tree
            current.terminal_states_count += self.is_terminal
            current.max_deepness = max(current.max_deepness, self.deepness)

    def get_tree_stats(self):
        assert isinstance(self.parent, TreeRoot), "Must be called on the first node of the tree"
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
        next_node = mcts.create_root_uct_node(
            root_node.children[move].game_state)
        next_node.move = nove
    
    return next_node


async def UCT_search(root_node: UCTNode, num_reads, async_nn, cpuct=1.0, max_pending_evals=8, dirichlet=(1.0, 0.25)):
    import multiprocessing as mp
    async def _search():
        leaf = root_node.select_leaf()
        if not leaf.is_terminal: 
            child_priors, value_estimate = await async_nn(leaf.game_state)
        else:
            child_priors, value_estimate = np.zeros(
                leaf.game_state.get_actions_size()), leaf.game_state.get_result()

        leaf.expand(child_priors *
                    leaf.game_state.get_valid_moves(as_indices=False))
        leaf.backup(value_estimate)

    UCTNode.CPUCT = cpuct

    if not root_node.is_expanded:
        await _search()
        
    # add noise to root_node.child_priors
    alpha, coeff = dirichlet
    
    cpsum = root_node.child_priors.sum()
    if cpsum != 0:
        probs = root_node.child_priors / root_node.child_priors.sum()
    else:
        probs = np.zeros(len(root_node.child_priors))

    valid_actions = root_node.game_state.get_valid_moves()
    valid_actions[valid_actions == 0] = 1e-60
    noise = np.random.dirichlet(valid_actions*alpha, 1).ravel()
    noise *= root_node.game_state.get_valid_moves()
    root_node.child_priors = (1-coeff)*probs + coeff*noise

    max_pend = min(max_pending_evals, len(root_node.game_state.get_valid_moves()))
    pending = set()
    for i in range(num_reads):
        if len(pending) >= max_pend:
            _, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
            max_pend = max_pending_evals
        pending.add(asyncio.ensure_future(_search()))

    if len(pending) > 0:
        done, pending = await asyncio.wait(pending)
 
    return root_node.child_number_visits

def print_mcts_tree(root_node: UCTNode, prefix=""):
    print("{}{} -> {}".format(prefix, root_node.move, root_node.game_state.hash))
    for node in root_node.children.values():
        print_mcts_tree(node, prefix + "    ")

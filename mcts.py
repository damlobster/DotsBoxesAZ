import asyncio
import collections
from functools import partial
import math
import numpy as np

from game import GameState
from utils import DictWithDefault


class DummyNode():
    def __init__(self):
        self.parent = None
        self.child_total_value = collections.defaultdict(float)
        self.child_number_visits = collections.defaultdict(float)


class UCTNode():
    __slots__ = ('game_state', 'move', 'is_expanded', 'parent', 'children',
                 'child_priors', 'child_total_value', 'child_number_visits', 'is_terminal')

    CPUCT = 1.0

    def __init__(self, game_state: GameState, move: int, parent=None):
        self.game_state = game_state
        self.move = move
        self.is_expanded = False
        self.is_terminal = game_state.get_result() is not None
        self.parent = parent
        self.children = DictWithDefault(lambda move: UCTNode(self.game_state.play(move), move, parent=self))
        self.child_priors = np.zeros(
            [game_state.get_actions_size()], dtype=np.float32)
        self.child_total_value = np.zeros(
            [game_state.get_actions_size()], dtype=np.float32)
        self.child_number_visits = np.zeros(
            [game_state.get_actions_size()], dtype=np.float32)

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

    # def maybe_add_child(self, move):
    #     if move not in self.children:
    #         self.children[move] = UCTNode(
    #             self.game_state.play(move), move, parent=self)
    #     return self.children[move]
    
    def expand(self, child_priors):
        if not self.is_terminal:
            self.is_expanded = True
            self.child_priors = child_priors

    def backup(self, value_estimate: float):
        current = self
        v = value_estimate
        while current.parent is not None:
            v = v if current.game_state.player == current.game_state.next_player else -v
            current.total_value += v + 1
            current = current.parent

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
    return UCTNode(game_state, move=None, parent=DummyNode())


def UCT_search(root_node: UCTNode, num_reads, nn, cpuct=1.0):
    UCTNode.CPUCT = cpuct
    root_node.parent = DummyNode()
    for _ in range(num_reads):
        leaf = root_node.select_leaf()
        if not leaf.is_terminal:
            child_priors, value_estimate = nn(leaf.game_state) ###### !!!!!!!!!
        else:
            child_priors, value_estimate = np.zeros(
                leaf.game_state.get_actions_size()), leaf.game_state.get_result()

        leaf.expand(child_priors *
                    leaf.game_state.get_valid_moves(as_indices=False))
        leaf.backup(value_estimate)

    return root_node.child_number_visits


async def UCT_search_async(root_node: UCTNode, num_reads, nn, cpuct=1.0):
    async def async_callback(leaf, pv):
        priors, value_estimate = pv
        leaf.expand(priors * leaf.game_state.get_valid_moves(as_indices=False))
        leaf.backup(value_estimate)

    UCTNode.CPUCT = cpuct
    root_node.parent = DummyNode()
    for _ in range(num_reads):
        leaf = root_node.select_leaf()
        if not leaf.is_terminal:
            await nn(leaf, partial(async_callback, leaf))
        else:
            await async_callback(leaf, (np.zeros(leaf.game_state.get_actions_size()),
                           leaf.game_state.get_result()))
    return root_node.child_number_visits

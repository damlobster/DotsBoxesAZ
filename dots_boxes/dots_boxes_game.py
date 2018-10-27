import logging
import copy
import math
import random
import numpy as np

from game import GameState

class BoxesState(GameState):
    __slots__ = 'hash', 'board', 'player', 'next_player', 'boxes_to_close'
    BOARD_DIM = (3, 3)
    FEATURES_SHAPE = (3, BOARD_DIM[0]+1, BOARD_DIM[1]+1)
    NB_ACTIONS = 2 * (BOARD_DIM[0]+1) * (BOARD_DIM[1]+1)
    NB_BOXES = BOARD_DIM[0] * BOARD_DIM[1]

    @staticmethod
    def init_static_fields(dims):
        BoxesState.BOARD_DIM = dims
        BoxesState.FEATURES_SHAPE = (3, \
            BoxesState.BOARD_DIM[0]+1, BoxesState.BOARD_DIM[1]+1)
        BoxesState.NB_ACTIONS = 2 * \
            (BoxesState.BOARD_DIM[0]+1) * (BoxesState.BOARD_DIM[1]+1)
        BoxesState.NB_BOXES = BoxesState.BOARD_DIM[0] * BoxesState.BOARD_DIM[1]

    def __init__(self):
        l, c = BoxesState.BOARD_DIM
        self.hash = (0, 0)
        self.board = np.zeros((2, l+1, c+1), dtype=np.uint8)
        self.board[1, l, :] = 1
        self.board[0, :, c] = 1
        self.player = 0
        self.next_player = 0
        win_thres = BoxesState.NB_BOXES/2
        self.boxes_to_close = [win_thres, win_thres]

    def get_actions_size(self):
        return self.NB_ACTIONS

    def get_valid_moves(self, as_indices=False):
        m = self.board.ravel() == 0
        if as_indices:
            return np.argwhere(m).ravel().tolist()
        else:
            return m

    def get_result(self):
        if self.player is None:
            return None

        if self.boxes_to_close[0] == 0 and self.boxes_to_close[1] == 0:
            return 0
        if self.boxes_to_close[self.player] < 0:
            return 1
        elif self.boxes_to_close[1-self.player] < 0:
            return -1
        else:
            return None

    def play_(self, move):
        p, l, c = np.unravel_index(move, self.board.shape)
        if self.board[p, l, c] != 0:
            raise ValueError("Illegal move: " + str(move) + "->" + str((p, l, c)) + "\n" + str(self))

        self.board[p, l, c] = 255

        closed_idx = []
        if p == 0:  # horizontal edge
            if l > 0 and self._check_box(l-1, c):
                closed_idx.append((l-1, c))
            if l < self.board.shape[1] - 1 and self._check_box(l, c):
                closed_idx.append((l, c))
        else:  # vertical edge
            if c > 0 and self._check_box(l, c-1):
                closed_idx.append((l, c-1))
            if c < self.board.shape[2] - 1 and self._check_box(l, c):
                closed_idx.append((l, c))

        self.player = self.next_player
        if len(closed_idx) == 0:
            self.next_player = 1 - self.player
        else:
            self.boxes_to_close[self.player] -= len(closed_idx)

        self._update_hash(move)
        return closed_idx

    def play(self, move):
        new_state = copy.deepcopy(self)
        new_state.play_(move)
        return new_state

    def get_features(self):
        board = self.board//255
        boxes_to_close = np.full_like(
            board[0], self.boxes_to_close[self.player]*2, dtype=np.int8)
        return np.concatenate((board, np.expand_dims(boxes_to_close, 0)), axis=0)

    def _check_box(self, l, c):
        edges_idx = ((0, 0, 1, 1), (l, l+1, l, l), (c, c, c, c+1))
        return self.board[edges_idx].sum()==4*255

    def _update_hash(self, move):
        b, _ = self.hash
        b += 1 << move
        self.hash = (b, self.boxes_to_close[self.next_player])

    def get_hash(self):
        return self.hash
        
    def __hash__(self):
        return self.hash.__hash__()

    def __eq__(self, other):
        return self.hash == other.hash

    def __repr__(self):
        b = self.board
        strings = []
        _, lines, cols = b.shape
        strings.append("-"*30)
        strings.append("Player = " + str(self.player))
        strings.append("Next player = " + str(self.next_player))
        strings.append("Boxes to close = " + str(self.boxes_to_close))
        strings.append("Result = " + str(self.get_result()))
        for l in range(lines):
            s = "+"
            for c in range(cols-1):
                if b[0, l, c] == 255:
                    s += "---+"
                else:
                    s += "   +"
            strings.append(s)
            s = ''
            if(l < lines-1):
                for c in range(cols):
                    if b[1, l, c] == 255:
                        s += "|   "
                    else:
                        s += "    "
            strings.append(s)
        return "\n".join(strings)

def nn_batch_builder(*game_states):
    return np.stack([gs[0].get_features() for gs in game_states], axis=0)

def moves_to_string(moves, visits_counts=None):
    g = BoxesState()
    boxes = [[" " for _ in range(BoxesState.BOARD_DIM[1])]
             for _ in range(BoxesState.BOARD_DIM[0])]
    vc = None
    if visits_counts is not None:
        sum = visits_counts.sum()
        vc = visits_counts if sum == 0 else visits_counts/sum

    for m in moves:
        just_closed_boxes = g.play_(m)
        for l, c in just_closed_boxes:
            boxes[l][c] = "0" if g.player == 0 else "1"

    b = g.board
    strings = []
    _, lines, cols = g.board.shape
    strings.append("-"*30)
    strings.append("Player = " + str(g.player))
    strings.append("Next player = " + str(g.next_player))
    strings.append("Boxes to close = " + str(g.boxes_to_close))
    strings.append("Result = " + str(g.get_result()))
    for l in range(lines):
        s = "+"
        for c in range(cols-1):
            if b[0, l, c] == 255:
                s += "---+"
            else:
                if vc is None:
                    s += "   +"
                else:
                    v = vc[np.ravel_multi_index((0, l, c), b.shape)]
                    count = 0 if np.isnan(v) else int(math.floor(10*v))
                    s += " {} +".format(count if count != 0 else " ")

        strings.append(s.ljust(30))
        s = ''
        if(l < lines-1):
            for c in range(cols):
                if b[1, l, c] == 255:
                    if c < cols-1:
                        s += "| " + boxes[l][c] + " "
                    else:
                        s += "|   "
                else:
                    if vc is None:
                        s += "    "
                    else:
                        v = vc[np.ravel_multi_index((1, l, c), b.shape)]
                        count = 0 if np.isnan(v) else int(math.floor(10*v))
                        s += "{}   ".format(count if count != 0 else " ")
        strings.append(s.ljust(30))
    return "\n".join(strings)

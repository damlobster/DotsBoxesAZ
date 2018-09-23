from game import GameState
import copy
import math
import numpy as np
import random

class BoxesState(GameState):
  __slots__ = 'hash', 'nb_boxes', 'board', 'player', 'next_player', 'boxes_to_close'

  def __init__(self, board_dim=(3, 3)):
    l, c = board_dim
    self.hash = (0, 0)
    self.nb_boxes = l*c
    self.board = np.zeros((2, l+1, c+1))
    self.board[1, l, :] = 1e-12
    self.board[0, :, c] = 1e-12
    self.player = None
    self.next_player = 0
    win_thres = self.nb_boxes/2
    self.boxes_to_close = [win_thres, win_thres]

  def get_actions_size(self):
    p, l, c = self.board.shape
    return p*l*c  # - (l+1) - (c+1)

  def get_valid_moves(self, as_indices=False):
    m = self.board.ravel() == 0.0
    if as_indices:
      return np.argwhere(m).ravel().tolist()
    else:
      return m.astype(float)

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
      raise ValueError("Illegal move: " + str(move) + "->" +
                       str((p, l, c)) + "\n" + str(self))

    self.board[p, l, c] = 1

    closed_idx = []
    if p == 0:  # horizontal edge
      if l > 0 and self._check_box(l-1, c) > 0:
        closed_idx.append((l-1, c))
      if l < self.board.shape[1] - 1 and self._check_box(l, c) > 0:
        closed_idx.append((l, c))
    else:  # vertical edge
      if c > 0 and self._check_box(l, c-1) > 0:
        closed_idx.append((l, c-1))
      if c < self.board.shape[2] - 1 and self._check_box(l, c) > 0:
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

  def _check_box(self, l, c):
    edges_idx = ((0, 0, 1, 1), (l, l+1, l, l), (c, c, c, c+1))
    return math.floor(self.board[edges_idx].sum()/4)

  def _update_hash(self, move):
    b, _ = self.hash
    b += 1 << move
    self.hash = (b, self.boxes_to_close[self.next_player])

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
        if b[0, l, c] == 1:
          s += "---+"
        else:
          s += "   +"
      strings.append(s)
      s = ''
      if(l < lines-1):
        for c in range(cols):
          if b[1, l, c] == 1:
            s += "|   "
          else:
            s += "    "
      strings.append(s)
    return "\n".join(strings)

  @staticmethod
  def moves_to_string(moves, board_dim=(3, 3)):
    g = BoxesState(board_dim)
    boxes = [[" " for _ in range(board_dim[1])] for _ in range(board_dim[0])]

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
        if b[0, l, c] == 1:
          s += "---+"
        else:
          s += "   +"
      strings.append(s)
      s = ''
      if(l < lines-1):
        for c in range(cols):
          if b[1, l, c] == 1:
            if c < cols-1:
              s += "| " + boxes[l][c] + " "
            else:
              s += "|   "
          else:
            s += "    "
      strings.append(s)
    return "\n".join(strings)

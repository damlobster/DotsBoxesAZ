class GameState(object):
  __slots__ = []
  def get_actions_size(self):
    pass

  def get_valid_moves(self, as_indices=False):
    pass
  
  def get_result(self):
    pass

  def play_(self, move):
    pass

  def play(self, move):
    pass

  def get_features(self):
    pass

  def __hash__(self):
    pass